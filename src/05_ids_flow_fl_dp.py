"""
05_ids_flow_fl_dp.py — flow-level IDS (NFStream) with FL (Flower) + DP (Opacus) on CVE5G
==========================================================================================
Trains and evaluates the IDS on the three CVE5G scenarios (one client per scenario) in
the configurations Central | Central+clip | Central+DP | FL | FL+clip | FL+DP (fixed σ)
| FL+DP (target ε, σ chosen per client). For each configuration and seed it reports
detection (macro F1, FAR, DR), the decision with a threshold calibrated on the holdout,
the per-client ε (PRV accountant), a loss-threshold MIA, and the training cost.

TO RUN:
  python 05_ids_flow_fl_dp.py --runs 5 --sigmas 0.5 1 2 --target-eps 2 8
  python 05_ids_flow_fl_dp.py --model cnn --first-seed 44 --runs 3 --sigmas 0.5 1 2
  python 05_ids_flow_fl_dp.py --model mlp --mu 0.01 --attacker-ips 192.168.70.200
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_COL = 'bidirectional_first_seen_ms'


# =============================== CONFIG =====================================
@dataclass
class Config:
    # DP
    max_grad_norm: float = 1.0
    delta: float = 1e-6
    # training
    num_rounds: int = 10
    local_epochs: int = 2
    batch_size: int = 64
    lr: float = 1e-3
    mu: float = 0.0
    use_class_weight: bool = True
    max_class_weight: float = 20.0
    seed: int = 42
    split: str = 'time'            # 'time' | 'random'
    test_frac: float = 0.25
    holdout_frac: float = 0.20
    max_per_class: int = 10_000
    min_class_flows: int = 50
    log1p: bool = True
    drop_cve: bool = True
    background_as_benign: bool = False
    attacker_ips: Tuple[str, ...] = ()
    drop_ip_prefixes: Tuple[str, ...] = ()
    model_type: str = 'mlp'
    calib: str = 'youden'          # 'youden' | 'far'
    target_far: float = 0.01
    in_dim: int = field(default=0, init=False)
    n_classes: int = field(default=0, init=False)
    class_names: List[str] = field(default_factory=list, init=False)


@dataclass
class Spec:
    """One experimental configuration."""
    name: str
    federated: bool
    dp: bool
    sigma: float = 0.0
    target_eps: float = 0.0
    client_sigmas: Tuple[float, ...] = ()

    def sigma_of(self, cid: int) -> float:
        return self.client_sigmas[cid] if self.client_sigmas else self.sigma


# ========================== LABELS AND FEATURES ================================
LABEL_PRIORITY = {
    'benign_iso_tcp': 1, 'benign_ytdlp_https': 1, 'benign_interrupted_by_cve': 1,
    'background': 0,
    'attack_portscan': 2, 'attack_syn_flood': 2,
    'attack_udp_flood': 2, 'attack_icmp_flood': 2,
    'attack_cve_65805': 3,
}
ATTACK_PROTO = {'attack_syn_flood': 6, 'attack_udp_flood': 17, 'attack_icmp_flood': 1}
CVE_LABEL = 'attack_cve_65805'

FLOW_FEATURES = [
    'bidirectional_duration_ms', 'bidirectional_packets', 'bidirectional_bytes',
    'src2dst_duration_ms', 'src2dst_packets', 'src2dst_bytes',
    'dst2src_duration_ms', 'dst2src_packets', 'dst2src_bytes',
    'bidirectional_min_ps', 'bidirectional_mean_ps',
    'bidirectional_stddev_ps', 'bidirectional_max_ps',
    'src2dst_min_ps', 'src2dst_mean_ps', 'src2dst_max_ps',
    'dst2src_min_ps', 'dst2src_mean_ps', 'dst2src_max_ps',
    'bidirectional_min_piat_ms', 'bidirectional_mean_piat_ms',
    'bidirectional_stddev_piat_ms', 'bidirectional_max_piat_ms',
    'bidirectional_syn_packets', 'bidirectional_ack_packets',
    'bidirectional_psh_packets', 'bidirectional_rst_packets',
    'bidirectional_fin_packets', 'bidirectional_urg_packets',
    'protocol',
]
PROTO_ONEHOT = {6: 'proto_tcp', 17: 'proto_udp', 1: 'proto_icmp', 132: 'proto_sctp'}


def infer_attacker_ips(ldf: pd.DataFrame) -> Tuple[str, ...]:
    """Source IPs of the attack rows in labels.csv (if it has a src_ip column)."""
    if 'src_ip' not in ldf.columns:
        return ()
    atk = ldf[ldf['label'].astype(str).str.startswith('attack_')]
    return tuple(sorted(set(atk['src_ip'].dropna().astype(str).str.strip())))


def assign_labels(df: pd.DataFrame, labels_csv: Path,
                  attacker_ips: Tuple[str, ...]) -> Tuple[pd.Series, int]:
    """
    Labels flows using the time windows in labels.csv.
    """
    ldf = pd.read_csv(labels_csv)
    ldf.columns = [c.strip('\ufeff').strip() for c in ldf.columns]
    attacker_ips = tuple(attacker_ips) or infer_attacker_ips(ldf)
    t_rel = (df[T_COL] - df[T_COL].min()) / 1000.0
    proto = pd.to_numeric(df.get('protocol'), errors='coerce').fillna(-1).astype(int)
    from_attacker = None
    if attacker_ips:
        from_attacker = df['src_ip'].isin(attacker_ips) | df['dst_ip'].isin(attacker_ips)

    result = pd.Series('background', index=df.index)
    priority = pd.Series(0, index=df.index)
    in_benign_window = pd.Series(False, index=df.index)

    for _, row in ldf.iterrows():
        lbl = str(row['label']).strip()
        ts, te = float(row['t_start']), float(row['t_end'])
        prio = LABEL_PRIORITY.get(lbl, 1)
        window = t_rel.between(ts, te)
        if lbl.startswith('benign'):
            in_benign_window |= window
        mask = window & (prio > priority)
        if lbl.startswith('attack_'):
            if lbl in ATTACK_PROTO:
                mask &= (proto == ATTACK_PROTO[lbl])
            if from_attacker is not None:
                mask &= from_attacker
        result[mask] = lbl
        priority[mask] = prio

    n_reassigned = n_other = n_reverse = 0
    if from_attacker is not None:
        stray = from_attacker & ~result.str.startswith('attack_')
        present = set(ldf['label'].astype(str).str.strip())
        for lbl, pnum in ATTACK_PROTO.items():
            if lbl in present:
                m = stray & (proto == pnum)
                result[m] = lbl
                n_reassigned += int(m.sum())
        left = from_attacker & ~result.str.startswith('attack_')
        result[left] = 'attacker_other'
        n_other = int(left.sum())
        n_reverse = int((result.str.startswith('attack_') & ~df['src_ip'].isin(attacker_ips)).sum())

    is_attack = result.str.startswith('attack_') & (result != CVE_LABEL)
    result.attrs.update(attacker_ips=attacker_ips, n_reassigned=n_reassigned,
                        n_attacker_other=n_other, n_reverse=n_reverse)
    return result, int((in_benign_window & is_attack).sum())


def canonical_label(lbl: str, cfg: Config) -> Optional[str]:
    """Merges the benign labels; returns None for what must be discarded."""
    if lbl.startswith('benign'):
        return 'benign'
    if lbl == 'attacker_other':
        return None
    if lbl == 'background':
        return 'benign' if cfg.background_as_benign else None
    if lbl == CVE_LABEL and cfg.drop_cve:
        return None
    return lbl


def build_flow_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    feats, names = [], []
    for col in FLOW_FEATURES:
        if col == 'protocol':
            proto = pd.to_numeric(df.get('protocol'), errors='coerce').fillna(-1).astype(int)
            for num, name in PROTO_ONEHOT.items():
                feats.append((proto == num).astype(np.float32).values)
                names.append(name)
        else:
            feats.append(pd.to_numeric(df.get(col), errors='coerce')
                         .fillna(0).astype(np.float32).values)
            names.append(col)
    return np.column_stack(feats).astype(np.float32), names


def split_indices(df: pd.DataFrame, cfg: Config, rng: np.random.Generator
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per class: [---- training period ----][-- test (last 25%) --]
    A RANDOM holdout (validation set and MIA non-members) is drawn from the training
    period, and the rest becomes training data, capped at max_per_class.
    """
    tr, ho, te = [], [], []
    for lbl, g in df.groupby('label'):
        idx = g.sort_values(T_COL).index.to_numpy()
        if cfg.split == 'random':
            idx = rng.permutation(idx)
        n = len(idx)
        if n < 5:
            print(f"    [warning] class '{lbl}' has {n} flows: all of them go to training")
            tr.append(idx)
            continue
        n_te = max(1, int(round(n * cfg.test_frac)))
        pool = rng.permutation(idx[:n - n_te])
        te.append(idx[n - n_te:])
        n_ho = max(1, int(round(len(pool) * cfg.holdout_frac)))
        ho.append(pool[:n_ho])
        keep = pool[n_ho:]
        if cfg.max_per_class and len(keep) > cfg.max_per_class:
            keep = keep[:cfg.max_per_class]
        tr.append(keep)
    cat = lambda xs: np.concatenate(xs) if xs else np.array([], dtype=int)
    return cat(tr), cat(ho), cat(te)


def label_scenario(flow_csv: Path, labels_csv: Path, cfg: Config, verbose: bool = False
                   ) -> Tuple[pd.DataFrame, int, int]:
    """Reads the flow CSV and applies the labels. Returns (df, n_background, n_overlap)."""
    df = pd.read_csv(flow_csv, low_memory=False)
    df = df.dropna(subset=[T_COL]).reset_index(drop=True)
    n_dup = 0
    if cfg.drop_ip_prefixes:
        pref = tuple(cfg.drop_ip_prefixes)
        dup = (df['src_ip'].astype(str).str.startswith(pref)
               | df['dst_ip'].astype(str).str.startswith(pref))
        n_dup = int(dup.sum())
    raw, n_overlap = assign_labels(df, labels_csv, cfg.attacker_ips)
    a = raw.attrs
    if not a.get('attacker_ips'):
        n_overlap = -abs(n_overlap)
    if cfg.drop_ip_prefixes:
        raw = raw[~dup]; df = df[~dup]
    if verbose:
        print(f"  {flow_csv.name}: attacker={list(a.get('attacker_ips', ())) or 'unknown'} | "
              f"{a.get('n_reassigned', 0):,} attacker flows outside the windows reassigned by protocol | "
              f"{a.get('n_attacker_other', 0):,} discarded (attacker_other) | "
              f"{a.get('n_reverse', 0):,} attack flows are replies from the victim"
              + (f" | {n_dup:,} flows removed by --drop-ip-prefix" if cfg.drop_ip_prefixes else ""))
    n_bg = int((raw == 'background').sum())
    df = df.assign(label=raw.map(lambda s: canonical_label(s, cfg)))
    return df[df['label'].notna()].reset_index(drop=True), n_bg, n_overlap


def split_scenario(df: pd.DataFrame, name: str, n_bg: int, n_overlap: int, cfg: Config,
                   rng: np.random.Generator, verbose: bool) -> Dict:
    X, names = build_flow_features(df)
    if cfg.log1p:
        X = np.log1p(np.clip(X, 0, None))
    y = df['label'].values
    tr, ho, te = split_indices(df, cfg, rng)

    if verbose:
        dist = df['label'].value_counts().to_dict()
        print(f"  {name}: train={len(tr):,} holdout={len(ho):,} test={len(te):,}")
        print(f"    classes (total): {dist}")
        print(f"    background: {n_bg:,} flows "
              f"({'used as benign' if cfg.background_as_benign else 'discarded'})")
        if n_overlap < 0:
            print(f"    [attention] {-n_overlap:,} flows in a BENIGN window were labeled as an "
                  f"attack by time+protocol only (set --attacker-ips)")
    return dict(tr=(X[tr], y[tr]), ho=(X[ho], y[ho]), te=(X[te], y[te]), names=names)


def federated_standardizer(client_X: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Global mean/std computed from each client's (n, sum, sum of squares)."""
    n = sum(len(X) for X in client_X)
    s = sum(X.astype(np.float64).sum(0) for X in client_X)
    ss = sum((X.astype(np.float64) ** 2).sum(0) for X in client_X)
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 0.0))
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def prepare_data(flow_csvs: List[Path], label_csvs: List[Path], cfg: Config,
                 verbose: bool = True) -> Dict:
    rng = np.random.default_rng(cfg.seed)
    if verbose:
        print(f"Loading scenarios (split={cfg.split})...")
    labeled = [label_scenario(f, l, cfg, verbose) for f, l in zip(flow_csvs, label_csvs)]
    totals = pd.concat([d['label'] for d, _, _ in labeled]).value_counts()
    rare = totals[totals < cfg.min_class_flows]
    if len(rare) and verbose:
        print(f"  [warning] classes removed for having < {cfg.min_class_flows} flows in total: "
              f"{rare.to_dict()}")
    scen = []
    for (d, n_bg, n_ov), f in zip(labeled, flow_csvs):
        d = d[~d['label'].isin(rare.index)].reset_index(drop=True)
        scen.append(split_scenario(d, f.name, n_bg, n_ov, cfg, rng, verbose))

    all_y = np.concatenate([np.concatenate([s['tr'][1], s['ho'][1], s['te'][1]]) for s in scen])
    le = LabelEncoder().fit(all_y)
    cfg.n_classes, cfg.class_names = len(le.classes_), list(le.classes_)
    cfg.in_dim = scen[0]['tr'][0].shape[1]
    mean, std = federated_standardizer([s['tr'][0] for s in scen])

    def to_ds(X, y):
        return TensorDataset(torch.tensor((X - mean) / std, dtype=torch.float32),
                             torch.tensor(le.transform(y), dtype=torch.long))

    def pooled(key):
        return to_ds(np.concatenate([s[key][0] for s in scen]),
                     np.concatenate([s[key][1] for s in scen]))

    if verbose:
        print(f"\nClasses ({cfg.n_classes}): {cfg.class_names}")
        print(f"Features ({cfg.in_dim})" + (" with log1p" if cfg.log1p else ""))
    return dict(client_ds=[to_ds(*s['tr']) for s in scen],
                train_all=pooled('tr'), hold_ds=pooled('ho'), test_ds=pooled('te'))


# ============================== MODELS ======================================
class FlowMLP(nn.Module):
    """Baseline MLP: three linear layers with ReLU and Dropout."""
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, n_classes),
        )
    def forward(self, x): return self.net(x)


class Flow1DCNN(nn.Module):
    """
    Lightweight 1D-CNN of Fadaei & Barekatain (J. Supercomput., 2026), Fig. 3:
      Conv1D(F, K) → ReLU → MaxPooling1D → Dropout → Flatten → Dense(H) → ReLU → output.
    Hyperparameters from Table 6 (federated learning, binary scenario):
      F = 16 filters, K = 3, H = 32 dense units, dropout 0.2.
    """
    def __init__(self, in_dim: int, n_classes: int, filters: int = 16,
                 kernel_size: int = 3, dense_units: int = 32, dropout: float = 0.2,
                 pool: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(1, filters, kernel_size)            
        self.pool = nn.MaxPool1d(pool)
        self.drop = nn.Dropout(dropout)
        flat = filters * ((in_dim - kernel_size + 1) // pool)
        self.fc1 = nn.Linear(flat, dense_units)
        self.fc2 = nn.Linear(dense_units, n_classes)

    def forward(self, x):
        x = F.relu(self.conv(x.unsqueeze(1)))    
        x = self.drop(self.pool(x)).flatten(1)  
        return self.fc2(F.relu(self.fc1(x)))


class Flow1DCNNGMP(nn.Module):
    """
    Earlier 1D-CNN with global max pooling (--model cnn_gmp; not used in the paper).
    """
    def __init__(self, in_dim: int, n_classes: int, filters: int = 64,
                 kernel_size: int = 3, dense_units: int = 64):
        super().__init__()
        self.conv = nn.Conv1d(1, filters, kernel_size, padding=kernel_size // 2)
        self.fc1 = nn.Linear(filters, dense_units)
        self.fc2 = nn.Linear(dense_units, n_classes)

    def forward(self, x):
        x = F.relu(self.conv(x.unsqueeze(1)))   
        x = x.amax(dim=2)                        
        return self.fc2(F.relu(self.fc1(x)))


def make_model(cfg: Config) -> nn.Module:
    if cfg.model_type == 'cnn':
        return Flow1DCNN(cfg.in_dim, cfg.n_classes)
    if cfg.model_type == 'cnn_gmp':
        return Flow1DCNNGMP(cfg.in_dim, cfg.n_classes)
    return FlowMLP(cfg.in_dim, cfg.n_classes)


def get_weights(model: nn.Module) -> List[np.ndarray]:
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def set_weights(model: nn.Module, weights: List[np.ndarray]) -> None:
    sd = model.state_dict()
    for k, w in zip(sd.keys(), weights):
        sd[k] = torch.tensor(w)
    model.load_state_dict(sd)


# ========================== LOCAL TRAINING ==================================
def class_weights(dataset: TensorDataset, cfg: Config) -> torch.Tensor:
    """
    Balanced weights over the classes PRESENT in the dataset only, capped, and
    normalized so that max = 1.
    """
    if not cfg.use_class_weight:
        return torch.ones(cfg.n_classes, device=DEVICE)
    counts = torch.bincount(dataset.tensors[1], minlength=cfg.n_classes).float()
    present = counts > 0
    w = torch.zeros(cfg.n_classes)
    w[present] = counts.sum() / (present.sum() * counts[present])
    w = w.clamp(max=cfg.max_class_weight)
    return (w / w.max()).to(DEVICE)


def train_local(model: nn.Module, dataset: TensorDataset, cfg: Config,
                dp: bool, sigma: float, epochs: int, mu: float = 0.0) -> None:
    import warnings as _w                      
    _w.filterwarnings("ignore")
    model.to(DEVICE)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    w = class_weights(dataset, cfg)
    ce = nn.CrossEntropyLoss(reduction='none')
    # global model reference for the proximal term (FedProx)
    ref = [p.detach().clone() for p in model.parameters()] if mu > 0 else None

    net = model
    if dp:
        from opacus import PrivacyEngine
        net, opt, loader = PrivacyEngine().make_private(
            module=model, optimizer=opt, data_loader=loader,
            noise_multiplier=sigma, max_grad_norm=cfg.max_grad_norm)

    net.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = (ce(net(xb), yb) * w[yb]).mean()
            loss.backward()
            opt.step()
            if ref is not None:
                with torch.no_grad():
                    for p, g in zip(model.parameters(), ref):
                        p.add_(p - g, alpha=-cfg.lr * mu)


# ============================ EPSILON =======================================
def epsilon_for(n_samples: int, cfg: Config, sigma: float, total_epochs: int) -> float:
    """
    ε composed over all DP-SGD steps on a dataset with n_samples.
    Opacus uses Poisson sampling with q = 1/len(loader). Uses the PRV accountant
    (the PrivacyEngine default); if it fails, falls back to RDP, which is a valid
    but looser upper bound.
    """
    if sigma <= 0:
        return float('inf')
    from opacus.accountants import PRVAccountant, RDPAccountant
    steps_per_epoch = math.ceil(n_samples / cfg.batch_size)
    history = [(sigma, 1.0 / steps_per_epoch, steps_per_epoch * total_epochs)]
    for Acc in (PRVAccountant, RDPAccountant):
        try:
            acc = Acc(); acc.history = list(history)
            return float(acc.get_epsilon(cfg.delta))
        except Exception:
            continue
    return float('nan')


_SIGMA_CACHE: Dict[Tuple, float] = {}


def sigma_for_eps(n_samples: int, cfg: Config, target_eps: float, total_epochs: int,
                  lo: float = 0.1, hi: float = 200.0, iters: int = 40) -> float:
    """
    Smallest σ with ε(σ) ≤ target_eps, by bisection on the SAME epsilon_for used to
    report ε (same accountant → the reported ε stays ≤ the target).
    """
    key = (n_samples, cfg.batch_size, cfg.delta, total_epochs, round(target_eps, 6))
    if key in _SIGMA_CACHE:
        return _SIGMA_CACHE[key]
    if epsilon_for(n_samples, cfg, hi, total_epochs) > target_eps:
        raise ValueError(f"ε ≤ {target_eps} unreachable with σ ≤ {hi} (N={n_samples})")
    if epsilon_for(n_samples, cfg, lo, total_epochs) <= target_eps:
        _SIGMA_CACHE[key] = lo
        return lo
    for _ in range(iters):                     
        mid = math.sqrt(lo * hi)               
        if epsilon_for(n_samples, cfg, mid, total_epochs) <= target_eps:
            hi = mid
        else:
            lo = mid
        if hi / lo < 1.001:
            break
    _SIGMA_CACHE[key] = hi
    return hi


def resolve_spec(spec: Spec, cfg: Config, data: Dict, verbose: bool) -> Spec:
    """Fills in client_sigmas when the configuration uses a per-client target ε."""
    if not (spec.dp and spec.federated and spec.target_eps > 0):
        return spec
    total_epochs = cfg.num_rounds * cfg.local_epochs
    sig = tuple(sigma_for_eps(len(d), cfg, spec.target_eps, total_epochs)
                for d in data['client_ds'])
    if verbose:
        print("    σ per client: " + ", ".join(
            f"N={len(d):,} → σ={s:.3f}" for d, s in zip(data['client_ds'], sig)))
    return replace(spec, client_sigmas=sig, sigma=float(np.median(sig)))


def spec_epsilon(spec: Spec, cfg: Config, data: Dict, verbose: bool
                 ) -> Tuple[float, float, List[float]]:
    """(ε of the worst client, median ε across clients, ε of each client)."""
    if not spec.dp:
        return float('nan'), float('nan'), []
    total_epochs = cfg.num_rounds * cfg.local_epochs
    sizes = ([len(d) for d in data['client_ds']] if spec.federated
             else [len(data['train_all'])])
    sigmas = [spec.sigma_of(i) for i in range(len(sizes))]
    eps = [epsilon_for(n, cfg, s, total_epochs) for n, s in zip(sizes, sigmas)]
    if verbose and max(sigmas) > 0:
        for n, s, e in zip(sizes, sigmas, eps):
            flag = "  [δ ≥ 1/N!]" if cfg.delta >= 1.0 / n else ""
            print(f"    ε(N={n:,}, σ={s:.3f}) = {e:.3f}{flag}")
    return max(eps), float(np.median(eps)), eps


# ========================== EVALUATION ======================================
@torch.no_grad()
def predict(model: nn.Module, dataset: TensorDataset) -> Tuple[np.ndarray, np.ndarray]:
    model.eval().to(DEVICE)
    ys, preds = [], []
    for xb, yb in DataLoader(dataset, batch_size=4096):
        preds.append(model(xb.to(DEVICE)).argmax(1).cpu()); ys.append(yb)
    return torch.cat(ys).numpy(), torch.cat(preds).numpy()


def evaluate(model: nn.Module, dataset: TensorDataset, cfg: Config) -> Dict:
    y, pred = predict(model, dataset)
    cn = cfg.class_names
    labels = list(range(len(cn)))
    out = dict(
        accuracy=float((pred == y).mean()),
        macro_f1=float(f1_score(y, pred, labels=sorted(set(y.tolist())), average='macro',
                                zero_division=0)),
        report=classification_report(y, pred, labels=labels, target_names=cn,
                                     digits=3, zero_division=0),
        cm=confusion_matrix(y, pred, labels=labels),
        benign_f1=float('nan'), benign_recall=float('nan'),
        far=float('nan'), det_rate=float('nan'),
    )
    if 'benign' in cn:
        b = cn.index('benign')
        is_b, pred_b = (y == b), (pred == b)
        out['benign_f1'] = float(f1_score(is_b, pred_b, zero_division=0))
        out['benign_recall'] = float((is_b & pred_b).sum() / max(is_b.sum(), 1))
        out['far'] = float((is_b & ~pred_b).sum() / max(is_b.sum(), 1))        # false alarm
        out['det_rate'] = float((~is_b & ~pred_b).sum() / max((~is_b).sum(), 1))  # attack detected
    return out


@torch.no_grad()
def predict_logits(model: nn.Module, dataset: TensorDataset) -> Tuple[np.ndarray, np.ndarray, float]:
    """(labels, logits, inference seconds over the whole dataset)."""
    model.eval().to(DEVICE)
    ys, outs = [], []
    t0 = time.perf_counter()
    for xb, yb in DataLoader(dataset, batch_size=4096):
        outs.append(model(xb.to(DEVICE)).cpu()); ys.append(yb)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return torch.cat(ys).numpy(), torch.cat(outs).numpy().astype(np.float64), dt


def benign_score(logits: np.ndarray, b: int) -> np.ndarray:
    """s = log P(benign) − log P(attack) = logit_b − logsumexp(attack logits)."""
    other = np.delete(logits, b, axis=1)
    m = other.max(1, keepdims=True)
    lse = (m + np.log(np.exp(other - m).sum(1, keepdims=True)))[:, 0]
    return logits[:, b] - lse


def calibrate_threshold(s: np.ndarray, is_benign: np.ndarray, mode: str = 'youden',
                        target_far: float = 0.01) -> Tuple[float, Dict]:
    """
    Threshold t on the benign score (benign if s ≥ t), chosen on the holdout.
      youden: max. TNR(t) + DR(t) − 1; ties → middle of the optimal interval
              (largest margin on both sides).
      far   : largest t with FAR(t) ≤ target_far (more attacks detected).
    Returns (t, metrics on the holdout).
    """
    sb, sa = np.sort(s[is_benign]), np.sort(s[~is_benign])
    if len(sb) == 0 or len(sa) == 0:
        return 0.0, dict(far_val=float('nan'), dr_val=float('nan'))
    cand = np.unique(np.concatenate([sb, sa]))
    cand = np.concatenate([[cand[0] - 1.0], (cand[:-1] + cand[1:]) / 2, [cand[-1] + 1.0]])
    far = np.searchsorted(sb, cand, side='left') / len(sb)        
    dr = np.searchsorted(sa, cand, side='left') / len(sa)         
    if mode == 'far':
        ok = np.where(far <= target_far)[0]
        i = ok[np.argmax(cand[ok])] if len(ok) else 0
        t = float(cand[i])
    else:
        j = dr - far
        if j.max() <= 1e-9:
            t = 0.0
        else:
            best = np.where(j >= j.max() - 1e-12)[0]
            runs = np.split(best, np.where(np.diff(best) > 1)[0] + 1)
            run = max(runs, key=lambda r: cand[r[-1]] - cand[r[0]])
            t = float((cand[run[0]] + cand[run[-1]]) / 2)
    return t, dict(far_val=float(np.mean(sb < t)), dr_val=float(np.mean(sa < t)))


def metrics_from_pred(y: np.ndarray, pred: np.ndarray, cfg: Config) -> Dict:
    out = dict(macro_f1=float(f1_score(y, pred, labels=sorted(set(y.tolist())), average='macro',
                                       zero_division=0)),
               benign_f1=float('nan'), far=float('nan'), det_rate=float('nan'))
    if 'benign' in cfg.class_names:
        b = cfg.class_names.index('benign')
        is_b, pred_b = (y == b), (pred == b)
        out['benign_f1'] = float(f1_score(is_b, pred_b, zero_division=0))
        out['far'] = float((is_b & ~pred_b).sum() / max(is_b.sum(), 1))
        out['det_rate'] = float((~is_b & ~pred_b).sum() / max((~is_b).sum(), 1))
    return out


def evaluate_calibrated(model: nn.Module, data: Dict, cfg: Config) -> Dict:
    """Decision with the threshold calibrated on the holdout + inference latency on the test set."""
    yt, lt, dt = predict_logits(model, data['test_ds'])
    out = dict(infer_us_per_flow=1e6 * dt / max(len(yt), 1),
               thr_cal=float('nan'), far_val_cal=float('nan'), dr_val_cal=float('nan'),
               macro_f1_cal=float('nan'), benign_f1_cal=float('nan'),
               far_cal=float('nan'), det_rate_cal=float('nan'))
    if 'benign' not in cfg.class_names or cfg.n_classes < 2:
        return out
    b = cfg.class_names.index('benign')
    yh, lh, _ = predict_logits(model, data['hold_ds'])
    t, val = calibrate_threshold(benign_score(lh, b), yh == b, cfg.calib, cfg.target_far)
    st = benign_score(lt, b)
    atk = np.delete(np.arange(cfg.n_classes), b)
    pred = atk[np.argmax(lt[:, atk], axis=1)]            
    pred[st >= t] = b
    m = metrics_from_pred(yt, pred, cfg)
    out.update(thr_cal=t, far_val_cal=val['far_val'], dr_val_cal=val['dr_val'],
               macro_f1_cal=m['macro_f1'], benign_f1_cal=m['benign_f1'],
               far_cal=m['far'], det_rate_cal=m['det_rate'])
    return out


@torch.no_grad()
def _losses(model: nn.Module, ds: TensorDataset, idx: np.ndarray) -> np.ndarray:
    model.eval().to(DEVICE)
    ce = nn.CrossEntropyLoss(reduction='none')
    X, y = ds.tensors
    out = []
    for i in range(0, len(idx), 4096):
        j = torch.as_tensor(idx[i:i + 4096])
        out.append(ce(model(X[j].to(DEVICE)), y[j].to(DEVICE)).cpu().numpy())
    return np.concatenate(out)


def mia_loss_attack(model: nn.Module, members: TensorDataset, nonmembers: TensorDataset,
                    rng: np.random.Generator, cls: Optional[int] = None,
                    cap: int = 5000) -> Tuple[float, float, int]:
    """
    Loss-threshold membership inference attack (baseline; LiRA in 08_mia_audit.py is
    stronger). Members = training set; non-members = holdout. Class-balanced sample.
    Returns (AUC, TPR@1%FPR, n per side).
    """
    ym, yn = members.tensors[1].numpy(), nonmembers.tensors[1].numpy()
    classes = [cls] if cls is not None else sorted(set(ym.tolist()) & set(yn.tolist()))

    per = max(1, cap // max(len(classes), 1))
    pm, pn = [], []
    for c in classes:
        im, inn = np.where(ym == c)[0], np.where(yn == c)[0]
        k = min(len(im), len(inn), per)
        if k:
            pm.append(rng.choice(im, k, replace=False)); pn.append(rng.choice(inn, k, replace=False))
    n = sum(len(x) for x in pm)
    if n < 20:
        return float('nan'), float('nan'), n
    lm = _losses(model, members, np.concatenate(pm))
    ln = _losses(model, nonmembers, np.concatenate(pn))
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    scores = np.concatenate([-lm, -ln])
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(roc_auc_score(labels, scores)), float(tpr[fpr <= 0.01].max()), n


# ======================= CENTRALIZED AND FL =================================
def run_centralized(cfg: Config, data: Dict, spec: Spec) -> Tuple[nn.Module, Dict]:
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    model = make_model(cfg).to(DEVICE)
    t0 = time.perf_counter()
    train_local(model, data['train_all'], cfg, spec.dp, spec.sigma,
                epochs=cfg.num_rounds * cfg.local_epochs)
    dt = time.perf_counter() - t0
    clean = make_model(cfg).to(DEVICE)          # without the Opacus hooks
    clean.load_state_dict(model.state_dict())
    return clean, dict(client_train_s=dt, client_parallel_s=dt, client_train_max_s=dt)


from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation


class FlowClient(NumPyClient):
    def __init__(self, ds, cfg: Config, spec: Spec, cid: int):
        self.ds, self.cfg, self.spec, self.cid = ds, cfg, spec, cid
        self.model = make_model(cfg).to(DEVICE)

    def get_parameters(self, config): return get_weights(self.model)

    def fit(self, parameters, config):
        rnd = int(config.get("round", 0))
        s = self.cfg.seed * 10_007 + self.cid * 101 + rnd
        np.random.seed(s); torch.manual_seed(s)
        set_weights(self.model, parameters)
        t0 = time.perf_counter()
        train_local(self.model, self.ds, self.cfg, self.spec.dp, self.spec.sigma_of(self.cid),
                    epochs=self.cfg.local_epochs, mu=self.cfg.mu)
        dt = time.perf_counter() - t0
        return get_weights(self.model), len(self.ds), {"train_s": float(dt), "cid": self.cid}


class SavingFedAvg(FedAvg):
    """FedAvg that stores the parameters of the last round and the local training time."""
    latest_parameters = None

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.round_times: List[Dict[int, float]] = []      # per round: {cid: seconds}

    def aggregate_fit(self, rnd, results, failures):
        self.round_times.append({int(r.metrics.get("cid", i)): float(r.metrics.get("train_s", 0.0))
                                 for i, (_, r) in enumerate(results)})
        agg = super().aggregate_fit(rnd, results, failures)
        if agg[0] is not None:
            self.latest_parameters = agg[0]
        return agg

    def timing(self) -> Dict[str, float]:
        if not self.round_times:
            return dict(client_train_s=float('nan'), client_parallel_s=float('nan'),
                        client_train_max_s=float('nan'))
        per_client: Dict[int, float] = {}
        for rt in self.round_times:
            for c, s in rt.items():
                per_client[c] = per_client.get(c, 0.0) + s
        return dict(
            client_train_s=float(sum(per_client.values())),               # sum over all clients
            client_parallel_s=float(sum(max(rt.values()) for rt in self.round_times)),
            client_train_max_s=float(max(per_client.values())),           # most expensive client
        )


def run_federated(cfg: Config, data: Dict, spec: Spec) -> Tuple[nn.Module, Dict]:
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    n = len(data['client_ds'])
    init = ndarrays_to_parameters(get_weights(make_model(cfg)))   # deterministic start
    strat = SavingFedAvg(fraction_fit=1.0, fraction_evaluate=0.0,
                         min_fit_clients=n, min_available_clients=n,
                         initial_parameters=init,
                         on_fit_config_fn=lambda rnd: {"round": rnd})
    _data, _cfg, _spec = data, copy.copy(cfg), spec

    def client_fn(context: Context):
        cid = int(context.node_config["partition-id"])
        return FlowClient(_data['client_ds'][cid], _cfg, _spec, cid).to_client()

    def server_fn(context: Context):
        return ServerAppComponents(strategy=strat,
                                   config=ServerConfig(num_rounds=_cfg.num_rounds))

    flwr_log = logging.getLogger("flwr")
    flwr_log.setLevel(logging.ERROR)            
    class _NoDeprecated(logging.Filter):
        def filter(self, r): return "deprecated" not in r.getMessage().lower()
    flt = _NoDeprecated(); flwr_log.addFilter(flt)
    run_simulation(server_app=ServerApp(server_fn=server_fn),
                   client_app=ClientApp(client_fn=client_fn),
                   num_supernodes=n,
                   backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}})
    flwr_log.removeFilter(flt)

    final = make_model(cfg).to(DEVICE)
    set_weights(final, parameters_to_ndarrays(strat.latest_parameters))
    return final, strat.timing()


# ============================== SUITE =======================================
def build_specs(sigmas: List[float], target_eps: Optional[List[float]] = None,
                central_dp: bool = True) -> List[Spec]:
    specs = [
        Spec("Central",                        federated=False, dp=False),
        Spec("Central+clip (s=0)",             federated=False, dp=True, sigma=0.0),
    ]
    if central_dp:
        specs.append(Spec(f"Central+DP s={sigmas[0]:g}", federated=False, dp=True,
                          sigma=sigmas[0]))
    specs += [
        Spec("FL",                             federated=True,  dp=False),
        Spec("FL+clip (s=0)",                  federated=True,  dp=True, sigma=0.0),
    ]
    specs += [Spec(f"FL+DP s={s:g}", federated=True, dp=True, sigma=s) for s in sigmas]
    specs += [Spec(f"FL+DP eps<={e:g}", federated=True, dp=True, target_eps=e)
              for e in (target_eps or [])]
    return specs


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def warmup(cfg: Config, data: Dict) -> None:
    """
    Trains a throwaway model on a few examples (without and with Opacus) so that the
    PyTorch/Opacus initialization is not counted in the time of the 1st measured
    configuration.
    """
    X, y = data['train_all'].tensors
    small = TensorDataset(X[:256], y[:256])
    for dp in (False, True):
        train_local(make_model(cfg), small, cfg, dp=dp, sigma=1.0, epochs=1)
    predict(make_model(cfg), small)


def run_suite(cfg: Config, data: Dict, specs: List[Spec], verbose: bool) -> List[Dict]:
    rows = []
    warmup(cfg, data)
    b = cfg.class_names.index('benign') if 'benign' in cfg.class_names else None
    n_clients = len(data['client_ds'])
    for spec in specs:
        print(f"\n  >> {spec.name}")
        spec = resolve_spec(spec, cfg, data, verbose=True)
        eps, eps_med, eps_list = spec_epsilon(spec, cfg, data, verbose)
        t0 = time.perf_counter()
        model, timing = (run_federated if spec.federated else run_centralized)(cfg, data, spec)
        wall = time.perf_counter() - t0
        res = evaluate(model, data['test_ds'], cfg)
        cal = evaluate_calibrated(model, data, cfg)
        npar = n_params(model)
        kb_round = (2 * n_clients * 4 * npar / 1000.0) if spec.federated else 0.0
        rng = np.random.default_rng(cfg.seed + 7)
        auc_all, tpr_all, _ = mia_loss_attack(model, data['train_all'], data['hold_ds'], rng)
        auc_b, tpr_b, n_b = (mia_loss_attack(model, data['train_all'], data['hold_ds'], rng, cls=b)
                             if b is not None else (float('nan'), float('nan'), 0))
        print(f"     macro-F1={res['macro_f1']:.3f}  benign F1={res['benign_f1']:.3f}  "
              f"FAR={res['far']:.3f}  DR={res['det_rate']:.3f}  "
              f"MIA benign={auc_b:.3f} (n={n_b})  MIA all={auc_all:.3f}")
        print(f"     calibrated (t={cal['thr_cal']:.2f}; holdout FAR={cal['far_val_cal']:.3f} "
              f"DR={cal['dr_val_cal']:.3f}): macro-F1={cal['macro_f1_cal']:.3f}  "
              f"FAR={cal['far_cal']:.3f}  DR={cal['det_rate_cal']:.3f}")
        print(f"     time: wall={wall:.1f}s  local training (sum)={timing['client_train_s']:.1f}s  "
              f"parallel={timing['client_parallel_s']:.1f}s  | {npar:,} parameters"
              + (f"  | {kb_round:.1f} kB/round" if spec.federated else "")
              + f"  | inference {cal['infer_us_per_flow']:.2f} µs/flow")
        if verbose:
            for line in res['report'].splitlines():
                print(f"       {line}")
            print("       Confusion matrix (row = true, column = predicted):")
            print("       " + pd.DataFrame(res['cm'], index=cfg.class_names,
                                           columns=cfg.class_names)
                  .to_string().replace("\n", "\n       "))
        rows.append(dict(
            model=cfg.model_type, config=spec.name, seed=cfg.seed, sigma=spec.sigma,
            eps=eps, eps_median=eps_med, accuracy=res['accuracy'], macro_f1=res['macro_f1'],
            benign_f1=res['benign_f1'], benign_recall=res['benign_recall'],
            far=res['far'], det_rate=res['det_rate'],
            mia_auc_benign=auc_b, mia_tpr1_benign=tpr_b, mia_n_benign=n_b,
            mia_auc_all=auc_all, mia_tpr1_all=tpr_all,
            # target ε, per-client σ and ε, calibrated threshold, and cost
            target_eps=spec.target_eps,
            sigma_clients=";".join(f"{spec.sigma_of(i):.4f}" for i in range(n_clients))
                          if spec.federated and spec.dp else "",
            eps_clients=";".join(f"{e:.4f}" for e in eps_list),
            n_clients_train=";".join(str(len(d)) for d in data['client_ds']),
            **cal, calib=cfg.calib,
            train_wall_s=wall, **timing,
            n_params=npar, kb_per_round=kb_round,
            kb_total=kb_round * cfg.num_rounds))
    return rows


def print_summary(df: pd.DataFrame, n_runs: int) -> None:
    cols = [('macro_f1', 'Macro-F1'), ('benign_f1', 'Benign F1'), ('far', 'FAR'),
            ('det_rate', 'DR'), ('eps', 'Eps worst'), ('eps_median', 'Eps med.'),
            ('mia_auc_benign', 'MIA ben'), ('mia_auc_all', 'MIA all')]
    cw = 14 if n_runs > 1 else 10
    W = 8 + 22 + cw * len(cols)
    print("\n" + "=" * W)
    print(f"SUMMARY — {n_runs} run(s)" + (" (mean ± std)" if n_runs > 1 else ""))
    print("=" * W)
    print(f"{'Model':<8}{'Configuration':<22}" + "".join(f"{t:>{cw}}" for _, t in cols))
    print("-" * W)
    last = None
    for (model, config), g in df.groupby(['model', 'config'], sort=False):
        if last is not None and model != last:
            print("-" * W)
        last = model
        cells = []
        for k, _ in cols:
            v = g[k].astype(float)
            if k in ('eps', 'eps_median'):
                cells.append("—" if v.isna().all() else
                             "inf" if np.isinf(v).all() else f"{v.max():.3f}")
            elif v.isna().all():
                cells.append("n/a")
            elif n_runs > 1:
                cells.append(f"{v.mean():.3f}±{v.std(ddof=0):.3f}")
            else:
                cells.append(f"{v.mean():.3f}")
        print(f"{model:<8}{config:<22}" + "".join(f"{c:>{cw}}" for c in cells))
    print("=" * W)

    # --- calibrated threshold and cost --------------------------------------
    cols2 = [('macro_f1_cal', 'F1 cal'), ('far_cal', 'FAR cal'), ('det_rate_cal', 'DR cal'),
             ('train_wall_s', 'Wall(s)'), ('client_parallel_s', 'Paral(s)'),
             ('client_train_s', 'Sum(s)'), ('kb_per_round', 'kB/round'),
             ('infer_us_per_flow', 'µs/flow')]
    if all(k in df.columns for k, _ in cols2):
        W2 = 8 + 22 + cw * len(cols2)
        print(f"\nTHRESHOLD CALIBRATED ON THE HOLDOUT ({df['calib'].dropna().iloc[0] if 'calib' in df else ''}) "
              f"AND COST")
        print("=" * W2)
        print(f"{'Model':<8}{'Configuration':<22}" + "".join(f"{t:>{cw}}" for _, t in cols2))
        print("-" * W2)
        last = None
        for (model, config), g in df.groupby(['model', 'config'], sort=False):
            if last is not None and model != last:
                print("-" * W2)
            last = model
            cells = []
            for k, _ in cols2:
                v = pd.to_numeric(g[k], errors='coerce')
                if v.isna().all():
                    cells.append("n/a")
                elif k in ('kb_per_round',):
                    cells.append(f"{v.mean():.1f}" if v.mean() > 0 else "—")
                elif k.endswith('_s') or k == 'infer_us_per_flow':
                    cells.append(f"{v.mean():.2f}")
                elif n_runs > 1:
                    cells.append(f"{v.mean():.3f}±{v.std(ddof=0):.3f}")
                else:
                    cells.append(f"{v.mean():.3f}")
            print(f"{model:<8}{config:<22}" + "".join(f"{c:>{cw}}" for c in cells))
        print("=" * W2)
        if 'eps_clients' in df.columns:
            fl = df[df['sigma_clients'].fillna('').astype(str).str.len() > 0]
            if len(fl):
                print("ε per client (1st seed of each configuration):")
                for (model, config), g in fl.groupby(['model', 'config'], sort=False):
                    r = g.iloc[0]
                    print(f"  {model:<6}{config:<22} ε=[{r['eps_clients']}]"
                          + f"  σ=[{r['sigma_clients']}]"
                          + f"  N=[{r['n_clients_train']}]")

    print("How to read:")
    print("  FAR     : fraction of benign traffic classified as an attack (lower = better).")
    print("  DR      : fraction of attack flows detected (higher = better).")
    print("  Eps     : composed over all rounds, flow level. 'worst' = client with the largest ε")
    print("            (small client → high sampling rate → high ε); 'med.' = median over clients.")
    print("  MIA     : AUC of the loss attack; 0.5 = the attacker cannot tell members from non-members.")
    print("  +clip   : per-sample clipping WITHOUT noise (not private). It is the fair reference for DP:")
    print("            the difference +clip → +DP is the cost of the noise.")
    print("  cal     : decision with a threshold on the benign score chosen on the holdout (not on the test set).")
    print("  Wall    : total training time (in FL, includes the Flower/Ray simulator overhead).")
    print("  Paral   : sum over rounds of the slowest client (clients training in parallel).")
    print("  Sum     : sum of the local training time of all clients (total computational cost).")
    print("  eps<=E  : σ chosen per client so that all clients have ε ≤ E.")


# ============================== MAIN ========================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--flows', nargs='+',
                    default=['data/scenario_1/flows_s1_nfstream.csv',
                             'data/scenario_2/flows_s2_nfstream.csv',
                             'data/scenario_3/flows_s3_nfstream.csv'])
    ap.add_argument('--labels', nargs='+',
                    default=['data/scenario_1/cve5g_s1_labels.csv',
                             'data/scenario_2/cve5g_s2_labels.csv',
                             'data/scenario_3/cve5g_s3_labels.csv'])
    ap.add_argument('--model', choices=['mlp', 'cnn', 'cnn_gmp', 'both'], default='both')
    ap.add_argument('--runs', type=int, default=1, help='number of seeds (42, 43, ...)')
    ap.add_argument('--sigmas', type=float, nargs='+', default=[1.0],
                    help='FL+DP noise multipliers (the 1st one is also used for Central+DP)')
    ap.add_argument('--mu', type=float, default=0.0, help='FedProx (0 = FedAvg)')
    ap.add_argument('--rounds', type=int, default=10)
    ap.add_argument('--local-epochs', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--delta', type=float, default=1e-6)
    ap.add_argument('--split', choices=['time', 'random'], default='time')
    ap.add_argument('--max-per-class', type=int, default=10_000,
                    help='cap per class on the training set of each client (0 = no cap)')
    ap.add_argument('--min-class-flows', type=int, default=50,
                    help='removes classes with fewer flows than this (summing all scenarios)')
    ap.add_argument('--attacker-ips', nargs='*', default=[],
                    help='attacker IPs (default: src_ip of the attack rows in labels.csv)')
    ap.add_argument('--drop-ip-prefix', nargs='*', default=[],
                    help="removes flows with src/dst in this prefix; e.g., '10.0.0.' removes the "
                         "intra-tunnel (GTP) copy of the UE flows, which duplicates the post-NAT copy")
    ap.add_argument('--background-as-benign', action='store_true',
                    help="uses 'background' flows as benign instead of discarding them")
    ap.add_argument('--no-class-weight', action='store_true')
    ap.add_argument('--first-seed', type=int, default=42,
                    help='seed of the 1st run (e.g., 44 with --runs 3 → 44, 45, 46)')
    ap.add_argument('--target-eps', type=float, nargs='*', default=[],
                    help='adds FL+DP with per-client σ for ε ≤ E (e.g., 2 8)')
    ap.add_argument('--calib', choices=['youden', 'far'], default='youden',
                    help='criterion of the threshold calibrated on the holdout')
    ap.add_argument('--target-far', type=float, default=0.01, help='target FAR with --calib far')
    ap.add_argument('--no-central-dp', action='store_true', help='does not run Central+DP')
    ap.add_argument('--out', default='results_05_v3.csv')
    args = ap.parse_args()

    flow_csvs, label_csvs = [Path(p) for p in args.flows], [Path(p) for p in args.labels]
    missing = [p for p in flow_csvs + label_csvs if not p.exists()]
    if missing:
        print("Files not found:")
        for m in missing: print(f"  {m}")
        return

    cfg = Config(num_rounds=args.rounds, local_epochs=args.local_epochs,
                 batch_size=args.batch_size, lr=args.lr, mu=args.mu, delta=args.delta,
                 split=args.split, max_per_class=args.max_per_class,
                 min_class_flows=args.min_class_flows,
                 attacker_ips=tuple(args.attacker_ips),
                 drop_ip_prefixes=tuple(args.drop_ip_prefix),
                 background_as_benign=args.background_as_benign,
                 use_class_weight=not args.no_class_weight,
                 calib=args.calib, target_far=args.target_far)
    specs = build_specs(args.sigmas, args.target_eps, central_dp=not args.no_central_dp)
    models = ['mlp', 'cnn'] if args.model == 'both' else [args.model]

    rows: List[Dict] = []
    for model_type in models:
        cfg.model_type = model_type
        print(f"\n{'#' * 68}\n#  MODEL: {model_type.upper()}\n{'#' * 68}")
        for run in range(args.runs):
            cfg.seed = args.first_seed + run
            verbose = (run == 0)
            _SIGMA_CACHE.clear()
            print(f"\n  {'─' * 60}\n  Run {run + 1}/{args.runs}  (seed={cfg.seed})\n  {'─' * 60}")
            data = prepare_data(flow_csvs, label_csvs, cfg, verbose=verbose)
            rows += run_suite(cfg, data, specs, verbose)
            pd.DataFrame(rows).to_csv(args.out, index=False)   # saves partial results

    df = pd.DataFrame(rows)
    print_summary(df, args.runs)
    print(f"\nPer-run results saved to {args.out}")


if __name__ == '__main__':
    main()