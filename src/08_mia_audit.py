"""
08_mia_audit.py — privacy audit (MIA) of Central | FL | FL+DP
=========================================================================
PROTOCOL (LiRA, Carlini et al. 2022; same design as Duong et al. 2026):
  - The pool = training data of each client. Flows with exact copies are either
    removed (--dup-mode drop) or always kept in training and not audited
    (--dup-mode always-in, as in Steinke et al. 2023; used in the paper).
  - K random masks: in each (client, class), half of the audited pool is a member
    (IN) and half a non-member (OUT). The SAME masks are used for all
    configurations → paired comparison Central × FL × FL+DP.
  - For each mask k, one model of each configuration is trained. Each model is the
    target once; the other K-1 are the shadow (reference) models.
  - The audit models are trained on fewer data than the IDS: their ε is recomputed
    here (it is larger than the ε of the main experiment with the same σ).
  - FedAvg runs in-process (without Ray): same weighted aggregation by n, same
    train_local, and same seeds as 05. This makes it possible to train many models
    and to store the local updates of each round.

TO RUN (K = number of models; 16 is a good compromise):
  python 08_mia_audit.py --dataset cve5g --models 16 --sigmas 0.5 1.0 2.0
  python 08_mia_audit.py --dataset 5gnidd --csv Combined.csv --models 16
  python 08_mia_audit.py --dataset cve5g --models 8 --model mlp --local-baseline
  e.g., as in the paper (add --model cnn --no-overfit-control for the 1D-CNN):
  python 08_mia_audit.py --dataset cve5g --models 16 --sigmas 0.5 1 2 --dup-mode always-in
  python 08_mia_audit.py --dataset 5gnidd --csv Combined.csv --models 16 --sigmas 0.5 1 2
Outputs: <out>.csv (one row per configuration × attack × subset),
         <out>_cost.csv (time, parameters, bytes per round), <out>_roc.npz.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from torch.utils.data import TensorDataset

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


def _load(name: str, fname: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v2 = _load('ids_v2', '05_ids_flow_fl_dp.py')
DEVICE = v2.DEVICE
_trapz = getattr(np, 'trapezoid', None) or np.trapz      


# ============================ CONFIGURATIONS ================================
@dataclass
class AuditSpec:
    name: str
    federated: bool = False
    dp: bool = False
    sigma: float = 0.0
    local_only: bool = False      
    overfit: bool = False         


def build_audit_specs(sigmas: List[float], overfit: bool, local: bool) -> List[AuditSpec]:
    # the names 'Local (sem FL)' (= without FL) and 'Overfit (controle +)' (= positive
    # control) are kept as in the published CSVs
    specs = [AuditSpec('Central'),
             AuditSpec(f'Central+DP s={sigmas[0]:g}', dp=True, sigma=sigmas[0]),
             AuditSpec('FL', federated=True)]
    specs += [AuditSpec(f'FL+DP s={s:g}', federated=True, dp=True, sigma=s) for s in sigmas]
    if local:
        specs.append(AuditSpec('Local (sem FL)', local_only=True))
    if overfit:
        specs.append(AuditSpec('Overfit (controle +)', overfit=True))
    return specs


OVERFIT_EPOCH_MULT = 10


def fresh_model(cfg, spec: AuditSpec) -> nn.Module:
    if spec.overfit:
        m = v2.FlowMLP(cfg.in_dim, cfg.n_classes, hidden=512)
        for mod in m.modules():
            if isinstance(mod, nn.Dropout):
                mod.p = 0.0
        return m.to(DEVICE)
    return v2.make_model(cfg).to(DEVICE)


def _seed(s: int) -> None:
    np.random.seed(s % (2 ** 32)); torch.manual_seed(s)


def _concat(dss: List[TensorDataset]) -> TensorDataset:
    return TensorDataset(torch.cat([d.tensors[0] for d in dss]),
                         torch.cat([d.tensors[1] for d in dss]))


# ============================== TRAINING ====================================
def train_target(cfg, spec: AuditSpec, client_sets: List[TensorDataset], seed: int,
                 record: bool) -> Dict:
    """
    Returns dict(final=[weights] (1 per client if local_only), rounds=[(global, [locals])]).
    FedAvg reproduces the SavingFedAvg of 05: average weighted by the number of samples.
    """
    mult = OVERFIT_EPOCH_MULT if spec.overfit else 1
    total_epochs = cfg.num_rounds * cfg.local_epochs * mult
    if not spec.federated:
        if spec.local_only:
            finals = []
            for c, ds in enumerate(client_sets):
                _seed(seed * 10_007 + c)
                m = fresh_model(cfg, spec)
                v2.train_local(m, ds, cfg, spec.dp, spec.sigma, epochs=total_epochs)
                finals.append(v2.get_weights(m))
            return dict(final=finals, rounds=None)
        _seed(seed)
        m = fresh_model(cfg, spec)
        v2.train_local(m, _concat(client_sets), cfg, spec.dp, spec.sigma, epochs=total_epochs)
        return dict(final=[v2.get_weights(m)], rounds=None)

    _seed(seed)
    w = v2.get_weights(fresh_model(cfg, spec))
    n = np.array([len(d) for d in client_sets], dtype=np.float64)
    rounds = []
    for r in range(1, cfg.num_rounds + 1):
        local_w = []
        for c, ds in enumerate(client_sets):
            m = fresh_model(cfg, spec)             
            _seed(seed * 10_007 + c * 101 + r)     
            v2.set_weights(m, w)
            v2.train_local(m, ds, cfg, spec.dp, spec.sigma,
                           epochs=cfg.local_epochs * mult, mu=cfg.mu)
            local_w.append(v2.get_weights(m))
        if record:
            rounds.append((w, local_w))
        w = [sum(n[c] * local_w[c][i] for c in range(len(client_sets))) / n.sum()
             for i in range(len(w))]
    return dict(final=[w], rounds=rounds)


# ======================== PER-EXAMPLE STATISTICS ============================
@torch.no_grad()
def logits_of(cfg, spec: AuditSpec, weights, X: torch.Tensor) -> torch.Tensor:
    m = fresh_model(cfg, spec)
    v2.set_weights(m, weights)
    m.eval()
    out = [m(X[i:i + 8192].to(DEVICE)).double().cpu() for i in range(0, len(X), 8192)]
    return torch.cat(out)


def example_stats(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, np.ndarray]:
    """loss, φ = log(p_y/(1−p_y)) (LiRA scale), and modified entropy."""
    ar = torch.arange(len(y))
    logp = F.log_softmax(logits, 1)
    lp_y = logp[ar, y]
    z = logits.clone()
    zy = z[ar, y].clone()
    z[ar, y] = -float('inf')
    phi = zy - torch.logsumexp(z, 1)                   
    p = logp.exp()
    log1m = torch.log1p(-p.clamp(max=1 - 1e-12))
    p_y = p[ar, y]
    others = -(p * log1m).sum(1) + p_y * log1m[ar, y]   # Σ_{j≠y} −p_j log(1−p_j)
    mentr = -(1 - p_y) * lp_y + others
    return dict(loss=(-lp_y).numpy(), phi=phi.numpy(), mentr=mentr.numpy())


def loss_vec(cfg, spec, weights, X, y) -> np.ndarray:
    return F.cross_entropy(logits_of(cfg, spec, weights, X), y, reduction='none').numpy()


# ============================== ATTACKS =====================================
def lira_scores(stat: np.ndarray, masks: np.ndarray, t: int, fix_var: bool = True
                ) -> Dict[str, np.ndarray]:
    """
    LiRA (Carlini et al. 2022). stat: (K, N) statistic per model and example;
    masks: (K, N) bool (True = IN). Target = model t; shadows = the others.
    fix_var=True uses a global variance (recommended with few shadows).
    """
    sh = np.delete(np.arange(stat.shape[0]), t)
    S, M = stat[sh], masks[sh]
    n_in, n_out = M.sum(0), (~M).sum(0)
    mu_in = np.where(n_in > 0, (S * M).sum(0) / np.maximum(n_in, 1), np.nan)
    mu_out = np.where(n_out > 0, (S * ~M).sum(0) / np.maximum(n_out, 1), np.nan)
    if fix_var:
        r_in = (S - mu_in)[M]
        r_out = (S - mu_out)[~M]
        sd_in = np.full(S.shape[1], np.nanstd(r_in) + 1e-8)
        sd_out = np.full(S.shape[1], np.nanstd(r_out) + 1e-8)
    else:
        sd_in = np.sqrt((((S - mu_in) ** 2) * M).sum(0) / np.maximum(n_in - 1, 1)) + 1e-8
        sd_out = np.sqrt((((S - mu_out) ** 2) * ~M).sum(0) / np.maximum(n_out - 1, 1)) + 1e-8
    x = stat[t]
    off = (x - mu_out) / sd_out
    on = stats.norm.logpdf(x, mu_in, sd_in) - stats.norm.logpdf(x, mu_out, sd_out)
    on = np.where(np.isnan(on), off, on)                # example without IN shadows → offline
    return dict(off=np.nan_to_num(off), on=np.nan_to_num(on), mu_out=mu_out)


def tpr_at(fpr: np.ndarray, tpr: np.ndarray, a: float) -> float:
    m = fpr <= a
    return float(tpr[m].max()) if m.any() else 0.0


def eps_lower_bound(lab: np.ndarray, sc: np.ndarray, delta: float,
                    rng: np.random.Generator, conf: float = 0.95) -> float:
    """
    Empirical ε (lower bound; Jagielski et al. 2020, Nasr et al. 2021):
    threshold chosen on half of the examples, Clopper-Pearson on the other half.
    """
    idx = rng.permutation(len(lab))
    A, B = idx[: len(idx) // 2], idx[len(idx) // 2:]
    a = (1 - conf) / 2

    def bound(l, s, thr):
        pos, neg = l == 1, l == 0
        tp, fp = int((s[pos] >= thr).sum()), int((s[neg] >= thr).sum())
        npos, nneg = int(pos.sum()), int(neg.sum())
        if npos == 0 or nneg == 0:
            return 0.0
        tpr_lo = stats.beta.ppf(a, tp, npos - tp + 1) if tp > 0 else 0.0
        fpr_hi = stats.beta.ppf(1 - a, fp + 1, nneg - fp) if fp < nneg else 1.0
        e1 = math.log((tpr_lo - delta) / fpr_hi) if tpr_lo > delta and fpr_hi > 0 else 0.0
        tnr_lo, fnr_hi = 1 - fpr_hi, 1 - tpr_lo
        e2 = math.log((tnr_lo - delta) / fnr_hi) if tnr_lo > delta and fnr_hi > 0 else 0.0
        return max(0.0, e1, e2)

    cands = np.unique(np.quantile(sc[A], np.linspace(0.0, 1.0, 201)))
    best = max(cands, key=lambda thr: bound(lab[A], sc[A], thr))
    return bound(lab[B], sc[B], best)


def dp_theory(eps: float, delta: float) -> Dict[str, float]:
    """What (ε, δ)-DP guarantees against ANY membership attack."""
    if not np.isfinite(eps):
        return dict(th_tpr1=1.0, th_tpr01=1.0, th_adv=1.0, th_auc=1.0)
    e = math.exp(eps)
    f = lambda a: np.minimum(1.0, np.minimum(e * a + delta, 1 - (1 - delta - a) / e))
    grid = np.linspace(0, 1, 20_001)
    return dict(th_tpr1=float(f(0.01)), th_tpr01=float(f(0.001)),
                th_adv=(e - 1 + 2 * delta) / (e + 1), th_auc=float(_trapz(f(grid), grid)))


def roc_summary(lab: np.ndarray, sc: np.ndarray) -> Dict[str, float]:
    fpr, tpr, _ = roc_curve(lab, sc)
    nneg = int((lab == 0).sum())
    return dict(auc=float(roc_auc_score(lab, sc)), tpr1=tpr_at(fpr, tpr, 0.01),
                tpr01=tpr_at(fpr, tpr, 0.001) if nneg >= 1000 else float('nan'),
                adv=float((tpr - fpr).max()), n_neg=nneg)


# ============================== AUDIT =======================================
def make_pool(data: Dict, verbose: bool, dup_mode: str = 'drop'):
    """
    Audit pool built from each client's training set. Flows with exact copies
    (features + label) make membership ambiguous (identical member and non-member).
      drop      : removes the copies (keeps 1 of each); all examples are audited.
                  Changes the composition of the training set (CVE5G: 40,478 → 15,870 flows).
      always-in : keeps ALL flows; those with a copy are always in training
                  (fixed = True) and are not audited; only the unique flows are
                  drawn as member/non-member (design of Steinke et al. 2023).
                  Training stays equal to that of the main IDS, minus half of the unique flows.
    Returns (X, y, owner, fixed).
    """
    Xs, ys, own = [], [], []
    for c, ds in enumerate(data['client_ds']):
        Xs.append(ds.tensors[0]); ys.append(ds.tensors[1])
        own.append(torch.full((len(ds),), c, dtype=torch.long))
    X, y, owner = torch.cat(Xs), torch.cat(ys), torch.cat(own)
    key = np.concatenate([X.numpy().round(6), y.numpy()[:, None]], 1)
    _, first, inv, cnt = np.unique(key, axis=0, return_index=True, return_inverse=True,
                                   return_counts=True)
    if dup_mode == 'always-in':
        fixed = cnt[inv.ravel()] > 1
        if verbose:
            print(f"  pool: {len(X):,} flows; {int(fixed.sum()):,} have an exact copy and are "
                  f"always in training; {int((~fixed).sum()):,} unique flows are audited")
        return X, y, owner.numpy(), fixed
    keep = np.sort(first)
    if verbose:
        print(f"  pool: {len(X):,} → {len(keep):,} examples without exact duplicates "
              f"({1 - len(keep) / len(X):.1%} removed)")
    return X[keep], y[keep], owner[keep].numpy(), np.zeros(len(keep), dtype=bool)


def add_canaries(X: torch.Tensor, y: torch.Tensor, owner: np.ndarray, cfg, frac: float,
                 min_per_client: int, kind: str, rng: np.random.Generator,
                 count_mask: Optional[np.ndarray] = None):
    """
    Canaries for auditing (Jagielski et al. 2020; Nasr et al. 2021): outlier records
    that enter training with prob. 1/2. They measure what the model CAN memorize about
    a unique record (e.g., a subscriber with unusual traffic), even when the natural
    data are too redundant to leak.
      ood      : features uniform in [−4, 4] (standardized space), random label
      mislabel : copy of a benign flow of the client (+ small noise) with an attack label
    Each client receives max(min_per_client, frac × number of audited flows of the client)
    (count_mask = flows that count; default: all).
    """
    if frac <= 0:
        return X, y, owner, np.zeros(len(y), dtype=bool)
    d, nc = X.shape[1], cfg.n_classes
    b = cfg.class_names.index('benign') if 'benign' in cfg.class_names else None
    Xc, yc, oc = [], [], []
    for c in np.unique(owner):
        base = (owner == c) if count_mask is None else ((owner == c) & count_mask)
        n = max(min_per_client, int(round(frac * base.sum())))
        if kind == 'mislabel' and b is not None:
            src = np.where((owner == c) & (y.numpy() == b))[0]
            if len(src) == 0:
                src = np.where(owner == c)[0]
            base = X[rng.choice(src, n, replace=True)].numpy()
            xs = base + rng.normal(0, 0.05, base.shape)
            lab = rng.choice([k for k in range(nc) if k != b], n)
        else:
            xs = rng.uniform(-4, 4, (n, d))
            lab = rng.integers(0, nc, n)
        Xc.append(xs.astype(np.float32)); yc.append(lab); oc.append(np.full(n, c))
    Xc, yc, oc = np.concatenate(Xc), np.concatenate(yc), np.concatenate(oc)
    is_can = np.concatenate([np.zeros(len(y), bool), np.ones(len(yc), bool)])
    return (torch.cat([X, torch.tensor(Xc)]), torch.cat([y, torch.tensor(yc, dtype=torch.long)]),
            np.concatenate([owner, oc]), is_can)


def near_duplicate_report(X: torch.Tensor, y: np.ndarray, class_names: List[str]) -> pd.DataFrame:
    """
    Distance from each flow to its nearest neighbor OF THE SAME CLASS (standardized space).
    If almost every flow has a "twin", member and non-member are indistinguishable and
    no MIA works on the natural data.
    """
    from sklearn.neighbors import NearestNeighbors
    rows = []
    Xn = X.numpy()
    for k, name in enumerate(class_names):
        idx = np.where(y == k)[0]
        if len(idx) < 3:
            continue
        dist, _ = NearestNeighbors(n_neighbors=2).fit(Xn[idx]).kneighbors(Xn[idx])
        nn = dist[:, 1]
        rows.append(dict(class_=name, n=len(idx), median=np.median(nn),
                         p90=np.quantile(nn, 0.9), frac_lt_0_1=(nn < 0.1).mean(),
                         frac_lt_0_5=(nn < 0.5).mean()))
    return pd.DataFrame(rows).set_index('class_')


def make_masks(y: np.ndarray, owner: np.ndarray, K: int, rng: np.random.Generator,
               is_canary: Optional[np.ndarray] = None,
               fixed: Optional[np.ndarray] = None) -> np.ndarray:
    """
    In each (client, class, canary?), half IN and half OUT, drawn K times.
    Examples with fixed = True are IN in all partitions and are not drawn.
    """
    masks = np.zeros((K, len(y)), dtype=bool)
    can = np.zeros(len(y), bool) if is_canary is None else is_canary
    fx = np.zeros(len(y), bool) if fixed is None else fixed
    free = np.where(~fx)[0]
    groups = pd.Series(free).groupby([owner[free], y[free], can[free]]).apply(list)
    for k in range(K):
        for idx in groups:
            idx = rng.permutation(np.asarray(idx))
            masks[k, idx[: len(idx) // 2]] = True
    masks[:, fx] = True
    return masks


def run_audit(cfg, data: Dict, specs: List[AuditSpec], K: int, seed: int = 0,
              verbose: bool = True, canary_frac: float = 0.01, canary_min: int = 10,
              canary_type: str = 'ood', dup_mode: str = 'drop'):
    rng = np.random.default_rng(seed)
    X, y, owner, fixed = make_pool(data, verbose, dup_mode)
    nd = near_duplicate_report(X[~fixed], y.numpy()[~fixed], cfg.class_names)
    X, y, owner, is_can = add_canaries(X, y, owner, cfg, canary_frac, canary_min,
                                       canary_type, rng, count_mask=~fixed)
    fixed = np.concatenate([fixed, np.zeros(len(y) - len(fixed), dtype=bool)])
    yn = y.numpy()
    C = len(data['client_ds'])
    masks = make_masks(yn, owner, K, rng, is_can, fixed)
    b = cfg.class_names.index('benign') if 'benign' in cfg.class_names else None
    if verbose:
        dist = pd.crosstab(pd.Series(owner[~is_can], name='client'),
                           pd.Series(yn[~is_can], name='class').map(dict(enumerate(cfg.class_names))))
        if is_can.any():
            dist[f'canaries ({canary_type})'] = pd.Series(owner[is_can]).value_counts()
        print("  pool per client × class:\n    " + dist.fillna(0).astype(int).to_string()
              .replace("\n", "\n    "))
        print("  nearest neighbor of the same class (distance in the standardized space):\n    "
              + nd.to_string(float_format=lambda v: f"{v:.3f}").replace("\n", "\n    "))

    store: Dict[str, Dict[str, np.ndarray]] = {}
    cost_rows, util_rows = [], []
    for k in range(K):
        sets = [TensorDataset(X[(owner == c) & masks[k]], y[(owner == c) & masks[k]])
                for c in range(C)]
        for spec in specs:
            t0 = time.perf_counter()
            res = train_target(cfg, spec, sets, seed=1000 * seed + k, record=spec.federated)
            dt = time.perf_counter() - t0
            st = store.setdefault(spec.name, {})

            if spec.local_only:                   # each example is evaluated by its client's model
                parts = [example_stats(logits_of(cfg, spec, w, X), y) for w in res['final']]
                ex = {key: np.choose(owner, [p[key] for p in parts]) for key in parts[0]}
            else:
                ex = example_stats(logits_of(cfg, spec, res['final'][0], X), y)
            for key, v in ex.items():
                st.setdefault(key, np.zeros((K, len(yn))))[k] = v

            if res['rounds']:                     # server view: local updates per round
                drop = np.zeros(len(yn)); local_loss = np.zeros((C, len(yn)))
                own_max = np.full(len(yn), -np.inf)       # Nasr et al. 2021, Distinguisher 2
                for w_g, locs in res['rounds']:
                    lg = loss_vec(cfg, spec, w_g, X, y)
                    for c, w_c in enumerate(locs):
                        lc = loss_vec(cfg, spec, w_c, X, y)
                        local_loss[c] += lc
                        m = owner == c
                        drop[m] += (lg - lc)[m]
                        own_max[m] = np.maximum(own_max[m], lc[m])
                R = len(res['rounds'])
                own_mean = local_loss[owner, np.arange(len(yn))] / R
                st.setdefault('srv_drop', np.zeros((K, len(yn))))[k] = drop
                st.setdefault('own_mean', np.zeros((K, len(yn))))[k] = own_mean
                st.setdefault('own_max', np.zeros((K, len(yn))))[k] = own_max
                st.setdefault('sia_pred', np.zeros((K, len(yn)), dtype=int))[k] = local_loss.argmin(0)

            ws = res['final']
            f1s, fars, per_cls = [], [], []
            for w in ws:
                m = fresh_model(cfg, spec); v2.set_weights(m, w)
                ev = v2.evaluate(m, data['test_ds'], cfg)
                f1s.append(ev['macro_f1']); fars.append(ev['far'])
                yt, pt = v2.predict(m, data['test_ds'])
                per_cls.append(f1_score(yt, pt, labels=list(range(cfg.n_classes)), average=None,
                                        zero_division=0))
            per_cls = np.mean(per_cls, 0)
            util_rows.append(dict(config=spec.name, k=k, macro_f1=np.mean(f1s), far=np.mean(fars),
                                  **{f"f1_{n}": v for n, v in zip(cfg.class_names, per_cls)}))
            n_params = sum(int(np.prod(a.shape)) for a in ws[0])
            cost_rows.append(dict(config=spec.name, k=k, train_s=dt, n_params=n_params,
                                  bytes_per_round=(2 * C * n_params * 4 if spec.federated else 0)))
            if verbose:
                cls = "  ".join(f"{n.replace('attack_', '')[:10]}={v:.2f}" for n, v in zip(cfg.class_names, per_cls))
                print(f"  [k={k + 1}/{K}] {spec.name:<22} {dt:6.1f}s  macro-F1={np.mean(f1s):.3f}"
                      f"  | F1: {cls}")

    # ε of the audit models (trained on fewer data than the IDS)
    eps_of = {}
    for spec in specs:
        if not spec.dp or spec.sigma <= 0:
            eps_of[spec.name] = float('inf')
            continue
        tot = cfg.num_rounds * cfg.local_epochs
        sizes = ([int(((owner == c) & masks[0]).sum()) for c in range(C)] if spec.federated
                 else [int(masks[0].sum())])
        eps_of[spec.name] = max(v2.epsilon_for(n, cfg, spec.sigma, tot) for n in sizes)

    subsets = {'all': ~is_can & ~fixed}          # audited natural data (without canaries)
    if b is not None:
        subsets['benign'] = (yn == b) & ~is_can & ~fixed
    if is_can.any():
        subsets['canary'] = is_can
    rows, rocs = [], {}
    for spec in specs:
        st = store[spec.name]
        attacks = {}
        lira_bb = [lira_scores(st['phi'], masks, t) for t in range(K)]
        loss_mu_out = [lira_scores(st['loss'], masks, t)['mu_out'] for t in range(K)]
        attacks['loss'] = [-st['loss'][t] for t in range(K)]
        attacks['mentr'] = [-st['mentr'][t] for t in range(K)]
        attacks['loss_cal'] = [-(st['loss'][t] - np.nan_to_num(loss_mu_out[t])) for t in range(K)]
        attacks['lira_off'] = [r['off'] for r in lira_bb]
        attacks['lira_on'] = [r['on'] for r in lira_bb]
        if 'srv_drop' in st:
            attacks['srv_drop'] = [st['srv_drop'][t] for t in range(K)]
            attacks['srv_lira'] = [lira_scores(st['srv_drop'], masks, t)['on'] for t in range(K)]
            attacks['nasr_mean'] = [-st['own_mean'][t] for t in range(K)]
            attacks['nasr_max'] = [-st['own_max'][t] for t in range(K)]
        th = dp_theory(eps_of[spec.name], cfg.delta)

        for sub, sel in subsets.items():
            for atk, scs in attacks.items():
                per = [roc_summary(masks[t][sel].astype(int), scs[t][sel]) for t in range(K)]
                lab = np.concatenate([masks[t][sel] for t in range(K)]).astype(int)
                sc = np.concatenate([scs[t][sel] for t in range(K)])
                pooled = roc_summary(lab, sc)
                rows.append(dict(
                    config=spec.name, subset=sub, attack=atk, eps_audit=eps_of[spec.name],
                    auc_mean=np.mean([p['auc'] for p in per]),
                    auc_std=np.std([p['auc'] for p in per]),
                    tpr1=pooled['tpr1'], tpr01=pooled['tpr01'], adv=pooled['adv'],
                    eps_lb=eps_lower_bound(lab, sc, cfg.delta, rng),
                    n_per_target=int(sel.sum()), **th))
                fpr, tpr, _ = roc_curve(lab, sc)
                keep = np.unique(np.geomspace(1, len(fpr), 400).astype(int) - 1)
                rocs[f"{spec.name}|{sub}|{atk}"] = np.stack([fpr[keep], tpr[keep]])

        if 'sia_pred' in st:                     # source inference on members only
            for sub, sel in subsets.items():
                if sub == 'canary':
                    continue
                hit, prior = [], []
                for t in range(K):
                    m = masks[t] & sel
                    hit.append((st['sia_pred'][t][m] == owner[m]).mean())
                    tab = pd.crosstab(yn[m], owner[m])           
                    prior.append((tab.max(1).sum() / tab.values.sum()))
                rows.append(dict(config=spec.name, subset=sub, attack='source_inference',
                                 eps_audit=eps_of[spec.name], sia_acc=np.mean(hit),
                                 sia_label_prior=np.mean(prior), sia_random=1.0 / C,
                                 n_per_target=int(sel.sum())))
    util_k = pd.DataFrame(util_rows)
    util = util_k.groupby('config', sort=False)[['macro_f1', 'far']].agg(['mean', 'std'])
    util.columns = ['macro_f1', 'macro_f1_std', 'far', 'far_std']
    cost = pd.DataFrame(cost_rows).groupby('config', sort=False)[
        ['train_s', 'n_params', 'bytes_per_round']].mean()
    return pd.DataFrame(rows), util, cost, rocs, util_k


def print_report(res: pd.DataFrame, util: pd.DataFrame, cost: pd.DataFrame, cfg) -> None:
    W = 118
    subs = [s for s in ('benign', 'canary') if (res['subset'] == s).any()] or ['all']
    titles = {'benign': "natural BENIGN flows (the personal data)",
              'canary': "CANARIES (worst case: a subscriber's outlier record)",
              'all': "all natural flows"}
    for sub in subs:
        print("\n" + "=" * W)
        print(f"MIA AUDIT — {titles[sub]}; the other combinations are in the CSV")
        print("=" * W)
        mia = res[(res['subset'] == sub) & (res['attack'] != 'source_inference')]
        for cfg_name, g in mia.groupby('config', sort=False):
            e = g['eps_audit'].iloc[0]
            u = util.loc[cfg_name]
            print(f"\n{cfg_name}  |  ε(audit)={'∞' if not np.isfinite(e) else f'{e:.2f}'}  "
                  f"|  macro-F1={u['macro_f1']:.3f}±{u['macro_f1_std']:.3f}  FAR={u['far']:.3%}  "
                  f"|  DP bound: AUC≤{g['th_auc'].iloc[0]:.3f}  TPR@1%≤{g['th_tpr1'].iloc[0]:.3f}  "
                  f"adv.≤{g['th_adv'].iloc[0]:.3f}")
            print(f"  {'attack':<10}{'AUC (mean±std)':>18}{'TPR@1%':>10}{'TPR@0.1%':>10}"
                  f"{'advantage':>10}{'empirical ε':>12}")
            for _, r in g.iterrows():
                t01 = "  n/a" if np.isnan(r['tpr01']) else f"{r['tpr01']:.4f}"
                print(f"  {r['attack']:<10}{r['auc_mean']:>11.3f}±{r['auc_std']:.3f}"
                      f"{r['tpr1']:>10.4f}{t01:>10}{r['adv']:>10.3f}{r['eps_lb']:>12.2f}")
    sia = res[res['attack'] == 'source_inference']
    if len(sia):
        print("\nSource inference (the server finds the client that owns a member):")
        for _, r in sia.iterrows():
            print(f"  {r['config']:<22}{r['subset']:<8} accuracy={r['sia_acc']:.3f}  "
                  f"(label only={r['sia_label_prior']:.3f}, random={r['sia_random']:.3f})")
    print("\nCost per model (mean):")
    print("  " + cost.to_string(float_format=lambda v: f"{v:,.1f}").replace("\n", "\n  "))
    print("=" * W)
    print("How to read: TPR@1% = fraction of members found with 1% false positives (random = 0.01).\n"
          "  The DP bound holds for any attack; empirical ≪ bound is expected.\n"
          "  TPR@1% = 0.0000 with AUC ≈ 0.5: many tied scores (loss = 0 for members and non-members).\n"
          "  Natural data without leakage even in the positive control: flows are too redundant; use the canaries.")


# ================================ MAIN ======================================
def load_data(args, cfg) -> Dict:
    if args.dataset == 'cve5g':
        flows = [Path(p) for p in args.flows]
        labels = [Path(p) for p in args.labels]
        cfg.attacker_ips = tuple(args.attacker_ips)
        cfg.drop_ip_prefixes = tuple(args.drop_ip_prefix)
        cfg.background_as_benign = args.background_as_benign
        return v2.prepare_data(flows, labels, cfg, verbose=True)
    m07 = _load('ids_07', '07_ids_5gnidd_fl_dp.py')
    paths = [Path(p) for p in args.csv]
    df = pd.concat([m07.read_csv(p, args.task) for p in paths], ignore_index=True)
    feats, _ = m07.build_features(df, args.keep_ttl)
    dup = feats.assign(__y=df['__y'].values, __s=df['__src'].values).duplicated().values
    df, feats = df[~dup].reset_index(drop=True), feats[~dup].reset_index(drop=True)
    ns = SimpleNamespace(clients=args.clients, alpha=args.alpha,
                         min_class_flows=args.min_class_flows,
                         min_client_flows=args.min_client_flows)
    return m07.prepare_data(df, feats, cfg, ns, verbose=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=['cve5g', '5gnidd'], required=True)
    ap.add_argument('--flows', nargs='+', default=['data/scenario_1/flows_s1_nfstream.csv',
                                                   'data/scenario_2/flows_s2_nfstream.csv',
                                                   'data/scenario_3/flows_s3_nfstream.csv'])
    ap.add_argument('--labels', nargs='+', default=['data/scenario_1/cve5g_s1_labels.csv',
                                                    'data/scenario_2/cve5g_s2_labels.csv',
                                                    'data/scenario_3/cve5g_s3_labels.csv'])
    ap.add_argument('--attacker-ips', nargs='*', default=[])
    ap.add_argument('--drop-ip-prefix', nargs='*', default=[])
    ap.add_argument('--background-as-benign', action='store_true')
    # 5G-NIDD (same defaults as 07)
    ap.add_argument('--csv', nargs='+', default=['Combined.csv'])
    ap.add_argument('--task', choices=['binary', 'type'], default='binary')
    ap.add_argument('--clients', type=int, default=5)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--min-client-flows', type=int, default=200)
    ap.add_argument('--keep-ttl', action='store_true')
    # training
    ap.add_argument('--model', choices=['mlp', 'cnn', 'cnn_gmp'], default='mlp')
    ap.add_argument('--models', type=int, default=16, help='K = number of models per configuration')
    ap.add_argument('--sigmas', type=float, nargs='+', default=[1.0])
    ap.add_argument('--rounds', type=int, default=10)
    ap.add_argument('--local-epochs', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--mu', type=float, default=0.0)
    ap.add_argument('--delta', type=float, default=1e-6)
    ap.add_argument('--max-per-class', type=int, default=10_000)
    ap.add_argument('--min-class-flows', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-overfit-control', action='store_true')
    ap.add_argument('--local-baseline', action='store_true',
                    help="includes 'Local (sem FL)': each client trains alone, without FL")
    ap.add_argument('--canary-frac', type=float, default=0.01,
                    help='canaries per client = max(--canary-min, frac × client pool); 0 disables them')
    ap.add_argument('--canary-min', type=int, default=10)
    ap.add_argument('--canary-type', choices=['ood', 'mislabel'], default='ood')
    ap.add_argument('--dup-mode', choices=['drop', 'always-in'], default='drop',
                    help="flows with an exact copy: 'drop' removes them (old default); 'always-in' "
                         "keeps them always in training and audits only the unique flows")
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    cfg = v2.Config(num_rounds=args.rounds, local_epochs=args.local_epochs,
                    batch_size=args.batch_size, lr=args.lr, mu=args.mu, delta=args.delta,
                    max_per_class=args.max_per_class, min_class_flows=args.min_class_flows,
                    seed=args.seed)
    cfg.model_type = args.model
    data = load_data(args, cfg)
    specs = build_audit_specs(args.sigmas, overfit=not args.no_overfit_control,
                              local=args.local_baseline)
    if args.models < 4:
        raise SystemExit("--models must be ≥ 4 (LiRA uses the other K−1 as shadows)")
    res, util, cost, rocs, util_k = run_audit(
        cfg, data, specs, args.models, seed=args.seed, canary_frac=args.canary_frac,
        canary_min=args.canary_min, canary_type=args.canary_type, dup_mode=args.dup_mode)
    print_report(res, util, cost, cfg)
    out = args.out or f"results_08_mia_{args.dataset}_{args.model}"
    res.to_csv(f"{out}.csv", index=False)
    util.join(cost).to_csv(f"{out}_cost.csv")
    util_k.to_csv(f"{out}_util_por_modelo.csv", index=False)
    np.savez_compressed(f"{out}_roc.npz", **rocs)
    print(f"\nSaved: {out}.csv, {out}_cost.csv, {out}_util_por_modelo.csv, {out}_roc.npz")


if __name__ == '__main__':
    main()