"""
07_ids_5gnidd_fl_dp.py — the same FL + DP pipeline of 05_ids_flow_fl_dp.py applied to 5G-NIDD
=================================================================================================
Imports 05_ids_flow_fl_dp.py for the models.

Reads the 5G-NIDD Combined.csv (Argus flows, already labeled) and runs the same
configurations as 05_ids_flow_fl_dp.py: Central | Central+clip | Central+DP | FL |
FL+clip | FL+DP(σ), with the same models, the same composed ε, and the same MIA.

TO RUN:
  python 07_ids_5gnidd_fl_dp.py --csv Combined.csv --runs 5 --sigmas 0.5 1.0 2.0
  python 07_ids_5gnidd_fl_dp.py --csv Combined.csv --task type --clients 5 --alpha 0.5
  python 07_ids_5gnidd_fl_dp.py --csv BTS_1.csv BTS_2.csv          # 2 natural clients
  python 07_ids_5gnidd_fl_dp.py --csv Combined.csv --describe      # only describes the data
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset


def load_v2():
    path = Path(__file__).with_name('05_ids_flow_fl_dp.py')
    spec = importlib.util.spec_from_file_location('ids_v3', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['ids_v3'] = mod
    spec.loader.exec_module(mod)
    # the simulator workers (Ray) cannot import a file whose name starts with a
    # digit; this ships the module code together with the objects.
    for name in ('ray.cloudpickle', 'cloudpickle'):
        try:
            importlib.import_module(name).register_pickle_by_value(mod)
        except Exception:
            pass
    return mod


v2 = load_v2()

# ============================ FEATURES ======================================
LABEL_COLS = ['Label', 'Attack Type', 'Attack Tool']
DROP_ALWAYS = ['Unnamed: 0', 'Seq', 'Offset', 'RunTime', 'Mean', 'Sum', 'Min', 'Max',
               'SrcTCPBase', 'DstTCPBase', 'sVid', 'dVid', 'sDSb', 'dDSb']
TTL_COLS = ['sTtl', 'dTtl', 'sHops', 'dHops']
CATEGORICAL = ['Proto', 'State', 'Cause']
MAX_CATEGORIES = 12           


def read_csv(path: Path, task: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    target = 'Label' if task == 'binary' else 'Attack Type'
    if target not in df.columns:
        raise SystemExit(f"{path}: column '{target}' not found. Columns: {list(df.columns)}")
    df = df.dropna(subset=[target]).reset_index(drop=True)
    y = df[target].astype(str).str.strip()
    df['__y'] = np.where(y.str.lower() == 'benign', 'benign', y)
    df['__src'] = path.name
    return df


def build_features(df: pd.DataFrame, keep_ttl: bool, vocab: Dict[str, List[str]] = None
                   ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    drop = set(DROP_ALWAYS + LABEL_COLS + ['__y', '__src']) | (set() if keep_ttl else set(TTL_COLS))
    num_cols = [c for c in df.columns if c not in drop and c not in CATEGORICAL]
    X = df[num_cols].apply(pd.to_numeric, errors='coerce')
    X = X.loc[:, X.notna().any()]                     
    X = X.fillna(0.0)
    X = np.log1p(X.clip(lower=0))                      
    vocab = dict(vocab or {})
    parts = [X]
    for c in CATEGORICAL:
        if c not in df.columns:
            continue
        s = df[c].astype(str).str.strip().str.lower().replace({'nan': 'ausente'})
        if c not in vocab:
            vocab[c] = list(s.value_counts().index[:MAX_CATEGORIES])
        s = s.where(s.isin(vocab[c]), 'outro')
        parts.append(pd.DataFrame({f"{c}={v}": (s == v).astype(np.float32)
                                   for v in vocab[c] + ['outro']}, index=df.index))
    return pd.concat(parts, axis=1).astype(np.float32), vocab


def describe(df: pd.DataFrame, feats: pd.DataFrame) -> None:
    print(f"\n{len(df):,} flows | {feats.shape[1]} features")
    print("\nClasses (Label × Attack Type):")
    cols = [c for c in ('Label', 'Attack Type') if c in df.columns]
    print(df.groupby(cols).size().rename('flows').to_string())
    dup = feats.assign(__y=df['__y'].values).duplicated()
    print(f"\nDuplicate rows (identical features + label): {dup.sum():,} ({dup.mean():.1%})")
    target = df['Attack Type'] if 'Attack Type' in df.columns else df['__y']
    tab = pd.DataFrame({'flows': target.value_counts(),
                        'distinct': target[~dup.values].value_counts()}).fillna(0).astype(int)
    tab['% distinct'] = (100 * tab['distinct'] / tab['flows']).round(1)
    print("Distinct flows per class (what remains without duplicates):")
    print(tab.to_string())
    ben = df['__y'] == 'benign'
    print("\nTopology columns (possible shortcut) — most common values per class:")
    for c in TTL_COLS:
        if c in df.columns:
            print(f"  {c}: benign={df.loc[ben, c].value_counts().head(4).to_dict()} | "
                  f"attack={df.loc[~ben, c].value_counts().head(4).to_dict()}")
    if 'Proto' in df.columns:
        print("\nProtocol × class:")
        print(pd.crosstab(df['Proto'], np.where(ben, 'benign', 'attack')).to_string())


# ============================ PARTITION =====================================
def dirichlet_partition(y: np.ndarray, k: int, alpha: float, rng: np.random.Generator
                        ) -> List[np.ndarray]:
    """Non-IID partition: for each class, the proportions among clients ~ Dirichlet(alpha)."""
    parts = [[] for _ in range(k)]
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        cuts = (np.cumsum(rng.dirichlet(np.full(k, alpha)))[:-1] * len(idx)).astype(int)
        for i, chunk in enumerate(np.split(idx, cuts)):
            parts[i].extend(chunk.tolist())
    return [np.array(sorted(p), dtype=int) for p in parts]


def cap_per_class(idx: np.ndarray, y: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    if not cap:
        return idx
    keep = []
    for cls in np.unique(y[idx]):
        c = idx[y[idx] == cls]
        keep.append(rng.choice(c, cap, replace=False) if len(c) > cap else c)
    return np.concatenate(keep)


def prepare_data(df: pd.DataFrame, feats: pd.DataFrame, cfg, args, verbose: bool) -> Dict:
    rng = np.random.default_rng(cfg.seed)
    X, y, src = feats.values, df['__y'].values, df['__src'].values

    counts = pd.Series(y).value_counts()
    rare = counts[counts < args.min_class_flows].index
    if len(rare):
        if verbose:
            print(f"  [warning] classes removed (< {args.min_class_flows} flows): "
                  f"{counts[rare].to_dict()}")
        m = ~np.isin(y, rare)
        X, y, src = X[m], y[m], src[m]

    if len(np.unique(y)) < 2:
        raise SystemExit("Fewer than 2 classes remain after the filters. Run with --describe to see "
                         "the counts; adjust --min-class-flows or use --keep-duplicates.")
    idx = np.arange(len(y))
    strat = y + '|' + src                               
    vc = pd.Series(strat).value_counts()
    strat = np.where(pd.Series(strat).map(vc).values < 5, y, strat)
    rest, te = train_test_split(idx, test_size=cfg.test_frac, stratify=strat, random_state=cfg.seed)
    tr, ho = train_test_split(rest, test_size=cfg.holdout_frac, stratify=strat[rest],
                              random_state=cfg.seed)

    files = sorted(set(src))
    if len(files) > 1:                                 
        client_idx = [tr[src[tr] == f] for f in files]
        names = files
    else:
        client_idx = [tr[p] for p in dirichlet_partition(y[tr], args.clients, args.alpha, rng)]
        names = [f"client {i}" for i in range(args.clients)]
    client_idx = [cap_per_class(ci, y, cfg.max_per_class, rng) for ci in client_idx]
    keep = [(n, ci) for n, ci in zip(names, client_idx) if len(ci) >= args.min_client_flows]
    if len(keep) < len(names) and verbose:
        print(f"  [warning] {len(names) - len(keep)} client(s) with < {args.min_client_flows} "
              f"flows were discarded")
    names, client_idx = [n for n, _ in keep], [ci for _, ci in keep]
    tr_used = np.concatenate(client_idx)                

    le = LabelEncoder().fit(y)
    cfg.n_classes, cfg.class_names, cfg.in_dim = len(le.classes_), list(le.classes_), X.shape[1]
    mean, std = v2.federated_standardizer([X[ci] for ci in client_idx])

    def to_ds(ix):
        return TensorDataset(torch.tensor((X[ix] - mean) / std, dtype=torch.float32),
                             torch.tensor(le.transform(y[ix]), dtype=torch.long))

    if verbose:
        print(f"  train={len(tr_used):,} holdout={len(ho):,} test={len(te):,} | "
              f"{len(client_idx)} clients | classes: {cfg.class_names}")
        for n, ci in zip(names, client_idx):
            print(f"    {n}: {len(ci):,} flows | {pd.Series(y[ci]).value_counts().to_dict()}")
    return dict(client_ds=[to_ds(ci) for ci in client_idx], train_all=to_ds(tr_used),
                hold_ds=to_ds(ho), test_ds=to_ds(te))


# ====================== NON-PRIVATE REFERENCES ==============================
def run_baselines(cfg, data: Dict) -> List[Dict]:
    Xtr, ytr = (t.numpy() for t in data['train_all'].tensors)
    Xte, yte = (t.numpy() for t in data['test_ds'].tensors)
    models = {
        'RF': RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                               n_jobs=-1, random_state=cfg.seed),
        'LogReg': LogisticRegression(max_iter=2000, class_weight='balanced'),
    }
    rows = []
    b = cfg.class_names.index('benign') if 'benign' in cfg.class_names else None
    for name, clf in models.items():
        t0 = time.perf_counter()
        clf.fit(Xtr, ytr)
        wall = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = clf.predict(Xte)
        infer = 1e6 * (time.perf_counter() - t0) / max(len(yte), 1)
        row = dict(model=name, config='Central (referência)', seed=cfg.seed, sigma=0.0,
                   eps=float('nan'), eps_median=float('nan'),
                   accuracy=float((pred == yte).mean()),
                   macro_f1=float(f1_score(yte, pred, average='macro', zero_division=0)),
                   benign_f1=float('nan'), benign_recall=float('nan'), far=float('nan'),
                   det_rate=float('nan'), mia_auc_benign=float('nan'),
                   mia_tpr1_benign=float('nan'), mia_n_benign=0,
                   mia_auc_all=float('nan'), mia_tpr1_all=float('nan'),
                   train_wall_s=wall, client_train_s=wall, client_parallel_s=wall,
                   infer_us_per_flow=infer)
        if b is not None:
            is_b, pred_b = (yte == b), (pred == b)
            row.update(benign_f1=float(f1_score(is_b, pred_b, zero_division=0)),
                       benign_recall=float((is_b & pred_b).sum() / max(is_b.sum(), 1)),
                       far=float((is_b & ~pred_b).sum() / max(is_b.sum(), 1)),
                       det_rate=float((~is_b & ~pred_b).sum() / max((~is_b).sum(), 1)))
        print(f"  >> {name}: macro-F1={row['macro_f1']:.3f}  benign F1={row['benign_f1']:.3f}  "
              f"FAR={row['far']:.3f}  DR={row['det_rate']:.3f}")
        rows.append(row)
    return rows


# ============================== MAIN ========================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', nargs='+', required=True,
                    help='Combined.csv, or several CSVs (each one becomes a client)')
    ap.add_argument('--task', choices=['binary', 'type'], default='binary',
                    help="binary = Label (benign × malicious); type = Attack Type")
    ap.add_argument('--clients', type=int, default=5)
    ap.add_argument('--alpha', type=float, default=0.5,
                    help='Dirichlet: smaller = more different clients (0.1 very non-IID, 100 ≈ IID)')
    ap.add_argument('--model', choices=['mlp', 'cnn', 'cnn_gmp', 'both'], default='both')
    ap.add_argument('--runs', type=int, default=1)
    ap.add_argument('--sigmas', type=float, nargs='+', default=[1.0])
    ap.add_argument('--mu', type=float, default=0.0)
    ap.add_argument('--rounds', type=int, default=10)
    ap.add_argument('--local-epochs', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--delta', type=float, default=1e-6)
    ap.add_argument('--max-per-class', type=int, default=10_000,
                    help='cap per class on the training set of each client (0 = no cap)')
    ap.add_argument('--min-class-flows', type=int, default=50)
    ap.add_argument('--min-client-flows', type=int, default=200)
    ap.add_argument('--keep-ttl', action='store_true', help='keeps sTtl/dTtl/sHops/dHops')
    ap.add_argument('--keep-duplicates', action='store_true')
    ap.add_argument('--no-baselines', action='store_true')
    ap.add_argument('--describe', action='store_true', help='only describes the data and exits')
    ap.add_argument('--first-seed', type=int, default=42)
    ap.add_argument('--target-eps', type=float, nargs='*', default=[])
    ap.add_argument('--calib', choices=['youden', 'far'], default='youden')
    ap.add_argument('--target-far', type=float, default=0.01)
    ap.add_argument('--no-central-dp', action='store_true')
    ap.add_argument('--out', default='results_07_5gnidd_v3.csv')
    args = ap.parse_args()

    paths = [Path(p) for p in args.csv]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"Files not found: {missing}")
    df = pd.concat([read_csv(p, args.task) for p in paths], ignore_index=True)
    feats, _ = build_features(df, args.keep_ttl)
    print(f"Read: {len(df):,} flows from {len(paths)} file(s) | task={args.task} | "
          f"TTL/hops {'kept' if args.keep_ttl else 'removed'}")
    if args.describe:
        describe(df, feats)
        return
    if not args.keep_duplicates:
        dup = feats.assign(__y=df['__y'].values, __s=df['__src'].values).duplicated().values
        print(f"Duplicates removed: {dup.sum():,} ({dup.mean():.1%}) → {(~dup).sum():,} flows")
        print("  per class after removal: "
              + str(df.loc[~dup, '__y'].value_counts().to_dict()))
        df, feats = df[~dup].reset_index(drop=True), feats[~dup].reset_index(drop=True)

    cfg = v2.Config(num_rounds=args.rounds, local_epochs=args.local_epochs,
                    batch_size=args.batch_size, lr=args.lr, mu=args.mu, delta=args.delta,
                    max_per_class=args.max_per_class,
                    calib=args.calib, target_far=args.target_far)
    specs = v2.build_specs(args.sigmas, args.target_eps, central_dp=not args.no_central_dp)
    models = ['mlp', 'cnn'] if args.model == 'both' else [args.model]

    rows: List[Dict] = []
    for run in range(args.runs):
        cfg.seed = args.first_seed + run
        print(f"\n{'─' * 64}\n  Run {run + 1}/{args.runs}  (seed={cfg.seed})\n{'─' * 64}")
        data = prepare_data(df, feats, cfg, args, verbose=(run == 0))
        if not args.no_baselines:
            rows += run_baselines(cfg, data)
        for model_type in models:
            cfg.model_type = model_type
            print(f"\n  ### MODEL: {model_type.upper()}")
            rows += v2.run_suite(cfg, data, specs, verbose=(run == 0))
            pd.DataFrame(rows).to_csv(args.out, index=False)

    out = pd.DataFrame(rows)
    out['model'] = pd.Categorical(out['model'], categories=list(dict.fromkeys(out['model'])))
    v2.print_summary(out.sort_values('model', kind='stable'), args.runs)
    print(f"\nPer-run results saved to {args.out}")


if __name__ == '__main__':
    main()