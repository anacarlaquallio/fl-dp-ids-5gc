"""
09_mia_art.py — MIA with an off-the-shelf library (ART) to compare with 08_mia_audit.py
==================================================================================
Uses the Adversarial Robustness Toolbox (ART, IBM / LF AI & Data), a third-party
implementation of the classic membership inference attacks, as a CONTROL for the
in-house implementation in 08. Same data, canaries, member/non-member masks, and
training as 08 (imported from it), so the numbers are directly comparable.

TO RUN (same data arguments as 08):
  pip install adversarial-robustness-toolbox
  python 09_mia_art.py --dataset cve5g --models 4 --sigmas 0.5 1.0 2.0
  python 09_mia_art.py --dataset 5gnidd --csv Combined.csv --models 4
  e.g., as in the paper (add --model cnn --no-overfit-control for the 1D-CNN):
  python 09_mia_art.py --dataset cve5g --models 4 --sigmas 0.5 1 2 --dup-mode always-in
Output: results_09_art_<dataset>_<model>.csv and a summary comparable to that of 08.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
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


m08 = _load('mia08', '08_mia_audit.py')
v2 = m08.v2

try:
    from art.attacks.inference.membership_inference import (
        MembershipInferenceBlackBox, MembershipInferenceBlackBoxRuleBased)
    from art.estimators.classification import PyTorchClassifier
except ImportError:
    raise SystemExit("Install ART: pip install adversarial-robustness-toolbox")


# ============================== ART =========================================
def wrap(cfg, spec, weights) -> PyTorchClassifier:
    model = m08.fresh_model(cfg, spec)
    v2.set_weights(model, weights)
    model.eval()
    return PyTorchClassifier(model=model, loss=nn.CrossEntropyLoss(),
                             input_shape=(cfg.in_dim,), nb_classes=cfg.n_classes,
                             device_type='gpu' if torch.cuda.is_available() else 'cpu')


def _summary(lab: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    fpr, tpr, _ = roc_curve(lab, score)
    m = fpr <= 0.01
    pred = score >= 0.5
    ok = (lab == 0).sum() >= 100
    return dict(auc=float(roc_auc_score(lab, score)),
                tpr1=(float(tpr[m].max()) if m.any() else 0.0) if ok else float('nan'),
                adv=float((tpr - fpr).max()),
                acc=float((pred == lab).mean()))


def art_attacks(clf: PyTorchClassifier, X: np.ndarray, y: np.ndarray, member: np.ndarray,
                rng: np.random.Generator, cap: int) -> Dict[str, Dict[str, float]]:
    """Trains the attack model on half of the members/non-members and evaluates it on the other half."""
    im, io = rng.permutation(np.where(member)[0]), rng.permutation(np.where(~member)[0])
    n = min(len(im), len(io))
    im, io = im[:n], io[:n]                           
    h = n // 2
    tr_in, te_in = im[:min(h, cap)], im[h:h + min(n - h, cap)]
    tr_out, te_out = io[:min(h, cap)], io[h:h + min(n - h, cap)]
    if min(len(tr_in), len(te_in)) < 10:
        return {}
    y = np.eye(clf.nb_classes, dtype=np.float32)[y]
    X_te = np.concatenate([X[te_in], X[te_out]])
    y_te = np.concatenate([y[te_in], y[te_out]])
    lab = np.concatenate([np.ones(len(te_in)), np.zeros(len(te_out))])
    out = {}

    rule = MembershipInferenceBlackBoxRuleBased(clf)
    out['art_rule'] = _summary(lab, rule.infer(X_te, y_te).ravel().astype(float))

    for name, itype, mtype in (('art_rf', 'prediction', 'rf'), ('art_nn_loss', 'loss', 'nn')):
        atk = MembershipInferenceBlackBox(clf, input_type=itype, attack_model_type=mtype,
                                          nn_model_epochs=30)
        atk.fit(X[tr_in], y[tr_in], X[tr_out], y[tr_out])
        prob = atk.infer(X_te, y_te, probabilities=True)
        prob = np.asarray(prob)
        prob = prob[:, -1] if prob.ndim == 2 and prob.shape[1] > 1 else prob.ravel()
        out[name] = _summary(lab, prob.astype(float))
    for v in out.values():
        v['n_per_side'] = len(te_in)
    return out


# ============================== AUDIT =======================================
def run_art_audit(cfg, data: Dict, specs: List, K: int, seed: int = 0, cap: int = 5000,
                  canary_frac: float = 0.01, canary_min: int = 10, canary_type: str = 'ood',
                  verbose: bool = True, dup_mode: str = 'drop') -> pd.DataFrame:
    rng = np.random.default_rng(seed)          # same sequence as 08 → same canaries and masks
    X, y, owner, fixed = m08.make_pool(data, verbose, dup_mode)
    X, y, owner, is_can = m08.add_canaries(X, y, owner, cfg, canary_frac, canary_min,
                                           canary_type, rng, count_mask=~fixed)
    fixed = np.concatenate([fixed, np.zeros(len(y) - len(fixed), dtype=bool)])
    yn = y.numpy()
    masks = m08.make_masks(yn, owner, K, rng, is_can, fixed)
    C = len(data['client_ds'])
    b = cfg.class_names.index('benign') if 'benign' in cfg.class_names else None
    subsets = {'all': ~is_can & ~fixed, 'canary': is_can}
    if b is not None:
        subsets['benign'] = (yn == b) & ~is_can & ~fixed
    Xn = X.numpy()

    rows = []
    arng = np.random.default_rng(seed + 99)
    for k in range(K):
        sets = [TensorDataset(X[(owner == c) & masks[k]], y[(owner == c) & masks[k]])
                for c in range(C)]
        for spec in specs:
            if spec.local_only:
                continue                          # ART attacks a single model
            t0 = time.perf_counter()
            res = m08.train_target(cfg, spec, sets, seed=1000 * seed + k, record=False)
            clf = wrap(cfg, spec, res['final'][0])
            for sub, sel in subsets.items():
                if sel.sum() < 40:
                    continue
                idx = np.where(sel)[0]
                out = art_attacks(clf, Xn[idx], yn[idx], masks[k][idx], arng, cap)
                for atk, s in out.items():
                    rows.append(dict(config=spec.name, k=k, subset=sub, attack=atk, **s))
            if verbose:
                got = {r['attack']: r['auc'] for r in rows
                       if r['config'] == spec.name and r['k'] == k and r['subset'] == 'canary'}
                print(f"  [k={k + 1}/{K}] {spec.name:<22} {time.perf_counter() - t0:6.1f}s  "
                      "canary AUC: " + "  ".join(f"{a}={v:.3f}" for a, v in got.items()))
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = (df.groupby(['subset', 'config', 'attack'], sort=False)
           .agg(auc=('auc', 'mean'), auc_sd=('auc', 'std'), tpr1=('tpr1', 'mean'),
                adv=('adv', 'mean'), acc=('acc', 'mean'), n=('n_per_side', 'mean'))
           .reset_index())
    for sub in ('benign', 'canary', 'all'):
        s = g[g['subset'] == sub]
        if not len(s):
            continue
        print("\n" + "=" * 92)
        print(f"ART — subset '{sub}'  (mean±std AUC across models; chance: AUC 0.5, TPR@1% 0.01)")
        print("=" * 92)
        print(f"{'configuration':<24}{'attack':<14}{'AUC':>14}{'TPR@1%':>10}{'advantage':>10}"
              f"{'accuracy':>10}{'n/side':>8}")
        for _, r in s.iterrows():
            t1 = "n/a" if np.isnan(r['tpr1']) else f"{r['tpr1']:.4f}"
            print(f"{r['config']:<24}{r['attack']:<14}{r['auc']:>8.3f}±{r['auc_sd']:.3f}"
                  f"{t1:>10}{r['adv']:>10.3f}{r['acc']:>10.3f}{r['n']:>8.0f}")
    print("\nart_rule is binary: AUC = (TPR + TNR)/2 and TPR@1% is usually 0.")
    print("TPR@1% = n/a when there are < 100 non-members in the test set (use --canary-frac 0.03 for more canaries).")
    print("The ART attacker knows half of the members (more informed than the LiRA attacker in 08).")
    return g


# ================================ MAIN ======================================
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
    ap.add_argument('--csv', nargs='+', default=['Combined.csv'])
    ap.add_argument('--task', choices=['binary', 'type'], default='binary')
    ap.add_argument('--clients', type=int, default=5)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--min-client-flows', type=int, default=200)
    ap.add_argument('--keep-ttl', action='store_true')
    ap.add_argument('--model', choices=['mlp', 'cnn', 'cnn_gmp'], default='mlp')
    ap.add_argument('--models', type=int, default=4, help='number of masks (models per configuration)')
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
    ap.add_argument('--canary-frac', type=float, default=0.01)
    ap.add_argument('--canary-min', type=int, default=10)
    ap.add_argument('--canary-type', choices=['ood', 'mislabel'], default='ood')
    ap.add_argument('--cap', type=int, default=5000,
                    help='max. number of examples per side in the attack training and test sets')
    ap.add_argument('--dup-mode', choices=['drop', 'always-in'], default='drop')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    cfg = v2.Config(num_rounds=args.rounds, local_epochs=args.local_epochs,
                    batch_size=args.batch_size, lr=args.lr, mu=args.mu, delta=args.delta,
                    max_per_class=args.max_per_class, min_class_flows=args.min_class_flows,
                    seed=args.seed)
    cfg.model_type = args.model
    data = m08.load_data(args, cfg)
    specs = m08.build_audit_specs(args.sigmas, overfit=not args.no_overfit_control, local=False)
    df = run_art_audit(cfg, data, specs, args.models, seed=args.seed, cap=args.cap,
                       canary_frac=args.canary_frac, canary_min=args.canary_min,
                       canary_type=args.canary_type, dup_mode=args.dup_mode)
    print_summary(df)
    out = args.out or f"results_09_art_{args.dataset}_{args.model}"
    df.to_csv(f"{out}.csv", index=False)
    print(f"\nSaved: {out}.csv")


if __name__ == '__main__':
    main()