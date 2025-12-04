"""
Script for comparing different clustering algorithms.

Usage example:

python compare_cluster_methods.py \
  --ratings_path ml-100k/u.data \
  --users_path ml-100k/u.user \
  --cold_mode pure
"""
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from data_loader import load_movielens_100k
from cold_start_dfp import DFPConfig, DFPRecommender
from evaluate_dfp_coldstart import (
    build_pure_cold_input,
    build_semi_cold_input,
    eval_precision_recall_f1,
)


def run_pure_cold_for_method(
    method: str,
    UM: pd.DataFrame,
    UP: pd.DataFrame,
    args,
) -> Tuple[float, float, float, int, int]:
    kf = KFold(
        n_splits=args.n_fold,
        shuffle=True,
        random_state=args.kfold_seed,
    )

    fold_precisions: List[float] = []
    fold_recalls: List[float] = []
    fold_f1s: List[float] = []
    fold_cells: List[int] = []
    fold_pos_cells: List[int] = []

    print(f"\n[ {method} ] PURE cold-start evaluation")
    print("---------------------------------------------------")

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(UM), start=1):
        train_users = UM.index[train_idx]
        test_users = UM.index[test_idx]

        UM_train = UM.loc[train_users]
        UP_train = UP.loc[train_users]

        UM_test = UM.loc[test_users]
        UP_test_full = UP.loc[test_users]

        cfg = DFPConfig(
            cluster_method=method,
            n_clusters=args.n_clusters,
            min_support=args.min_support,
            theta=args.theta,
            phi=args.phi,
            ct=args.ct,
            max_itemset_len=args.max_itemset_len,
            knn_k=args.knn_k,
            louvain_resolution=args.louvain_resolution,
        )
        model = DFPRecommender(cfg)
        model.fit(UM_train, UP_train)

        UP_test_input, _ = build_pure_cold_input(UP_test_full)
        UP_test_pred, _ = model.transform(UM_test, UP_test_input)

        prec, rec, f1, num_cells, num_pos = eval_precision_recall_f1(
            UP_test_full, UP_test_pred, UP_test_input
        )

        print(
            f"Fold {fold_idx:2d}: "
            f"prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}, "
            f"#cells={num_cells}, #pos={num_pos}"
        )

        fold_precisions.append(prec)
        fold_recalls.append(rec)
        fold_f1s.append(f1)
        fold_cells.append(num_cells)
        fold_pos_cells.append(num_pos)

    mean_prec = float(np.mean(fold_precisions))
    mean_rec = float(np.mean(fold_recalls))
    mean_f1 = float(np.mean(fold_f1s))
    total_cells = int(np.sum(fold_cells))
    total_pos_cells = int(np.sum(fold_pos_cells))

    print(f"[ {method} ] PURE cold-start summary")
    print("---------------------------------------------------")
    print(f"Precision mean: {mean_prec:.4f}")
    print(f"Recall    mean: {mean_rec:.4f}")
    print(f"F1        mean: {mean_f1:.4f}")
    print(f"Total test cells      : {total_cells}")
    print(f"Total positive cells  : {total_pos_cells}")
    print("===================================================\n")

    return mean_prec, mean_rec, mean_f1, total_cells, total_pos_cells


def run_semi_cold_for_method(
    method: str,
    UM: pd.DataFrame,
    UP: pd.DataFrame,
    args,
) -> Tuple[float, float, float, int, int]:
    kf = KFold(
        n_splits=args.n_fold,
        shuffle=True,
        random_state=args.kfold_seed,
    )

    fold_precisions: List[float] = []
    fold_recalls: List[float] = []
    fold_f1s: List[float] = []
    fold_cells: List[int] = []
    fold_pos_cells: List[int] = []

    print(f"\n[ {method} ] SEMI cold-start evaluation (p_delete={args.p_delete:.2f})")
    print("------------------------------------------------------------------")

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(UM), start=1):
        train_users = UM.index[train_idx]
        test_users = UM.index[test_idx]

        UM_train = UM.loc[train_users]
        UP_train = UP.loc[train_users]

        UM_test = UM.loc[test_users]
        UP_test_full = UP.loc[test_users]

        cfg = DFPConfig(
            cluster_method=method,
            n_clusters=args.n_clusters,
            min_support=args.min_support,
            theta=args.theta,
            phi=args.phi,
            ct=args.ct,
            max_itemset_len=args.max_itemset_len,
            knn_k=args.knn_k,
            louvain_resolution=args.louvain_resolution,
        )
        model = DFPRecommender(cfg)
        model.fit(UM_train, UP_train)

        UP_test_input, _ = build_semi_cold_input(
            UP_test_full,
            p=args.p_delete,
            random_state=args.p_seed,
        )
        UP_test_pred, _ = model.transform(UM_test, UP_test_input)

        prec, rec, f1, num_cells, num_pos = eval_precision_recall_f1(
            UP_test_full, UP_test_pred, UP_test_input
        )

        print(
            f"Fold {fold_idx:2d}: "
            f"prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}, "
            f"#cells={num_cells}, #pos={num_pos}"
        )

        fold_precisions.append(prec)
        fold_recalls.append(rec)
        fold_f1s.append(f1)
        fold_cells.append(num_cells)
        fold_pos_cells.append(num_pos)

    mean_prec = float(np.mean(fold_precisions))
    mean_rec = float(np.mean(fold_recalls))
    mean_f1 = float(np.mean(fold_f1s))
    total_cells = int(np.sum(fold_cells))
    total_pos_cells = int(np.sum(fold_pos_cells))

    print(f"[ {method} ] SEMI cold-start summary (p_delete={args.p_delete:.2f})")
    print("------------------------------------------------------------------")
    print(f"Precision mean: {mean_prec:.4f}")
    print(f"Recall    mean: {mean_rec:.4f}")
    print(f"F1        mean: {mean_f1:.4f}")
    print(f"Total test cells      : {total_cells}")
    print(f"Total positive cells  : {total_pos_cells}")
    print("==================================================================\n")

    return mean_prec, mean_rec, mean_f1, total_cells, total_pos_cells


# def plot_method_comparison(
#     methods: List[str],
#     precisions: List[float],
#     recalls: List[float],
#     f1s: List[float],
#     out_path: str,
#     title: str,
# ):
#     x = np.arange(len(methods))
#     width = 0.25

#     plt.figure(figsize=(8, 5))
#     plt.bar(x - width, precisions, width, label="Precision")
#     plt.bar(x, recalls, width, label="Recall")
#     plt.bar(x + width, f1s, width, label="F1")

#     plt.xticks(x, methods)
#     plt.ylabel("Score")
#     plt.title(title)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=300)
#     print(f"[INFO] Saved method comparison plot to: {out_path}")
def plot_method_comparison(
    methods: List[str],
    precisions: List[float],
    recalls: List[float],
    f1s: List[float],
    out_path: str,
    title: str,
):
    x = np.arange(len(methods))

    plt.figure(figsize=(8, 5))

    plt.plot(x, precisions, marker="o", label="Precision")
    plt.plot(x, recalls,    marker="s", label="Recall")
    plt.plot(x, f1s,        marker="^", label="F1")

    plt.xticks(x, methods)
    plt.xlabel("Clustering method")
    plt.ylabel("Score")
    plt.title(title)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"[INFO] Saved method comparison plot to: {out_path}")



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ratings_path", type=str, default="ml-100k/u.data")
    parser.add_argument("--users_path", type=str, default="ml-100k/u.user")

    parser.add_argument(
        "--cold_mode",
        type=str,
        default="pure",
        choices=["pure", "semi"],
    )

    parser.add_argument("--n_clusters", type=int, default=64)
    parser.add_argument("--min_support", type=float, default=0.15)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--phi", type=float, default=0.001)
    parser.add_argument("--ct", type=float, default=0.25)
    parser.add_argument("--max_itemset_len", type=int, default=None)

    parser.add_argument("--n_fold", type=int, default=10)
    parser.add_argument("--kfold_seed", type=int, default=10701)

    parser.add_argument("--p_delete", type=float, default=0.9)
    parser.add_argument("--p_seed", type=int, default=10701)

    parser.add_argument("--knn_k", type=int, default=20)
    parser.add_argument("--louvain_resolution", type=float, default=1.0)

    parser.add_argument(
        "--cluster_methods",
        nargs="+",
        default=["kmodes", "agg", "knn_louvain"],
        choices=["kmodes", "agg", "knn_louvain"],
        help="which clustering methods to compare",
    )

    args = parser.parse_args()

    print("========== Loading data ==========")
    UM, UP = load_movielens_100k(args.ratings_path, args.users_path)
    print(f"Loaded UM shape={UM.shape}, UP shape={UP.shape}")

    methods = args.cluster_methods
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    if args.cold_mode == "pure":
        print("========== PURE COLD-START COMPARISON ==========")
        for method in methods:
            mp, mr, mf, _, _ = run_pure_cold_for_method(method, UM, UP, args)
            precisions.append(mp)
            recalls.append(mr)
            f1s.append(mf)

        out_path = "cluster_method_comparison_pure.png"
        plot_method_comparison(
            methods,
            precisions,
            recalls,
            f1s,
            out_path,
            title=f"Pure cold-start (n_clusters={args.n_clusters})",
        )

    else:
        print(f"========== SEMI COLD-START COMPARISON (p_delete={args.p_delete:.2f}) ==========")
        for method in methods:
            mp, mr, mf, _, _ = run_semi_cold_for_method(method, UM, UP, args)
            precisions.append(mp)
            recalls.append(mr)
            f1s.append(mf)

        out_path = f"cluster_method_comparison_semi_p{args.p_delete:.2f}.png"
        plot_method_comparison(
            methods,
            precisions,
            recalls,
            f1s,
            out_path,
            title=f"Semi cold-start p={args.p_delete:.2f} (n_clusters={args.n_clusters})",
        )

    print("\n==================== SUMMARY TABLE ====================")
    print(f"cold_mode = {args.cold_mode}")
    header = "Method         Precision   Recall   F1"
    print(header)
    print("-" * len(header))
    for m, p, r, f in zip(methods, precisions, recalls, f1s):
        print(f"{m:<13} {p:9.4f} {r:8.4f} {f:8.4f}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
