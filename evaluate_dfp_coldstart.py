"""
Usage example:

python evaluate_dfp_coldstart.py \
  --ratings_path ml-100k/u.data \
  --users_path ml-100k/u.user \
  --cold_mode pure

python evaluate_dfp_coldstart.py \
  --ratings_path ml-100k/u.data \
  --users_path ml-100k/u.user \
  --cold_mode semi

"""

import argparse
import importlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


def build_pure_cold_input(UP_test_full: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Pure cold-start:
    The input UP_test_input contains only NaNs (the model has zero knowledge of the user's past interactions)
    """
    UP_test_input = pd.DataFrame(
        data=np.nan,
        index=UP_test_full.index,
        columns=UP_test_full.columns,
    )

    gt_pos: Dict[str, List[str]] = {}
    for u in UP_test_full.index:
        row = UP_test_full.loc[u]
        pos_items = list(row.index[row == 1])
        gt_pos[u] = pos_items

    return UP_test_input, gt_pos


def build_semi_cold_input(
    UP_test_full: pd.DataFrame,
    p: float = 0.9,
    random_state: int = 10701
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    In-matrix semi cold-start, according to paper:
    "a 10-fold cross-validation is performed for in-matrix prediction, i.e., for 
    each row of a subset, a randomly selected fraction p of cell values equal to 
    1 or 0 are deleted, thus representing the cold (new) users with a little rating 
    history for products. The deleted cells of each row of a subset (not the entire row) 
    form the test set and p varies from 30% to 90% of each row's number of cells. The 
    cells were removed by randomly selecting p/2 of cells of a row with value 1 and p/2 
    of cells of a row with value 0."
    """
    rng = np.random.default_rng(random_state)
    UP_test_input = UP_test_full.copy()
    gt_pos: Dict[str, List[str]] = {}

    for u in UP_test_full.index:
        row = UP_test_full.loc[u]

        pos_items = np.array(row.index[row == 1])
        neg_items = np.array(row.index[row == 0])

        n_pos = len(pos_items)
        n_neg = len(neg_items)

        if n_pos + n_neg == 0:
            gt_pos[u] = []
            continue

        # Delete p/2 pos and p/2 neg
        n_del_pos = int(np.floor((p / 2.0) * n_pos))
        n_del_neg = int(np.floor((p / 2.0) * n_neg))

        if n_del_pos > 0 and n_pos > 0:
            del_pos = rng.choice(pos_items, size=n_del_pos, replace=False)
        else:
            del_pos = np.array([], dtype=pos_items.dtype)

        if n_del_neg > 0 and n_neg > 0:
            del_neg = rng.choice(neg_items, size=n_del_neg, replace=False)
        else:
            del_neg = np.array([], dtype=neg_items.dtype)

        if len(del_pos) > 0:
            UP_test_input.loc[u, del_pos] = np.nan
        if len(del_neg) > 0:
            UP_test_input.loc[u, del_neg] = np.nan

        gt_pos[u] = list(del_pos)

    return UP_test_input, gt_pos


def eval_precision_recall_f1(
    UP_test_full: pd.DataFrame,
    UP_test_pred: pd.DataFrame,
    UP_test_input: pd.DataFrame,
) -> Tuple[float, float, float, int, int]:
    test_mask = UP_test_input.isna() & UP_test_full.notna()

    if not test_mask.values.any():
        return 0.0, 0.0, 0.0, 0, 0

    full_vals = UP_test_full.to_numpy()
    pred_vals = UP_test_pred.to_numpy()
    mask_flat = test_mask.to_numpy().astype(bool).ravel()

    y_true_flat = full_vals.ravel()[mask_flat]
    y_pred_flat = pred_vals.ravel()[mask_flat]

    y_true = (y_true_flat == 1).astype(int)

    y_pred_clean = np.nan_to_num(y_pred_flat, nan=0.0)
    y_pred = (y_pred_clean == 1).astype(int)

    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))
    P = int(np.sum(y_true == 1))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    num_test_cells = int(mask_flat.sum())
    num_positive_cells = P

    return float(precision), float(recall), float(f1), num_test_cells, num_positive_cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings_path", type=str, required=True)
    parser.add_argument("--users_path", type=str, required=True)

    # DFP model hyperparameters
    parser.add_argument("--min_support", type=float, default=0.15)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--phi", type=float, default=0.001)
    parser.add_argument("--ct", type=float, default=0.25)
    parser.add_argument("--max_itemset_len", type=int, default=None)

    parser.add_argument("--n_clusters", type=int, default=64)

    parser.add_argument("--n_fold", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=10701)
    parser.add_argument("--cold_mode", type=str, choices=["pure", "semi"], default="pure")

    # for semi cold-start
    parser.add_argument("--p_min", type=float, default=0.3)
    parser.add_argument("--p_max", type=float, default=0.9)
    parser.add_argument("--p_step", type=float, default=0.1)

    args = parser.parse_args()

    module = importlib.import_module("cold_start_dfp")
    DFPRecommender = module.DFPRecommender
    DFPConfig = module.DFPConfig
    load_movielens_100k = module.load_movielens_100k

    # load data
    UM, UP = load_movielens_100k(args.ratings_path, args.users_path)
    print(f"Loaded UM shape={UM.shape}, UP shape={UP.shape}")

    user_ids = np.array(UM.index)
    kf = KFold(n_splits=args.n_fold, shuffle=True, random_state=args.random_state)
    folds = list(kf.split(user_ids))

    # p (semi mode only)
    p_values = []
    p_prec_means = []
    p_rec_means = []
    p_f1_means = []

    # pure cold-start
    if args.cold_mode == "pure":
        print("\n==================== PURE COLD-START ====================")
        fold_precisions = []
        fold_recalls = []
        fold_f1s = []
        fold_cells = []
        fold_pos_cells = []

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            train_users = user_ids[train_idx]
            test_users = user_ids[test_idx]

            UM_train = UM.loc[train_users]
            UP_train = UP.loc[train_users]
            UM_test = UM.loc[test_users]
            UP_test_full = UP.loc[test_users]

            print(f"\n Fold {fold_idx + 1}/{args.n_fold}: "
                  f"Train users={len(train_users)}, Test users={len(test_users)}")

            cfg = DFPConfig(
                n_clusters=args.n_clusters,
                min_support=args.min_support,
                theta=args.theta,
                phi=args.phi,
                ct=args.ct,
                max_itemset_len=args.max_itemset_len
            )
            model = DFPRecommender(cfg)
            model.fit(UM_train, UP_train)

            UP_test_input, _ = build_pure_cold_input(UP_test_full)
            UP_test_pred, _ = model.transform(UM_test, UP_test_input)

            prec, rec, f1, num_cells, num_pos = eval_precision_recall_f1(
                UP_test_full, UP_test_pred, UP_test_input
            )
            print(f" Fold {fold_idx + 1} - "
                  f"Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, "
                  f"#cells={num_cells}, #pos={num_pos}")

            fold_precisions.append(prec)
            fold_recalls.append(rec)
            fold_f1s.append(f1)
            fold_cells.append(num_cells)
            fold_pos_cells.append(num_pos)

        mean_prec = float(np.mean(fold_precisions))
        std_prec = float(np.std(fold_precisions))
        mean_rec = float(np.mean(fold_recalls))
        std_rec = float(np.std(fold_recalls))
        mean_f1 = float(np.mean(fold_f1s))
        std_f1 = float(np.std(fold_f1s))
        total_cells = int(np.sum(fold_cells))
        total_pos_cells = int(np.sum(fold_pos_cells))

        print("\n==================== PURE COLD-START SUMMARY ====================")
        print(f"n_clusters        : {args.n_clusters}")
        print(f"Folds             : {args.n_fold}")
        print(f"Total test cells  : {total_cells}")
        print(f"Total positive cells: {total_pos_cells}")
        print("----------------------------------------------------")
        print(f"Precision (mean ± std): {mean_prec:.4f} ± {std_prec:.4f}")
        print(f"Recall    (mean ± std): {mean_rec:.4f} ± {std_rec:.4f}")
        print(f"F1        (mean ± std): {mean_f1:.4f} ± {std_f1:.4f}")
        print("====================================================\n")
        return

    # semi cold-start: for different p
    print("\n==================== SEMI COLD-START ====================")
    p_grid = np.arange(args.p_min, args.p_max + 1e-8, args.p_step)

    for p in p_grid:
        p = float(np.round(p, 2))

        print("\n===================================================================")
        print(f"Evaluating semi cold-start with delete_p = {p:.2f}")

        fold_precisions = []
        fold_recalls = []
        fold_f1s = []
        fold_cells = []
        fold_pos_cells = []

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            train_users = user_ids[train_idx]
            test_users = user_ids[test_idx]

            UM_train = UM.loc[train_users]
            UP_train = UP.loc[train_users]
            UM_test = UM.loc[test_users]
            UP_test_full = UP.loc[test_users]

            print(f"\n Fold {fold_idx + 1}/{args.n_fold}: "
                  f"Train users={len(train_users)}, Test users={len(test_users)}")

            cfg = DFPConfig(
                n_clusters=args.n_clusters,
                min_support=args.min_support,
                theta=args.theta,
                phi=args.phi,
                ct=args.ct,
                max_itemset_len=args.max_itemset_len
            )
            model = DFPRecommender(cfg)
            model.fit(UM_train, UP_train)

            UP_test_input, _ = build_semi_cold_input(
                UP_test_full,
                p=p,
                random_state=args.random_state
            )

            UP_test_pred, _ = model.transform(UM_test, UP_test_input)

            prec, rec, f1, num_cells, num_pos = eval_precision_recall_f1(
                UP_test_full, UP_test_pred, UP_test_input
            )
            print(f" Fold {fold_idx + 1} - "
                  f"Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, "
                  f"#cells={num_cells}, #pos={num_pos}")

            fold_precisions.append(prec)
            fold_recalls.append(rec)
            fold_f1s.append(f1)
            fold_cells.append(num_cells)
            fold_pos_cells.append(num_pos)

        mean_prec = float(np.mean(fold_precisions))
        std_prec = float(np.std(fold_precisions))
        mean_rec = float(np.mean(fold_recalls))
        std_rec = float(np.std(fold_recalls))
        mean_f1 = float(np.mean(fold_f1s))
        std_f1 = float(np.std(fold_f1s))
        total_cells = int(np.sum(fold_cells))
        total_pos_cells = int(np.sum(fold_pos_cells))

        print("\n-------------------- CV SUMMARY for p = "
              f"{p:.2f} --------------------")
        print(f"delete_p          : {p:.2f}")
        print(f"n_clusters        : {args.n_clusters}")
        print(f"Folds             : {args.n_fold}")
        print(f"Total test cells  : {total_cells}")
        print(f"Total positive cells: {total_pos_cells}")
        print("----------------------------------------------------")
        print(f"Precision (mean ± std): {mean_prec:.4f} ± {std_prec:.4f}")
        print(f"Recall    (mean ± std): {mean_rec:.4f} ± {std_rec:.4f}")
        print(f"F1        (mean ± std): {mean_f1:.4f} ± {std_f1:.4f}")
        print("-----------------------------------------------------------------\n")

        p_values.append(p)
        p_prec_means.append(mean_prec)
        p_rec_means.append(mean_rec)
        p_f1_means.append(mean_f1)

    # Precision & Recall vs p
    if p_values:
        plt.figure()
        plt.plot(p_values, p_prec_means, marker="o", label="Precision")
        plt.plot(p_values, p_rec_means, marker="s", label="Recall")
        plt.xlabel("delete fraction p")
        plt.ylabel("Score")
        plt.title(f"Semi Cold-Start: Precision & Recall vs p (n_clusters={args.n_clusters})")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        out_path = "precision_recall_vs_p.png"
        plt.savefig(out_path, dpi=300)
        print(f"\n[INFO] Saved Precision & Recall vs p plot to: {out_path}")

    print("\n==================== SEMI COLD-START p SUMMARY ====================")
    print("p_delete | Precision_mean | Recall_mean | F1_mean")
    print("--------------------------------------------------")
    for p, mp, mr, mf in zip(p_values, p_prec_means, p_rec_means, p_f1_means):
        print(f"{p:7.2f} | {mp:.4f}        | {mr:.4f}      | {mf:.4f}")
    print("===========================================================================")


if __name__ == "__main__":
    main()