import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd

from kmodes.kmodes import KModes
from mlxtend.frequent_patterns import apriori

def binarize_ratings(df: pd.DataFrame, user_col="userId", item_col="itemId", rating_col="rating", pos_threshold: float = 2.0) -> pd.DataFrame:
    pos = df[df[rating_col] > pos_threshold][[user_col, item_col]].copy()
    pos["val"] = 1
    neg = df[df[rating_col] <= pos_threshold][[user_col, item_col]].copy()
    neg["val"] = 0
    bin_df = pd.concat([pos, neg], ignore_index=True)
    bin_df = bin_df.drop_duplicates(subset=[user_col, item_col], keep="last")
    return bin_df


def pivot_user_item(bin_df: pd.DataFrame, user_col="userId", item_col="itemId") -> pd.DataFrame:
    ui = bin_df.pivot_table(index=user_col, columns=item_col, values="val", aggfunc="max")
    ui = ui.sort_index(axis=0).sort_index(axis=1)
    return ui


def ensure_categorical(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(str)


@dataclass
class DFPConfig:
    n_clusters: int = 8               
    init: str = "Huang"
    n_init: int = 5
    max_iter: int = 100
    min_support: float = 0.15         
    theta: float = 1.0                
    phi: float = 0.001                
    ct: float = 0.25                  
    max_itemset_len: Optional[int] = None  


class DFPRecommender:
    def __init__(self, cfg: DFPConfig):
        self.cfg = cfg
        self.km: Optional[KModes] = None
        self.cluster_labels_: Optional[pd.Series] = None
        self.cluster_itemsets_: Dict[int, Dict[frozenset, float]] = {}  
        self.discriminant_itemsets_: Dict[int, List[Set]] = {}          
        self.items_: List = []
    

    def fit(self, UM: pd.DataFrame, UP: pd.DataFrame):
        cfg = self.cfg
        UMc = ensure_categorical(UM.copy())
        self.items_ = list(UP.columns)
        km = KModes(n_clusters=cfg.n_clusters, init=cfg.init, n_init=cfg.n_init, max_iter=cfg.max_iter, verbose=0)
        labels = km.fit_predict(UMc.values)
        self.km = km
        self.cluster_labels_ = pd.Series(labels, index=UMc.index, name="cluster")
        cluster_itemsets = {}
        for c in range(cfg.n_clusters):
            users_c = self.cluster_labels_[self.cluster_labels_ == c].index
            if len(users_c) == 0:
                cluster_itemsets[c] = {}
                continue
            sub = UP.loc[users_c, self.items_]
            trans = (sub.fillna(0) == 1)
            trans = trans.loc[:, trans.any(axis=0)]
            trans = trans.astype(bool)
            if trans.shape[1] == 0:
                cluster_itemsets[c] = {}
                continue
            fis = apriori(
                trans,
                min_support=cfg.min_support,
                use_colnames=True,
                max_len=(cfg.max_itemset_len or 2),  
                low_memory=True
            )
            itemsets = {frozenset(row['itemsets']): float(row['support']) for _, row in fis.iterrows()}
            cluster_itemsets[c] = itemsets
        self.cluster_itemsets_ = cluster_itemsets
        self.discriminant_itemsets_ = self._compute_discriminant_itemsets()


    def _compute_discriminant_itemsets(self) -> Dict[int, List[Set]]:
        cfg = self.cfg
        n = cfg.n_clusters
        universe: Set[frozenset] = set()
        for c in range(n):
            universe |= set(self.cluster_itemsets_.get(c, {}).keys())
        rf_map: Dict[frozenset, List[float]] = {}
        for s in universe:
            rfs = []
            for c in range(n):
                rfs.append(self.cluster_itemsets_.get(c, {}).get(s, 0.0))
            rf_map[s] = rfs
        disc: Dict[int, List[Set]] = {c: [] for c in range(n)}
        for s, rfs in rf_map.items():
            denom = sum(rfs) + 1e-12  
            for c in range(n):
                rf_c = rfs[c]
                cond2 = (n * rf_c / denom) >= cfg.theta          
                cond3 = rf_c >= (cfg.phi * cfg.theta)            
                if cond2 and cond3:
                    disc[c].append(set(s))
        return disc
    

    def transform(self, UM: pd.DataFrame, UP: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        cfg = self.cfg
        UMc = ensure_categorical(UM.copy())
        labels = self.km.predict(UMc.values)
        user2cluster = pd.Series(labels, index=UMc.index)
        UP_new = UP.copy()
        recommendations: Dict = {}
        for u in UP.index:
            if u not in user2cluster.index:
                continue
            c = int(user2cluster.loc[u])
            discs = self.discriminant_itemsets_.get(c, [])
            row = UP_new.loc[u, self.items_]
            is_cold = row.isna().all()
            rec_items = set()
            if is_cold:
                for S in discs:
                    rec_items |= S
                for it in rec_items:
                    if it in row.index:
                        row.at[it] = 1
            else:  
                explicit_zero = set(row.index[(row == 0).values])
                for S in discs:
                    conflicts = len(explicit_zero.intersection(S))
                    if conflicts < (len(S) * cfg.ct):  
                        for it in S:
                            if pd.isna(row.at[it]):   
                                row.at[it] = 1
                                rec_items.add(it)
            UP_new.loc[u, self.items_] = row
            recommendations[u] = sorted(list(rec_items))
        return UP_new, {"recommendations": recommendations}


    def save_model(self, path: str):
        obj = {
            "cfg": self.cfg.__dict__,
            "cluster_centroids_": self.km.cluster_centroids_.tolist(),
        }
        with open(path, "w") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)


def load_movielens_100k(ratings_path: str, users_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(ratings_path, sep="\t", header=None, names=["userId", "itemId", "rating", "ts"])
    bin_df = binarize_ratings(ratings, pos_threshold=2.0)
    users = pd.read_csv(users_path, sep="|", header=None, names=["userId", "age", "gender", "occupation", "zip"])
    users["userId"] = users["userId"].astype(int).astype(str)
    users["age_bin"] = pd.cut(users["age"], bins=[0, 18, 25, 35, 45, 55, 200], right=False).astype(str)
    users["zip3"] = users["zip"].astype(str).str[:3]
    UM = users.set_index("userId")[["age_bin", "gender", "occupation", "zip3"]].astype(str)
    bin_df["userId"] = bin_df["userId"].astype(int).astype(str)
    bin_df["itemId"] = bin_df["itemId"].astype(int).astype(str)
    UP = pivot_user_item(bin_df, user_col="userId", item_col="itemId")
    commons = sorted(list(set(UM.index) & set(UP.index)))
    UM = UM.loc[commons]
    UP = UP.loc[commons]
    return UM, UP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings_path", type=str)
    parser.add_argument("--users_path", type=str)
    parser.add_argument("--n_clusters", type=int, default=8)
    parser.add_argument("--min_support", type=float, default=0.15)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--phi", type=float, default=0.001)
    parser.add_argument("--ct", type=float, default=0.25)
    parser.add_argument("--max_itemset_len", type=int, default=None)
    parser.add_argument("--out_prefix", type=str, default="dfp_out")
    args = parser.parse_args()

    cfg = DFPConfig(
        n_clusters=args.n_clusters,
        min_support=args.min_support,
        theta=args.theta,
        phi=args.phi,
        ct=args.ct,
        max_itemset_len=args.max_itemset_len
    )
    model = DFPRecommender(cfg)
    UM, UP = load_movielens_100k(args.ratings_path, args.users_path)
    model.fit(UM, UP)
    UP_new, info = model.transform(UM, UP)
    UP_new.to_parquet(f"{args.out_prefix}.parquet")
    with open(f"{args.out_prefix}_recommendations.json", "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"Save matrix to {args.out_prefix}.parquet")
    print(f"Save recommendations to {args.out_prefix}_recommendations.json")
    print(f"User number ={len(UP_new)}, item number={UP_new.shape[1]}")


if __name__ == "__main__":
    main()
