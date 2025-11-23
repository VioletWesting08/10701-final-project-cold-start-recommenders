"""
Module for running the cold start algorithm from the paper "Addressing the 
Cold-Start Problem in Recommender Systems Based on Frequent Patterns"

Usage example:

python cold_start_dfp.py \
  --ratings_path ./ml-100k/u.data \
  --users_path   ./ml-100k/u.user \
  --n_clusters 8 \
  --min_support 0.15 \
  --theta 1.0 \
  --phi 0.001 \
  --ct 0.25 \
  --out_prefix movielens100k
"""

import argparse
import json
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
from kmodes.kmodes import KModes
from mlxtend.frequent_patterns import apriori

from data_loader import load_movielens_100k


@dataclass
class DFPConfig:
    # The first 4 configurations are for the KNN clustering used to group users together
    n_clusters: int = 8  # How many clusters for KNN
    init: str = "Huang"  # How should the clusters be initialised
    n_init: int = 5  # The number of restarts
    max_iter: int = 100  # The number of KNN iterations

    # The next 2 configurations are used to determine what counts as a frequent itemset
    min_support: float = 0.15  # It is frequent if it shows up in this portion of items
    max_itemset_len: Optional[int] = None  # The size of an itemset must be at most this

    # The next 2 configurations are used in the discriminant itemset calculation
    theta: float = 1.0
    phi: float = 0.001

    # The next configuration is used to tune how much a recommendation can conflict with prior preferences
    ct: float = 0.25


class DFPRecommender:
    def __init__(self, cfg: DFPConfig):
        """
        Initialise the model
        """
        self.cfg = cfg
        self.km: Optional[KModes] = None
        self.cluster_labels_: Optional[pd.Series] = None
        self.cluster_itemsets_: Dict[int, Dict[frozenset, float]] = {}  
        self.discriminant_itemsets_: Dict[int, List[Set]] = {}          
        self.items_: List = []
    

    def fit(self, UM: pd.DataFrame, UP: pd.DataFrame):
        """
        Fits the model 
        """
        # Step 1 of the algorithm is to cluster users
        cfg = self.cfg
        UMc = UM.copy().astype(str)
        self.items_ = list(UP.columns)
        km = KModes(n_clusters=cfg.n_clusters, init=cfg.init, n_init=cfg.n_init, max_iter=cfg.max_iter, verbose=0)
        labels = km.fit_predict(UMc.values)
        self.km = km
        self.cluster_labels_ = pd.Series(labels, index=UMc.index, name="cluster")

        # Step 2 of the algorithm is to extract their frequent itemsets
        cluster_itemsets = {}
        for c in tqdm(range(cfg.n_clusters), leave=False, desc="Fitting Model"):
            # Prepare itemset
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
            # Run the apriori algorithm to find frequent itemsets in this cluster of users
            fis = apriori(
                trans,
                min_support=cfg.min_support,
                use_colnames=True,
                max_len=(cfg.max_itemset_len or 2),  
                low_memory=True
            )
            itemsets = {frozenset(row['itemsets']): float(row['support']) for _, row in fis.iterrows()}
            cluster_itemsets[c] = itemsets

        # Step 3 of the algorithm is to then extract discriminant frequent itemsets
        self.cluster_itemsets_ = cluster_itemsets
        self.discriminant_itemsets_ = self._compute_discriminant_itemsets()


    def _compute_discriminant_itemsets(self) -> Dict[int, List[Set]]:
        """
        Computes discriminant itemsets given itemsets per cluster
        """
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
        """
        Predicts based on the model
        """
        # Start by assinging users to clusters
        cfg = self.cfg
        UMc = UM.copy().astype(str)
        labels = self.km.predict(UMc.values)
        user2cluster = pd.Series(labels, index=UMc.index)
        UP_new = UP.copy()
        recommendations: Dict = {}

        # For every user, we generate new predictions
        for u in UP.index:
            if u not in user2cluster.index:
                continue
            c = int(user2cluster.loc[u])
            discs = self.discriminant_itemsets_.get(c, [])
            row = UP_new.loc[u, self.items_]
            is_cold = row.isna().all()
            rec_items = set()
            if is_cold:
                # Cold users simply get recommended products from their cluster
                for S in discs:
                    rec_items |= S
                for it in rec_items:
                    if it in row.index:
                        row.at[it] = 1
            else:
                # Non-cold users also take into account conflicting preferences
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
        """
        Saves the trained model to disk
        """
        obj = {
            "cfg": self.cfg.__dict__,
            "cluster_centroids_": self.km.cluster_centroids_.tolist(),
        }
        with open(path, "w") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)


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

    UM, UP = load_movielens_100k(args.ratings_path, args.users_path)

    cfg = DFPConfig(
        n_clusters=args.n_clusters,
        min_support=args.min_support,
        theta=args.theta,
        phi=args.phi,
        ct=args.ct,
        max_itemset_len=args.max_itemset_len
    )
    model = DFPRecommender(cfg)

    model.fit(UM, UP)
    UP_new, info = model.transform(UM, UP)

    UP_new.to_parquet(f"{args.out_prefix}.parquet")
    with open(f"{args.out_prefix}_recommendations.json", "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"Save matrix to {args.out_prefix}.parquet")
    print(f"Save recommendations to {args.out_prefix}_recommendations.json")
    print(f"User number = {len(UP_new)}, item number = {UP_new.shape[1]}")


if __name__ == "__main__":
    main()
