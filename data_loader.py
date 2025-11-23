"""
Module for loading the data from the movielens100k dataset
"""

from typing import Tuple
import pandas as pd


def binarize_ratings(df: pd.DataFrame, pos_threshold: float) -> pd.DataFrame:
    """
    Binarises the ratings column of a movie from a scale to simply 1/0,
    depending on if its greater than pos_threshold
    """
    pos = df[df["rating"] > pos_threshold][["userId", "itemId"]].copy()
    pos["val"] = 1
    neg = df[df["rating"] <= pos_threshold][["userId", "itemId"]].copy()
    neg["val"] = 0
    bin_df = pd.concat([pos, neg], ignore_index=True)
    bin_df = bin_df.drop_duplicates(subset=["userId", "itemId"], keep="last")
    return bin_df


def pivot_user_item(bin_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the table of [userId, itemId, val] tuples into a table where rows
    correspond to users, columns correspond to items, and each cell is the 
    rating (0 / 1 / NaN) the user gave to that movie
    """
    ui = bin_df.pivot_table(index="userId", columns="itemId", values="val", aggfunc="max")
    ui = ui.sort_index(axis=0).sort_index(axis=1)
    return ui


def load_movielens_100k(ratings_path: str, users_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the ratings and users data from the movielens
    """
    ratings = pd.read_csv(ratings_path, sep="\t", header=None, names=["userId", "itemId", "rating", "ts"])
    bin_df = binarize_ratings(ratings, 2.0)
    bin_df["userId"] = bin_df["userId"].astype(int).astype(str)
    bin_df["itemId"] = bin_df["itemId"].astype(int).astype(str)
    UP = pivot_user_item(bin_df)

    users = pd.read_csv(users_path, sep="|", header=None, names=["userId", "age", "gender", "occupation", "zip"])
    users["userId"] = users["userId"].astype(int).astype(str)
    users["age_bin"] = pd.cut(users["age"], bins=[0, 18, 25, 35, 45, 55, 200], right=False).astype(str)
    users["zip3"] = users["zip"].astype(str).str[:3]
    UM = users.set_index("userId")[["age_bin", "gender", "occupation", "zip3"]].astype(str)
    
    commons = sorted(list(set(UM.index) & set(UP.index)))
    UM = UM.loc[commons]
    UP = UP.loc[commons]
    return UM, UP
