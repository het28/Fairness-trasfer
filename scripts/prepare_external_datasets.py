"""ANONYMOUS ARTIFACT NOTE
This runner is included for optional FULL reproduction.
Repository-relative paths may still reference historical layout strings in constants.
For reviewer verification of reported numbers, use scripts/verify_artifact.sh instead of this file.
Scientific defaults (grids, seeds, formulas) must not be changed.
"""
#!/usr/bin/env python3
"""Build ECIR RecBole atomic datasets from LightGCN Amazon-Book / Gowalla dumps.

Protocol:
  - Use LightGCN-PyTorch filtered bipartite graphs (standard 10-core literature sizes).
  - Merge train.txt + test.txt interactions (ignore their leave-one-out split).
  - Write RecBole .inter; RecBole RS 8:1:1 (seeded) used at train time — same as ML-1M/LastFM.
  - Implicit feedback: rating=1.0; c_ui=1 (multiply_c_ui may be True but ones).
  - Popularity groups / T built from TRAIN split only inside MEG-RW injection (no test leakage).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DL = ROOT / "dataset" / "_downloads"
OUT_ROOT = ROOT / "dataset"


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[x >= 0]
    if x.size == 0 or x.sum() <= 0:
        return float("nan")
    x = np.sort(x)
    n = x.size
    i = np.arange(1, n + 1)
    return float((2.0 * np.sum(i * x) / (n * x.sum())) - (n + 1) / n)


def load_lightgcn_edges(train_path: Path, test_path: Path) -> pd.DataFrame:
    edges = []
    for path in (train_path, test_path):
        with path.open() as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                u = parts[0]
                for it in parts[1:]:
                    edges.append((u, it))
    df = pd.DataFrame(edges, columns=["user_id", "item_id"]).drop_duplicates()
    return df


def write_recbole(df: pd.DataFrame, name: str) -> Path:
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    inter = out / f"{name}.inter"
    # RecBole atomic header
    with inter.open("w") as f:
        f.write("user_id:token\titem_id:token\trating:float\n")
        for u, i in df[["user_id", "item_id"]].itertuples(index=False):
            f.write(f"{u}\t{i}\t1.0\n")
    return inter


def stats(df: pd.DataFrame, name: str) -> dict:
    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()
    n_inter = len(df)
    dens = n_inter / (n_users * n_items)
    udeg = df.groupby("user_id").size().to_numpy()
    ideg = df.groupby("item_id").size().to_numpy()
    # head/tail by item degree quartiles approx via fracs 0.1/0.2/0.3/0.4 catalog
    order = np.argsort(-ideg)
    labels = np.empty(n_items, dtype=object)
    # map item id -> degree rank groups by catalog share of items
    items = df["item_id"].drop_duplicates().to_numpy()
    # recompute degrees aligned
    ideg_s = df.groupby("item_id").size()
    items_sorted = ideg_s.sort_values(ascending=False)
    n = len(items_sorted)
    cuts = [0, int(0.1 * n), int(0.3 * n), int(0.6 * n), n]
    names = ["Head", "UpperMid", "LowerMid", "Tail"]
    item_group = {}
    for g, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        for it in items_sorted.index[a:b]:
            item_group[it] = names[g]
    df = df.copy()
    df["g"] = df["item_id"].map(item_group)
    mass = df["g"].value_counts(normalize=True)
    # connected components on bipartite via simple BFS on user-item
    from collections import defaultdict, deque
    adj_u = defaultdict(set)
    adj_i = defaultdict(set)
    for u, i in df[["user_id", "item_id"]].itertuples(index=False):
        adj_u[u].add(i)
        adj_i[i].add(u)
    seen_u, seen_i = set(), set()
    comps = 0
    for u0 in adj_u:
        if u0 in seen_u:
            continue
        comps += 1
        dq = deque([("u", u0)])
        seen_u.add(u0)
        while dq:
            typ, x = dq.popleft()
            if typ == "u":
                for ii in adj_u[x]:
                    if ii not in seen_i:
                        seen_i.add(ii)
                        dq.append(("i", ii))
            else:
                for uu in adj_i[x]:
                    if uu not in seen_u:
                        seen_u.add(uu)
                        dq.append(("u", uu))
    return {
        "name": name,
        "n_users": int(n_users),
        "n_items": int(n_items),
        "n_interactions": int(n_inter),
        "density": float(dens),
        "user_degree_mean": float(udeg.mean()),
        "item_degree_mean": float(ideg.mean()),
        "user_degree_gini": gini(udeg),
        "item_degree_gini": gini(ideg),
        "head_interaction_share": float(mass.get("Head", 0.0)),
        "tail_interaction_share": float(mass.get("Tail", 0.0)),
        "group_mass": {k: float(mass.get(k, 0.0)) for k in names},
        "bipartite_connected_components": int(comps),
        "sparsity": float(1.0 - dens),
    }


def main():
    specs = {
        "amazon-books-ecir": (
            DL / "amazon-book_train.txt",
            DL / "amazon-book_test.txt",
            {
                "source": "LightGCN-PyTorch data/amazon-book (gusye1234)",
                "url_train": "https://github.com/gusye1234/LightGCN-PyTorch/tree/master/data/amazon-book",
                "citation": "He et al., LightGCN, SIGIR 2020 (Amazon-Book preprocessing lineage)",
                "filtering": "As released by LightGCN-PyTorch (standard 10-core Amazon-Book)",
                "split_at_train_time": "RecBole RS 8:1:1 random by user (not LightGCN leave-one-out)",
                "c_ui": "implicit ones (rating=1.0)",
                "deduplication": "drop duplicate (user,item) after merge",
            },
        ),
        "gowalla-ecir": (
            DL / "gowalla_train.txt",
            DL / "gowalla_test.txt",
            {
                "source": "LightGCN-PyTorch data/gowalla (gusye1234)",
                "url_train": "https://github.com/gusye1234/LightGCN-PyTorch/tree/master/data/gowalla",
                "citation": "Cho et al. Gowalla; LightGCN SIGIR 2020 preprocessing lineage",
                "filtering": "As released by LightGCN-PyTorch (standard Gowalla CF filter)",
                "split_at_train_time": "RecBole RS 8:1:1 random by user",
                "c_ui": "implicit ones (rating=1.0)",
                "deduplication": "drop duplicate (user,item) after merge",
            },
        ),
    }
    all_stats = {}
    meta = {}
    for name, (tr, te, info) in specs.items():
        print("building", name)
        df = load_lightgcn_edges(tr, te)
        write_recbole(df, name)
        st = stats(df, name)
        all_stats[name] = st
        meta[name] = {**info, **st}
        print(json.dumps(st, indent=2))
    out = ROOT / "research_ecir2027" / "results" / "ecir2027" / "external_transfer"
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_stats.json").write_text(json.dumps(meta, indent=2))
    print("wrote", out / "dataset_stats.json")


if __name__ == "__main__":
    main()
