import os
import sys
import pathlib
import itertools
import re
import math
import numpy as np
import pandas as pd

os.chdir(pathlib.Path(__file__).parent.parent.parent)
sys.path.append("./plot_scripts")
import plot_util

# machine = "wisteria-o"
machine = "local"

artifacts_dir = "ityrbench_artifacts"

n_warmups = 1

def get_mpi_balance(benchmark, batch_name, filename_template, **args):
    files = [(os.path.join(artifacts_dir, machine, benchmark, batch_name, filename_template.format(**dict(p))), dict(p))
             for p in itertools.product(*[[(k, v) for v in args[k]] for k in args])]
    total_regex = re.compile("Traverse \(total\) *: *(?P<time>\d+\.\d+) *s")
    rank_regex = re.compile("Traverse \(rank (?P<rank>\d+)\) *: *(?P<time>\d+\.\d+) *s")
    rows = []
    for file in files:
        filename, ctx = file
        total_times = []
        rank_dict = {}
        with open(filename, "r") as f:
            for line in f:
                m = total_regex.search(line)
                if m:
                    total_times.append(m["time"])
                m = rank_regex.search(line)
                if m:
                    if m["rank"] in rank_dict:
                        rank_dict[m["rank"]].append(m["time"])
                    else:
                        rank_dict[m["rank"]] = [m["time"]]
        for rank, times in rank_dict.items():
            for i, t in enumerate(times):
                rows.append(dict(ctx, rank=rank, time=t, i=i, total_time=total_times[i]))
    df = plot_util.infer_dtypes(pd.DataFrame(rows))
    df = df.loc[df["i"] >= n_warmups].copy()
    df["busy"] = df["time"] / df["total_time"]
    return df

def get_mpi_balance_1M():
    if machine == "wisteria-o":
        nodes = [1, "2:torus", "2x3:torus", "2x3x2:torus", "3x4x3:torus", "6x6x4:torus"]
        duplicates = [0, 1, 2]
    elif machine == "local":
        nodes = [1, 2, 4]
        duplicates = [0]
    return get_mpi_balance("exafmm_mpi", "scale1M", "nodes_{nodes}_{duplicate}.out",
                           n_input=[1_000_000],
                           nodes=nodes,
                           duplicate=duplicates)

def get_mpi_balance_10M():
    if machine == "wisteria-o":
        nodes = [1, "2:torus", "2x3:torus", "2x3x2:torus", "3x4x3:torus", "6x6x4:torus"]
        duplicates = [0, 1, 2]
    elif machine == "local":
        return pd.DataFrame()
    return get_mpi_balance("exafmm_mpi", "scale10M", "nodes_{nodes}_{duplicate}.out",
                           n_input=[10_000_000],
                           nodes=nodes,
                           duplicate=duplicates)

if __name__ == "__main__":
    df = pd.concat([get_mpi_balance_1M(), get_mpi_balance_10M()])
    for n_input, df_n in df.groupby("n_input"):
        for nodes, df_nodes in df_n.groupby("nodes"):
            print(n_input, nodes)
            df_nodes = df_nodes.groupby("i").agg({"busy": ["mean", "median", "min", plot_util.ci_lower, plot_util.ci_upper]})
            print(df_nodes)
            print(df_nodes[("busy", "mean")].mean())
