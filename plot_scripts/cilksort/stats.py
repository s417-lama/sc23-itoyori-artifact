import os
import sys
import pathlib
import itertools
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

os.chdir(pathlib.Path(__file__).parent.parent.parent)
sys.path.append("./plot_scripts")
import plot_util

benchmark = "cilksort"

# machine = "wisteria-o"
machine = "local"

artifacts_dir = "ityrbench_artifacts"

fig_dir = os.path.join("figs", benchmark)

event_looks = dict(
    others=dict(
        rank=9,
        pattern="",
        solidity=0.5,
        color="#444444",
        title="Others",
    ),
    get=dict(
        rank=8,
        pattern="\\",
        solidity=0.5,
        color=plot_util.tol_cset("light").light_blue,
        title="Get",
    ),
    # put=dict(
    #     rank=7,
    #     pattern="/",
    #     solidity=0.5,
    #     color=plot_util.tol_cset("light").orange,
    #     title="Put",
    # ),
    checkout=dict(
        rank=6,
        pattern="|",
        solidity=0.5,
        color=plot_util.tol_cset("light").light_cyan,
        title="Checkout",
    ),
    checkin=dict(
        rank=5,
        pattern="-",
        solidity=0.5,
        color=plot_util.tol_cset("light").pink,
        title="Checkin",
    ),
    release=dict(
        rank=4,
        pattern=".",
        solidity=0.4,
        color=plot_util.tol_cset("light").pear,
        title="Release",
    ),
    release_lazy=dict(
        rank=3,
        pattern=".",
        solidity=0.7,
        color=plot_util.tol_cset("light").olive,
        title="Lazy Release",
    ),
    acquire=dict(
        rank=2,
        pattern="+",
        solidity=0.6,
        color=plot_util.tol_cset("light").mint,
        title="Acquire",
    ),
    merge_kernel=dict(
        rank=1,
        pattern="x",
        solidity=0.7,
        color="#BBBBBB",
        title="Serial Merge",
    ),
    quicksort_kernel=dict(
        rank=0,
        pattern="",
        solidity=0.5,
        color="#BBBBBB",
        title="Serial Quicksort",
    ),
)

pio.templates.default = "plotly_white"

n_warmups = 1

def get_result(batch_name, filename_template, **args):
    files = [(os.path.join(artifacts_dir, machine, benchmark, batch_name, filename_template.format(**dict(p))), dict(p))
             for p in itertools.product(*[[(k, v) for v in args[k]] for k in args])]
    df = plot_util.txt2df(files, [
            r'# of processes: *(?P<nproc>\d+)',
            r'\[(?P<i>\d+)\] *(?P<time>\d+) *ns',
        ], [
            r'^ *(?P<event>[a-zA-Z_]+) .*\( *(?P<acc>\d+) *ns */.*\)',
        ])
    df = df.loc[df["i"] >= n_warmups].copy()
    return df

def get_parallel_result_1G():
    if machine == "wisteria-o":
        nodes = [1, "2:torus", "2x3:torus", "2x3x2:torus", "3x4x3:torus"]
        duplicates = [0, 1, 2]
    elif machine == "local":
        nodes = [1, 2, 4]
        duplicates = [0]
    return get_result("scale1G", "nodes_{nodes}_p_{policy}_{duplicate}.out",
                      n_input=["1G elements"],
                      nodes=nodes,
                      policy=["nocache", "writeback_lazy"],
                      duplicate=duplicates)

def get_parallel_result_10G():
    if machine == "wisteria-o":
        nodes = ["2x3:torus", "2x3x2:torus", "3x4x3:torus"]
        duplicates = [0, 1, 2]
    else:
        return pd.DataFrame()
    return get_result("scale10G", "nodes_{nodes}_p_{policy}_{duplicate}.out",
                      n_input=["10G elements"],
                      nodes=nodes,
                      policy=["nocache", "writeback_lazy"],
                      duplicate=duplicates)

if __name__ == "__main__":
    fig = go.Figure()

    df = pd.concat([get_parallel_result_1G(), get_parallel_result_10G()])
    df = df[df["policy"] == "writeback_lazy"]
    df = df[df["event"].isin(event_looks.keys())]
    df = df.assign(totaltime=df["time"] * df["nproc"])
    df = df.groupby(["nproc", "n_input", "event"]).agg({"acc": "mean", "totaltime": "max"})
    df = df.reset_index()

    for (nproc, n_input), df_nproc in df.groupby(["nproc", "n_input"]):
        df = pd.concat([df, pd.DataFrame.from_records([
            {"nproc": nproc, "n_input": n_input,
             "event": "others", "totaltime": df_nproc["totaltime"].max(),
             "acc": df_nproc["totaltime"].max() - df_nproc["acc"].sum()}])])

    print(df)

    df = df.groupby("n_input").apply(lambda df_n: df_n.assign(acc=df_n["acc"] / df_n["totaltime"].max()))
    df = df.droplevel(0).reset_index()

    print(df)

    for event, looks in reversed(event_looks.items()):
        df_event = df[df["event"] == event]
        fig.add_trace(go.Bar(
            x=[df_event["n_input"], df_event["nproc"]],
            y=df_event["acc"],
            legendrank=looks["rank"],
            marker_color=looks["color"],
            marker_line_color="#333333",
            marker_line_width=1.5,
            marker_pattern_fillmode="replace",
            marker_pattern_solidity=looks["solidity"],
            marker_pattern_shape=looks["pattern"],
            marker_pattern_size=8,
            name=looks["title"],
        ))

    fig.update_xaxes(
        showline=True,
        linecolor="black",
        ticks="outside",
        mirror=True,
        exponentformat="SI",
        title_standoff=10,
        title_text="# of cores / Input size",
    )
    fig.update_yaxes(
        range=[0, 1],
        showline=True,
        linecolor="black",
        ticks="outside",
        mirror=True,
        showgrid=True,
    )
    fig.update_layout(
        width=550,
        height=290,
        margin=dict(l=0, r=0, b=75, t=0),
        barmode="stack",
        legend_font_size=15,
        font=dict(
            family="Linux Biolinum O, sans-serif",
            size=16,
        ),
    )

    plot_util.save_fig(fig, fig_dir, "stats_{}.html".format(machine))
