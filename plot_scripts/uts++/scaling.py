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

benchmark = "uts++"

# machine = "wisteria-o"
machine = "local"

plot_serial = True
# plot_serial = False

artifacts_dir = "ityrbench_artifacts"

fig_dir = os.path.join("figs", benchmark)

policy_looks = dict(
    nocache=dict(
        rank=3,
        color="#EE6677",
        dash="dot",
        title="No Cache",
        marker="circle-open",
    ),
    writethrough=dict(
        rank=2,
        color="#4477AA",
        dash="dash",
        title="Write-Through Policy",
        marker="diamond-open",
    ),
    writeback=dict(
        rank=1,
        color="#CCBB44",
        dash="dashdot",
        title="Write-Back Policy",
        marker="square-open",
    ),
    writeback_lazy=dict(
        rank=0,
        color="#228833",
        dash="solid",
        # title="Write-Back + Lazy Flush",
        title="Cache (Write-Back, Lazy)",
        marker="star-triangle-up-open",
    ),
)

pio.templates.default = "plotly_white"

linewidth = 2
itemwidth = 70
markersize = 12

n_warmups = 1

def get_result(batch_name, filename_template, **args):
    files = [(os.path.join(artifacts_dir, machine, benchmark, batch_name, filename_template.format(**dict(p))), dict(p))
             for p in itertools.product(*[[(k, v) for v in args[k]] for k in args])]
    df = plot_util.txt2df(files, [
            r'# of processes: *(?P<nproc>\d+)',
        ], [
            r'\[(?P<i>\d+)\] *(?P<time>\d+) *ns',
        ])
    df = df.loc[df["i"] >= n_warmups].copy()
    return df

def get_serial_result():
    policy="serial"
    # policy="writeback_lazy"
    return get_result("serial", "tree_{tree}_p_{policy}_{duplicate}.out",
                      tree=["T1L"],
                      policy=[policy],
                      duplicate=[0])

def get_parallel_result_T1L():
    if machine == "wisteria-o":
        nodes = [1, "2:torus", "2x3:torus", "2x3x2:torus", "3x4x3:torus"]
        # nodes = [1, "2:torus", "2x3:torus", "2x3x2:torus", "3x4x3:torus", "6x6x4:torus"]
        duplicates = [0, 1, 2]
    elif machine == "local":
        nodes = [1, 2, 4]
        duplicates = [0]
    else:
        raise Exception
    # policies = ["nocache", "writethrough", "writeback", "writeback_lazy"]
    policies = ["nocache", "writeback_lazy"]
    return get_result("T1L", "nodes_{nodes}_p_{policy}_{duplicate}.out",
                      tree=["T1L"],
                      nodes=nodes,
                      policy=policies,
                      duplicate=duplicates)

def get_parallel_result_T1XL():
    if machine == "wisteria-o":
        nodes = ["2x3x2:torus", "3x4x3:torus"]
        # nodes = ["2x3x2:torus", "3x4x3:torus", "6x6x4:torus"]
        duplicates = [0, 1, 2]
    else:
        return pd.DataFrame()
    # policies = ["nocache", "writethrough", "writeback", "writeback_lazy"]
    policies = ["nocache", "writeback_lazy"]
    return get_result("T1XL", "nodes_{nodes}_p_{policy}_{duplicate}.out",
                      tree=["T1XL"],
                      nodes=nodes,
                      policy=policies,
                      duplicate=duplicates)

def get_node_counts(tree):
    if tree == "T1L":
        return 102181082
    elif tree == "T1XL":
        return 1635119272
    elif tree == "T1XXL":
        return 4230646601
    elif tree == "T1WL":
        return 270751679750
    else:
        raise Exception("Unknown tree {}".format(tree))

if __name__ == "__main__":
    fig = go.Figure()
    log_axis = dict(type="log", dtick=1, minor=dict(ticks="inside", ticklen=5, showgrid=True))
    yaxis_title = "Throughput (nodes/s)"
    x_axis = log_axis
    y_axis = log_axis

    df_par = pd.concat([get_parallel_result_T1L(), get_parallel_result_T1XL()])

    if plot_serial:
        df_ser = get_serial_result()
        serial_throughput = get_node_counts(df_ser["tree"].iloc[0]) / df_ser["time"].mean()
        print("Serial throughput: {} Gnodes/s".format(serial_throughput))
        serial_throughput *= 1_000_000_000

    df_onenode = df_par[(df_par["policy"] == "nocache") & (df_par["nodes"] == 1)]
    onenode_cores = df_onenode["nproc"].iloc[0]
    onenode_throughput = get_node_counts(df_onenode["tree"].iloc[0]) / df_onenode["time"].mean()
    print("One-node ({} cores) throughput: {} Gnodes/s".format(onenode_cores, onenode_throughput))

    onenode_throughput *= 1_000_000_000

    for policy, df_p in df_par.groupby("policy"):
        print("## policy={}".format(policy))

        xs_all = []
        ys_all = []
        ci_uppers_all = []
        ci_lowers_all = []

        for tree, df_t in df_p.groupby("tree"):
            nc = get_node_counts(tree)

            df_t = df_t.groupby("nproc").agg({"time": ["mean", "median", "min", plot_util.ci_lower, plot_util.ci_upper]})

            xs = df_t.index
            ys = nc / df_t[("time", "mean")]
            ci_uppers = nc / df_t[("time", "ci_lower")] - ys
            ci_lowers = ys - nc / df_t[("time", "ci_upper")]

            ys *= 1_000_000_000
            ci_uppers *= 1_000_000_000
            ci_lowers *= 1_000_000_000

            xs_all.append(None)
            ys_all.append(None)
            ci_uppers_all.append(None)
            ci_lowers_all.append(None)

            xs_all.extend(xs)
            ys_all.extend(ys)
            ci_uppers_all.extend(ci_uppers)
            ci_lowers_all.extend(ci_lowers)

            if policy == "writeback_lazy":
                ay = 0
            else:
                ay = -10 if tree == "T1XL" else 10
            if machine == "wisteria-o":
                fig.add_annotation(
                    x=math.log10(xs.max()),
                    y=math.log10(ys.max()),
                    ax=30,
                    ay=ay,
                    text=tree,
                )

        error_y = dict(type="data", symmetric=False, array=ci_uppers_all, arrayminus=ci_lowers_all, thickness=linewidth)

        looks = policy_looks[policy]

        fig.add_trace(go.Scatter(
            x=xs_all,
            y=ys_all,
            error_y=error_y,
            line_width=linewidth,
            marker_line_width=linewidth,
            marker_color=looks["color"],
            marker_line_color=looks["color"],
            marker_symbol=looks["marker"],
            marker_size=markersize,
            line_dash=looks["dash"],
            name=looks["title"],
            legendrank=looks["rank"],
        ))

    # serial performance
    max_x = max(df_par["nproc"] * 1.5)
    if plot_serial:
        xrange = [0, max_x]
        coef = serial_throughput
        fig.add_trace(go.Scatter(
            x=xrange,
            y=[x * coef for x in xrange],
            marker_color="#999999",
            mode="lines",
            showlegend=False,
            line=dict(dash="2px,2px", width=1),
            name="Ideal",
        ))
        if machine == "wisteria-o":
            textangle = -28 # any automated way?
            ratio = 0.2
            text_x = 10 ** (math.log10(onenode_cores) * (1 - ratio) + math.log10(max_x) * ratio)
            fig.add_annotation(
                x=math.log10(text_x),
                y=math.log10(text_x * coef),
                textangle=textangle,
                font_size=14,
                yshift=22,
                showarrow=False,
                text="Linear speedup<br>(vs. serial)",
            )

    # ideal performance based on 1 node
    xrange = [onenode_cores, max_x]
    coef = onenode_throughput / onenode_cores
    fig.add_trace(go.Scatter(
        x=xrange,
        y=[x * coef for x in xrange],
        marker_color="#999999",
        mode="lines",
        showlegend=False,
        line=dict(dash="4px,3px", width=1.5),
        name="Ideal",
    ))
    if machine == "wisteria-o":
        ratio = 0.5
        text_x = 10 ** (math.log10(onenode_cores) * (1 - ratio) + math.log10(max_x) * ratio)
        fig.add_annotation(
            x=math.log10(text_x),
            y=math.log10(text_x * coef),
            textangle=textangle,
            font_size=14,
            yshift=12,
            showarrow=False,
            text="Linear speedup (vs. 1 node)",
        )

    fig.update_xaxes(
        showline=True,
        linecolor="black",
        ticks="outside",
        title_text="# of cores",
        title_standoff=5,
        **x_axis,
    )
    fig.update_yaxes(
        showline=True,
        linecolor="black",
        ticks="outside",
        title_text=yaxis_title,
        title_standoff=12,
        exponentformat="SI",
        **y_axis,
    )
    fig.update_layout(
        width=350,
        height=300,
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(
            yanchor="top",
            y=1.02,
            xanchor="left",
            x=0.02,
            itemwidth=itemwidth,
        ),
        font=dict(
            family="Linux Biolinum O, sans-serif",
            size=16,
        ),
    )

    plot_util.save_fig(fig, fig_dir, "scaling_{}.html".format(machine))
