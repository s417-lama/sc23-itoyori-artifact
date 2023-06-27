import os
import sys
import pathlib
import itertools
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

plot_types = ["exectime", "speedup"]

artifacts_dir = "ityrbench_artifacts"

fig_dir = os.path.join("figs", benchmark)

policy_looks = dict(
    nocache=dict(
        rank=0,
        color="#EE6677",
        dash="dot",
        title="No Cache",
        marker="circle-open",
    ),
    writethrough=dict(
        rank=1,
        color="#4477AA",
        dash="dash",
        title="Write-Through",
        marker="diamond-open",
    ),
    writeback=dict(
        rank=2,
        color="#CCBB44",
        dash="dashdot",
        title="Write-Back",
        marker="square-open",
    ),
    writeback_lazy=dict(
        rank=3,
        color="#228833",
        dash="solid",
        title="Write-Back (Lazy)",
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

def get_result_granularity():
    # duplicates = [0, 1, 2]
    duplicates = [0]
    return get_result("granularity", "c_{cutoff}_p_{policy}_{duplicate}.out",
                      cutoff=[64, 256, 1024, 4096, 16384, 65536],
                      policy=["nocache", "writethrough", "writeback", "writeback_lazy"],
                      duplicate=duplicates)

if __name__ == "__main__":
    fig = go.Figure()
    log_axis = dict(type="log", dtick=1, minor=dict(ticks="inside", ticklen=5, showgrid=True))
    x_axis = log_axis
    y_axis = log_axis
    # y_axis = dict(rangemode="tozero")

    df_all = get_result_granularity()
    for policy, df_p in df_all.groupby("policy"):
        print("## policy={}".format(policy))

        df_p["time"] /= 1_000_000_000.0
        df_c = df_p.groupby("cutoff").agg({"time": ["mean", "median", "min", plot_util.ci_lower, plot_util.ci_upper]})

        xs = df_c.index
        ys = df_c[("time", "mean")]
        ci_uppers = df_c[("time", "ci_upper")] - ys
        ci_lowers = ys - df_c[("time", "ci_lower")]
        error_y = dict(type="data", symmetric=False, array=ci_uppers, arrayminus=ci_lowers, thickness=linewidth)

        looks = policy_looks[policy]

        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
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

    fig.update_xaxes(
        **x_axis,
        showline=True,
        linecolor="black",
        ticks="outside",
        title_text="Cutoff count (# of elements)",
        title_standoff=7,
    )
    fig.update_yaxes(
        **y_axis,
        showline=True,
        linecolor="black",
        ticks="outside",
        title_text="Execution Time (s)",
        title_standoff=12,
    )
    fig.update_layout(
        width=350,
        height=300,
        margin=dict(l=0, r=10, b=0, t=0),
        legend=dict(
            yanchor="top",
            xanchor="right",
            y=1,
            x=1,
            itemwidth=itemwidth,
        ),
        font=dict(
            family="Linux Biolinum O, sans-serif",
            size=16,
        ),
    )

    plot_util.save_fig(fig, fig_dir, "granularity_{}.html".format(machine))
