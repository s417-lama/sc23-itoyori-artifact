import os
import sys
import pathlib
import itertools
import math
import numpy as np
import pandas as pd
import plotly.subplots
import plotly.graph_objects as go
import plotly.io as pio

os.chdir(pathlib.Path(__file__).parent.parent.parent)
sys.path.append("./plot_scripts")
import plot_util

# machine = "wisteria-o"
machine = "local"

plot_serial = True
# plot_serial = False

artifacts_dir = "ityrbench_artifacts"

fig_dir = os.path.join("figs", "exafmm")

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
    mpi=dict(
        rank=4,
        color="#AA3377",
        dash="12px,3px,3px,3px,3px,3px",
        title="MPI",
        marker="x-thin",
    ),
)

pio.templates.default = "plotly_white"

linewidth = 2
itemwidth = 60
markersize = 12

n_warmups = 1

def get_result(benchmark, batch_name, filename_template, **args):
    files = [(os.path.join(artifacts_dir, machine, benchmark, batch_name, filename_template.format(**dict(p))), dict(p))
             for p in itertools.product(*[[(k, v) for v in args[k]] for k in args])]
    df = plot_util.txt2df(files, [
            r'# of processes *: *(?P<nproc>\d+)',
            r'-* *Time average loop (?P<i>\d+) *-*',
        ], [
            r'Traverse *: *(?P<time>\d+\.\d+) *s',
        ])
    df = df.loc[df["i"] >= n_warmups].copy()
    df["ncore"] = df["nproc"]
    return df

def get_serial_result():
    return get_result("exafmm", "serial", "n_{n_input}_{duplicate}.out",
                      n_input=[1_000_000, 10_000_000],
                      duplicate=[0])

def get_parallel_result_1M():
    if machine == "wisteria-o":
        # nodes = [1, "2:torus", "2x3:torus", "2x3x2:torus", "3x4x3:torus", "6x6x4:torus"]
        nodes = [1, "2:torus", "2x3:torus", "2x3x2:torus", "3x4x3:torus"]
        duplicates = [0, 1, 2]
    elif machine == "local":
        nodes = [1, 2, 4]
        duplicates = [0]
    return get_result("exafmm", "scale1M", "nodes_{nodes}_p_{policy}_{duplicate}.out",
                      n_input=[1_000_000],
                      nodes=nodes,
                      policy=["nocache", "writethrough", "writeback", "writeback_lazy"],
                      duplicate=duplicates)

def get_parallel_result_10M():
    if machine == "wisteria-o":
        # nodes = [1, "2:torus", "2x3:torus", "2x3x2:torus", "3x4x3:torus", "6x6x4:torus"]
        nodes = [1, "2:torus", "2x3:torus", "2x3x2:torus", "3x4x3:torus"]
        duplicates = [0, 1, 2]
    elif machine == "local":
        return pd.DataFrame()
    return get_result("exafmm", "scale10M", "nodes_{nodes}_p_{policy}_{duplicate}.out",
                      n_input=[10_000_000],
                      nodes=nodes,
                      policy=["nocache", "writethrough", "writeback", "writeback_lazy"],
                      duplicate=duplicates)

def get_mpi_result(benchmark, batch_name, filename_template, **args):
    files = [(os.path.join(artifacts_dir, machine, benchmark, batch_name, filename_template.format(**dict(p))), dict(p))
             for p in itertools.product(*[[(k, v) for v in args[k]] for k in args])]
    df = plot_util.txt2df(files, [
            r'threads *: *(?P<nthread>\d+)',
            r'# of processes *: *(?P<nproc>\d+)',
            r'-* *Time average loop (?P<i>\d+) *-*',
        ], [
            r'Traverse \(total\) *: *(?P<time>\d+\.\d+) *s',
        ])
    df = df.loc[df["i"] >= n_warmups].copy()
    df["ncore"] = df["nproc"] * df["nthread"]
    df["policy"] = "mpi"
    return df

def get_mpi_result_1M():
    if machine == "wisteria-o":
        # nodes = [1, "2:torus", "2x3:torus", "2x3x2:torus", "3x4x3:torus", "6x6x4:torus"]
        nodes = [1, "2:torus", "2x3:torus", "2x3x2:torus", "3x4x3:torus"]
        duplicates = [0, 1, 2]
    elif machine == "local":
        nodes = [1, 2, 4]
        duplicates = [0]
    return get_mpi_result("exafmm_mpi", "scale1M", "nodes_{nodes}_{duplicate}.out",
                          n_input=[1_000_000],
                          nodes=nodes,
                          duplicate=duplicates)

def get_mpi_result_10M():
    if machine == "wisteria-o":
        # nodes = [1, "2:torus", "2x3:torus", "2x3x2:torus", "3x4x3:torus", "6x6x4:torus"]
        nodes = [1, "2:torus", "2x3:torus", "2x3x2:torus", "3x4x3:torus"]
        duplicates = [0, 1, 2]
    elif machine == "local":
        return pd.DataFrame()
    return get_mpi_result("exafmm_mpi", "scale10M", "nodes_{nodes}_{duplicate}.out",
                          n_input=[10_000_000],
                          nodes=nodes,
                          duplicate=duplicates)

if __name__ == "__main__":
    if plot_serial:
        df_ser = get_serial_result()
        serial_t_1m = df_ser[df_ser["n_input"] == 1_000_000]["time"].mean()
        serial_t_10m = df_ser[df_ser["n_input"] == 10_000_000]["time"].mean()
        print(serial_t_1m)
        print(serial_t_10m)

    fig = plotly.subplots.make_subplots(
        rows=1,
        cols=1,
        # shared_xaxes=True,
        x_title="# of cores",
        y_title="Execution time (s)",
        horizontal_spacing=0.08,
    )

    log_axis = dict(type="log", dtick=1, minor=dict(ticks="inside", ticklen=5, showgrid=True))
    x_axis = log_axis
    y_axis = log_axis

    df_par = pd.concat([get_parallel_result_1M(), get_parallel_result_10M(),
                        get_mpi_result_1M(), get_mpi_result_10M()])
    print(df_par)

    for policy, df_p in df_par.groupby("policy"):
        print("## policy={}".format(policy))

        for i, (n_input, df_n) in enumerate(df_p.groupby("n_input")):
            print("### n_input={}...".format(n_input))

            df_n = df_n.groupby("ncore").agg({"time": ["mean", "median", "min", plot_util.ci_lower, plot_util.ci_upper]})
            xs = df_n.index

            ys = df_n[("time", "mean")]
            ci_uppers = df_n[("time", "ci_upper")] - ys
            ci_lowers = ys - df_n[("time", "ci_lower")]

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
                showlegend=(i==0),
            ), row=1, col=i+1)

    if plot_serial:
        # Ideal (1M, serial)
        base_exec_time = serial_t_1m
        min_x = df_par["ncore"].min() * 0.8
        max_x = df_par["ncore"].max() * 1.3
        xrange = [min_x, max_x]
        textangle=40
        fig.add_trace(go.Scatter(
            x=xrange,
            y=[base_exec_time / x for x in xrange],
            marker_color="#999999",
            mode="lines",
            line=dict(dash="4px,3px", width=1.5),
            showlegend=False,
            name="Ideal",
        ), row=1, col=1)
        if machine == "wisteria-o":
            text_x = 80
            fig.add_annotation(
                x=math.log10(text_x),
                y=math.log10(base_exec_time / text_x),
                textangle=textangle,
                font_size=14,
                yshift=-24,
                showarrow=False,
                text="Linear speedup<br>(vs. serial)",
                row=1, col=1,
            )

        # Ideal (10M, serial)
        base_exec_time = serial_t_10m
        min_x = df_par["ncore"].min() * 0.8
        max_x = df_par["ncore"].max() * 1.3
        xrange = [min_x, max_x]
        textangle=40
        fig.add_trace(go.Scatter(
            x=xrange,
            y=[base_exec_time / x for x in xrange],
            marker_color="#999999",
            mode="lines",
            line=dict(dash="4px,3px", width=1.5),
            showlegend=False,
            name="Ideal",
        ), row=1, col=1)
        # if machine == "wisteria-o":
        #     text_x = 80
        #     fig.add_annotation(
        #         x=math.log10(text_x),
        #         y=math.log10(base_exec_time / text_x),
        #         textangle=textangle,
        #         font_size=14,
        #         yshift=-24,
        #         showarrow=False,
        #         text="Linear speedup<br>(vs. serial)",
        #         row=1, col=1,
        #     )

    # fig.add_annotation(
    #     x=0.35,
    #     y=1.0,
    #     xref="paper",
    #     yref="paper",
    #     showarrow=False,
    #     text="<b>1M bodies</b>",
    # )
    # fig.add_annotation(
    #     x=0.98,
    #     y=1.0,
    #     xref="paper",
    #     yref="paper",
    #     showarrow=False,
    #     text="<b>10M bodies</b>",
    # )

    if machine == "wisteria-o":
        yrange_1m = [math.log10(1), math.log10(100)]
        yrange_10m = [math.log10(10), math.log10(1000)]
    else:
        yrange_1m = None
        yrange_10m = None

    fig.update_xaxes(
        showline=True,
        linecolor="black",
        ticks="outside",
        title_standoff=5,
        **x_axis,
    )
    fig.update_yaxes(
        showline=True,
        linecolor="black",
        ticks="outside",
        title_standoff=12,
        **y_axis,
    )
    fig.update_yaxes(
        range=yrange_1m,
        row=1, col=2,
    )
    fig.update_yaxes(
        range=yrange_10m,
        row=1, col=1,
    )
    fig.update_layout(
        width=350,
        height=330,
        margin=dict(l=60, r=0, b=60, t=0),
        # legend=dict(
        #     orientation="h",
        #     yanchor="bottom",
        #     y=1.02,
        #     xanchor="left",
        #     x=-0.1,
        #     itemwidth=itemwidth,
        # ),
        legend=dict(
            yanchor="bottom",
            y=0,
            xanchor="left",
            x=0,
            itemwidth=itemwidth,
            bgcolor="rgba(0,0,0,0)",
        ),
        font=dict(
            family="Linux Biolinum O, sans-serif",
            size=16,
        ),
    )
    axis_font = dict(
        family="Linux Biolinum O, sans-serif",
        size=20,
    )
    # for shared axis
    fig.layout.annotations[0]["font"] = axis_font
    fig.layout.annotations[1]["font"] = axis_font

    plot_util.save_fig(fig, fig_dir, "scaling_{}.html".format(machine))
