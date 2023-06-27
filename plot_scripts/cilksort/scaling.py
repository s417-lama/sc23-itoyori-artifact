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

plot_serial = True
# plot_serial = False

plot_10G = True
# plot_10G = False

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
        title="Write-Through Policy",
        marker="diamond-open",
    ),
    writeback=dict(
        rank=2,
        color="#CCBB44",
        dash="dashdot",
        title="Write-Back Policy",
        marker="square-open",
    ),
    writeback_lazy=dict(
        rank=3,
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
    return get_result("serial", "n_{n_input}_exec_{exec_type}_{duplicate}.out",
                      n_input=[1_000_000_000],
                      exec_type=[0, 1],
                      duplicate=[0])

def get_parallel_result_1G():
    if machine == "wisteria-o":
        nodes = [1, "2:torus", "2x3:torus", "2x3x2:torus", "3x4x3:torus"]
        # nodes = [1, "2:torus", "2x3:torus", "2x3x2:torus", "3x4x3:torus", "6x6x4:torus"]
        # nodes = [1, "2:torus", "2x3:torus", "2x3x2:torus", "3x4x3:torus", "6x6x4:torus", "8x9x8:torus"]
        duplicates = [0, 1, 2]
    elif machine == "local":
        nodes = [1, 2, 4]
        duplicates = [0]
    return get_result("scale1G", "nodes_{nodes}_p_{policy}_{duplicate}.out",
                      n_input=[1_000_000_000],
                      nodes=nodes,
                      policy=["nocache", "writeback_lazy"],
                      # policy=["nocache", "writethrough", "writeback", "writeback_lazy"],
                      duplicate=duplicates)

def get_parallel_result_10G():
    if machine == "wisteria-o":
        nodes = ["2x3:torus", "2x3x2:torus", "3x4x3:torus"]
        # nodes = ["2x3:torus", "2x3x2:torus", "3x4x3:torus", "6x6x4:torus"]
        duplicates = [0, 1, 2]
    else:
        return pd.DataFrame()
    return get_result("scale10G", "nodes_{nodes}_p_{policy}_{duplicate}.out",
                      n_input=[10_000_000_000],
                      nodes=nodes,
                      # policy=["nocache", "writethrough", "writeback", "writeback_lazy"],
                      policy=["nocache", "writeback_lazy"],
                      duplicate=duplicates)

if __name__ == "__main__":
    if plot_serial:
        df_ser = get_serial_result()
        serial_exectimes = dict()
        for n_input, df in df_ser.groupby("n_input"):
            for exec_type, df_e in df.groupby("exec_type"):
                serial_exectimes[(n_input, exec_type)] = df_e["time"].mean()
        print(serial_exectimes)

    fig = go.Figure()
    log_axis = dict(type="log", dtick=1, minor=dict(ticks="inside", ticklen=5, showgrid=True))
    yaxis_title = "Execution time (s)"
    x_axis = log_axis
    y_axis = log_axis

    if plot_10G:
        df_par = pd.concat([get_parallel_result_1G(), get_parallel_result_10G()])
    else:
        df_par = get_parallel_result_1G()

    for policy, df_p in df_par.groupby("policy"):
        print("## policy={}".format(policy))

        xs_all = []
        ys_all = []
        ci_uppers_all = []
        ci_lowers_all = []

        for n_input, df_n in df_p.groupby("n_input"):
            print("### n_input={}...".format(n_input))

            df_n["time"] /= 1_000_000_000.0
            df_n = df_n.groupby("nproc").agg({"time": ["mean", "median", "min", plot_util.ci_lower, plot_util.ci_upper]})
            xs = df_n.index

            ys = df_n[("time", "mean")]
            ci_uppers = df_n[("time", "ci_upper")] - ys
            ci_lowers = ys - df_n[("time", "ci_lower")]

            xs_all.append(None)
            ys_all.append(None)
            ci_uppers_all.append(None)
            ci_lowers_all.append(None)

            xs_all.extend(xs)
            ys_all.extend(ys)
            ci_uppers_all.extend(ci_uppers)
            ci_lowers_all.extend(ci_lowers)

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

    # Ideal (1G, serial)
    if plot_serial:
        base_exec_time = serial_exectimes[(1_000_000_000, 0)]
        base_exec_time /= 1_000_000_000.0
        min_x = df_par["nproc"].min()
        max_x = df_par["nproc"].max() * 1.5
        xrange = [min_x * 0.8, max_x]
        textangle=34
        fig.add_trace(go.Scatter(
            x=xrange,
            y=[base_exec_time / x for x in xrange],
            marker_color="#999999",
            mode="lines",
            line=dict(dash="4px,3px", width=1.5),
            showlegend=False,
            name="Ideal",
        ))
        if machine == "wisteria-o":
            text_x = 100
            fig.add_annotation(
                x=math.log10(text_x),
                y=math.log10(base_exec_time / text_x),
                textangle=textangle,
                font_size=14,
                yshift=-24,
                showarrow=False,
                text="Linear speedup<br>(vs. serial)",
            )

    # 10G ideal
    if plot_serial and plot_10G:
        df_10G_base = df_par[(df_par["n_input"] == 10_000_000_000) &
                             (df_par["nodes"] == "2x3:torus") &
                             (df_par["policy"] == "writeback_lazy")]
        min_x = df_10G_base["nproc"].min()
        xrange = [min_x, max_x]
        base_exec_time = df_10G_base["time"].mean()
        base_exec_time /= 1_000_000_000.0
        base_exec_time *= min_x
        fig.add_trace(go.Scatter(
            x=xrange,
            y=[base_exec_time / x for x in xrange],
            marker_color="#999999",
            mode="lines",
            line=dict(dash="4px,3px", width=1.5),
            showlegend=False,
            name="Ideal",
        ))
        if machine == "wisteria-o":
            text_x = 1000
            fig.add_annotation(
                x=math.log10(text_x),
                y=math.log10(base_exec_time / text_x),
                textangle=textangle,
                font_size=14,
                yshift=-24,
                showarrow=False,
                text="Linear speedup<br>(vs. 288 cores)",
            )

    if machine == "wisteria-o":
        # annotations
        fig.add_annotation(
            x=math.log10(100),
            y=math.log10(10),
            showarrow=False,
            text="<b>1G elements</b>",
        )
        fig.add_annotation(
            x=math.log10(1000),
            y=math.log10(35),
            showarrow=False,
            text="<b>10G elements</b>",
        )

    if machine == "wisteria-o":
        yrange = [math.log10(0.1), math.log10(50)]
    else:
        yrange = None

    fig.update_xaxes(
        showline=True,
        linecolor="black",
        ticks="outside",
        title_text="# of cores",
        title_standoff=5,
        **x_axis,
    )
    fig.update_yaxes(
        range=yrange,
        showline=True,
        linecolor="black",
        ticks="outside",
        title_text=yaxis_title,
        title_standoff=12,
        **y_axis,
    )
    fig.update_layout(
        width=350,
        height=330,
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(
            yanchor="bottom",
            y=0.02,
            xanchor="left",
            x=0.02,
            itemwidth=itemwidth,
        ),
        font=dict(
            family="Linux Biolinum O, sans-serif",
            size=16,
        ),
    )

    plot_util.save_fig(fig, fig_dir, "scaling_exectime_{}.html".format(machine))
