from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "benchmark_results_interaction_snapshot.csv"
FIGURE_DIR = ROOT / "TeX" / "figures" / "benchmarks"
SUMMARY_PATH = ROOT / "results" / "interaction_snapshot_analysis_summary.txt"
BEST_TABLE_PATH = FIGURE_DIR / "interaction_snapshot_best_configs_table.tex"

REQUIRED_COLUMNS = {
    "Suite",
    "Scheduler",
    "Threads",
    "Agents",
    "GrainSize",
    "InteractionMode",
    "PerceptionRadius",
    "Frames",
    "AvgFrameMs",
    "MinFrameMs",
    "MaxFrameMs",
    "Speedup",
    "Efficiency",
    "ThroughputAgentsPerSec",
}

SCHEDULER_COLORS = {
    "Sequential": "#4f5965",
    "ThreadPool": "#2f6fbd",
    "WorkStealing": "#c45131",
}


@dataclass(frozen=True)
class PairwiseStats:
    pairs: int
    thread_pool_wins: int
    work_stealing_wins: int
    equal: int
    mean_ratio: float
    median_ratio: float
    max_work_stealing_gain: float
    max_thread_pool_gain: float


def load_data(path: Path = CSV_PATH) -> pd.DataFrame:
    """Read and validate the benchmark CSV."""
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    int_columns = ["Threads", "Agents", "GrainSize", "PerceptionRadius", "Frames"]
    for column in int_columns:
        df[column] = df[column].astype(int)

    numeric_columns = [
        "AvgFrameMs",
        "MinFrameMs",
        "MaxFrameMs",
        "Speedup",
        "Efficiency",
        "ThroughputAgentsPerSec",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column])

    df["FrameRangeRatio"] = (df["MaxFrameMs"] - df["MinFrameMs"]) / df["AvgFrameMs"]
    return df.sort_values(
        ["Agents", "GrainSize", "PerceptionRadius", "Scheduler", "Threads"]
    ).reset_index(drop=True)


def parallel_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Scheduler"].isin(["ThreadPool", "WorkStealing"])].copy()


def latex_decimal(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", "{,}")


def latex_int(value: int) -> str:
    return f"{int(value)}"


def scheduler_short(name: str) -> str:
    return {"ThreadPool": "TP", "WorkStealing": "WS", "Sequential": "Seq"}.get(name, name)


def save_figure(fig: plt.Figure, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def style_axis(ax: plt.Axes, ylabel: str | None = None, xlabel: str | None = None) -> None:
    ax.grid(True, axis="y", alpha=0.28)
    ax.grid(True, axis="x", alpha=0.12)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)


def plot_scaling_by_agents(df: pd.DataFrame) -> None:
    grouped_time = (
        df.groupby(["Agents", "Scheduler"], as_index=False)["AvgFrameMs"].mean()
        .sort_values(["Scheduler", "Agents"])
    )
    grouped_speedup = (
        parallel_only(df)
        .groupby(["Agents", "Scheduler"], as_index=False)["Speedup"]
        .mean()
        .sort_values(["Scheduler", "Agents"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for scheduler in ["Sequential", "ThreadPool", "WorkStealing"]:
        subset = grouped_time[grouped_time["Scheduler"] == scheduler]
        axes[0].plot(
            subset["Agents"],
            subset["AvgFrameMs"],
            marker="o",
            label=scheduler,
            color=SCHEDULER_COLORS[scheduler],
        )
    axes[0].set_xscale("log")
    axes[0].set_title("Среднее время кадра")
    style_axis(axes[0], "AvgFrameMs, мс", "Число агентов N")

    for scheduler in ["ThreadPool", "WorkStealing"]:
        subset = grouped_speedup[grouped_speedup["Scheduler"] == scheduler]
        axes[1].plot(
            subset["Agents"],
            subset["Speedup"],
            marker="o",
            label=scheduler,
            color=SCHEDULER_COLORS[scheduler],
        )
    axes[1].axhline(1.0, color="#666666", linewidth=1.0, linestyle="--")
    axes[1].set_xscale("log")
    axes[1].set_title("Ускорение относительно Sequential")
    style_axis(axes[1], "Speedup", "Число агентов N")

    for ax in axes:
        ax.legend(frameon=False)
    save_figure(fig, "interaction_snapshot_scaling_by_agents.pdf")


def plot_radius_effect(df: pd.DataFrame) -> None:
    grouped_time = (
        df.groupby(["PerceptionRadius", "Scheduler"], as_index=False)["AvgFrameMs"].mean()
        .sort_values(["Scheduler", "PerceptionRadius"])
    )
    grouped_speedup = (
        parallel_only(df)
        .groupby(["PerceptionRadius", "Scheduler"], as_index=False)["Speedup"]
        .mean()
        .sort_values(["Scheduler", "PerceptionRadius"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for scheduler in ["Sequential", "ThreadPool", "WorkStealing"]:
        subset = grouped_time[grouped_time["Scheduler"] == scheduler]
        axes[0].plot(
            subset["PerceptionRadius"],
            subset["AvgFrameMs"],
            marker="o",
            label=scheduler,
            color=SCHEDULER_COLORS[scheduler],
        )
    axes[0].set_title("Стоимость обновления")
    style_axis(axes[0], "AvgFrameMs, мс", "PerceptionRadius")

    for scheduler in ["ThreadPool", "WorkStealing"]:
        subset = grouped_speedup[grouped_speedup["Scheduler"] == scheduler]
        axes[1].plot(
            subset["PerceptionRadius"],
            subset["Speedup"],
            marker="o",
            label=scheduler,
            color=SCHEDULER_COLORS[scheduler],
        )
    axes[1].axhline(1.0, color="#666666", linewidth=1.0, linestyle="--")
    axes[1].set_title("Окупаемость параллелизма")
    style_axis(axes[1], "Speedup", "PerceptionRadius")

    for ax in axes:
        ax.set_xticks(sorted(df["PerceptionRadius"].unique()))
        ax.legend(frameon=False)
    save_figure(fig, "interaction_snapshot_radius_effect.pdf")


def plot_grain_effect(df: pd.DataFrame) -> None:
    grouped = (
        parallel_only(df)
        .groupby(["GrainSize", "Scheduler", "Threads"], as_index=False)["AvgFrameMs"]
        .mean()
        .sort_values(["Scheduler", "Threads", "GrainSize"])
    )

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    markers = {4: "o", 8: "s"}
    linestyles = {4: "-", 8: "--"}
    for scheduler in ["ThreadPool", "WorkStealing"]:
        for threads in [4, 8]:
            subset = grouped[
                (grouped["Scheduler"] == scheduler) & (grouped["Threads"] == threads)
            ]
            ax.plot(
                subset["GrainSize"],
                subset["AvgFrameMs"],
                marker=markers[threads],
                linestyle=linestyles[threads],
                color=SCHEDULER_COLORS[scheduler],
                label=f"{scheduler}, {threads} пот.",
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(df["GrainSize"].unique()))
    ax.set_xticklabels([str(x) for x in sorted(df["GrainSize"].unique())])
    style_axis(ax, "AvgFrameMs, мс", "grainSize")
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, "interaction_snapshot_grain_effect.pdf")


def scheduler_pairs(df: pd.DataFrame) -> pd.DataFrame:
    parallel = parallel_only(df)
    pairs = parallel.pivot_table(
        index=["Agents", "PerceptionRadius", "GrainSize", "Threads"],
        columns="Scheduler",
        values=["AvgFrameMs", "MaxFrameMs", "FrameRangeRatio"],
        aggfunc="first",
    )
    pairs.columns = [f"{metric}_{scheduler}" for metric, scheduler in pairs.columns]
    pairs = pairs.reset_index()
    pairs["TimeRatio_TP_over_WS"] = (
        pairs["AvgFrameMs_ThreadPool"] / pairs["AvgFrameMs_WorkStealing"]
    )
    pairs["MaxRatio_TP_over_WS"] = (
        pairs["MaxFrameMs_ThreadPool"] / pairs["MaxFrameMs_WorkStealing"]
    )
    return pairs


def plot_scheduler_ratio(pairs: pd.DataFrame) -> None:
    radii = sorted(pairs["PerceptionRadius"].unique())
    data = [
        pairs[pairs["PerceptionRadius"] == radius]["TimeRatio_TP_over_WS"].to_numpy()
        for radius in radii
    ]

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.boxplot(data, tick_labels=[str(radius) for radius in radii], showmeans=True)
    ax.axhline(1.0, color="#666666", linewidth=1.0, linestyle="--")
    style_axis(ax, "AvgFrameMs(ThreadPool) / AvgFrameMs(WorkStealing)", "PerceptionRadius")
    ax.set_title("Парное сравнение планировщиков")
    save_figure(fig, "interaction_snapshot_scheduler_ratio.pdf")


def thread_scaling_pairs(df: pd.DataFrame) -> pd.DataFrame:
    parallel = parallel_only(df)
    pairs = parallel.pivot_table(
        index=["Agents", "PerceptionRadius", "GrainSize", "Scheduler"],
        columns="Threads",
        values="AvgFrameMs",
        aggfunc="first",
    ).reset_index()
    pairs = pairs.rename(columns={4: "AvgFrameMs_4", 8: "AvgFrameMs_8"})
    pairs["Speedup_4_to_8"] = pairs["AvgFrameMs_4"] / pairs["AvgFrameMs_8"]
    return pairs


def plot_thread_scaling(thread_pairs: pd.DataFrame) -> None:
    grouped = (
        thread_pairs.groupby(["PerceptionRadius", "Scheduler"], as_index=False)[
            "Speedup_4_to_8"
        ]
        .mean()
        .sort_values(["Scheduler", "PerceptionRadius"])
    )

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    width = 0.34
    radii = sorted(grouped["PerceptionRadius"].unique())
    positions = range(len(radii))
    for offset, scheduler in [(-width / 2, "ThreadPool"), (width / 2, "WorkStealing")]:
        subset = grouped[grouped["Scheduler"] == scheduler]
        values = [subset[subset["PerceptionRadius"] == r]["Speedup_4_to_8"].iloc[0] for r in radii]
        ax.bar(
            [p + offset for p in positions],
            values,
            width=width,
            label=scheduler,
            color=SCHEDULER_COLORS[scheduler],
        )
    ax.axhline(2.0, color="#666666", linewidth=1.0, linestyle="--", label="Идеал")
    ax.set_xticks(list(positions))
    ax.set_xticklabels([str(r) for r in radii])
    style_axis(ax, "Speedup(4->8)", "PerceptionRadius")
    ax.legend(frameon=False, ncol=3)
    save_figure(fig, "interaction_snapshot_thread_scaling.pdf")


def plot_stability(df: pd.DataFrame) -> None:
    grouped = (
        df.groupby(["PerceptionRadius", "Scheduler"], as_index=False)["FrameRangeRatio"]
        .mean()
        .sort_values(["Scheduler", "PerceptionRadius"])
    )

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    width = 0.24
    radii = sorted(grouped["PerceptionRadius"].unique())
    positions = range(len(radii))
    layout = [(-width, "Sequential"), (0.0, "ThreadPool"), (width, "WorkStealing")]
    for offset, scheduler in layout:
        subset = grouped[grouped["Scheduler"] == scheduler]
        values = [subset[subset["PerceptionRadius"] == r]["FrameRangeRatio"].iloc[0] for r in radii]
        ax.bar(
            [p + offset for p in positions],
            values,
            width=width,
            label=scheduler,
            color=SCHEDULER_COLORS[scheduler],
        )
    ax.set_xticks(list(positions))
    ax.set_xticklabels([str(r) for r in radii])
    style_axis(ax, "(MaxFrameMs - MinFrameMs) / AvgFrameMs", "PerceptionRadius")
    ax.legend(frameon=False, ncol=3)
    save_figure(fig, "interaction_snapshot_stability.pdf")


def compute_pairwise_stats(pairs: pd.DataFrame) -> PairwiseStats:
    ratio = pairs["TimeRatio_TP_over_WS"]
    tp_wins = int((ratio < 1.0).sum())
    ws_wins = int((ratio > 1.0).sum())
    equal = int((ratio == 1.0).sum())
    ws_gain = float(ratio.max())
    tp_gain = float((1.0 / ratio).max())
    return PairwiseStats(
        pairs=len(pairs),
        thread_pool_wins=tp_wins,
        work_stealing_wins=ws_wins,
        equal=equal,
        mean_ratio=float(ratio.mean()),
        median_ratio=float(ratio.median()),
        max_work_stealing_gain=ws_gain,
        max_thread_pool_gain=tp_gain,
    )


def best_rows(df: pd.DataFrame, group: Iterable[str]) -> pd.DataFrame:
    group = list(group)
    indexes = df.groupby(group)["AvgFrameMs"].idxmin()
    return df.loc[indexes].sort_values(group).reset_index(drop=True)


def format_best_row(row: pd.Series, label: str, value: str) -> str:
    return (
        f"{label} & {value} & {latex_int(row['Agents'])} & "
        f"{latex_int(row['PerceptionRadius'])} & {latex_int(row['GrainSize'])} & "
        f"{latex_int(row['Threads'])} & \\texttt{{{row['Scheduler']}}} & "
        f"{latex_decimal(row['AvgFrameMs'], 2)} & {latex_decimal(row['Speedup'], 2)} \\\\"
    )


def write_best_config_table(df: pd.DataFrame) -> None:
    # The table compares measured configurations as they are present in the CSV.
    by_agents = best_rows(df, ["Agents"])
    by_radius = best_rows(df, ["PerceptionRadius"])
    global_best = df.loc[[df["AvgFrameMs"].idxmin()]]

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Лучшие конфигурации режима \\texttt{SnapshotNeighbors}}",
        "\\label{tab:interaction-snapshot-best}",
        "\\begin{tabularx}{\\textwidth}{|l|r|r|r|r|r|>{\\raggedright\\arraybackslash}X|r|r|}",
        "\\hline",
        "Срез & Знач. & $N$ & $R$ & $g$ & $p$ & Планировщик & Avg, мс & $S$ \\\\",
        "\\hline",
    ]
    for _, row in by_agents.iterrows():
        lines.append(format_best_row(row, "$N$", latex_int(row["Agents"])))
    lines.append("\\hline")
    for _, row in by_radius.iterrows():
        lines.append(format_best_row(row, "$R$", latex_int(row["PerceptionRadius"])))
    lines.append("\\hline")
    row = global_best.iloc[0]
    lines.append(format_best_row(row, "Глобально", "--"))
    lines.extend(["\\hline", "\\end{tabularx}", "\\end{table}", ""])

    BEST_TABLE_PATH.write_text("\n".join(lines), encoding="utf-8")


def build_summary(df: pd.DataFrame, pairs: pd.DataFrame, thread_pairs: pd.DataFrame) -> str:
    pair_stats = compute_pairwise_stats(pairs)
    parallel = parallel_only(df)

    speedup_by_n = (
        parallel.groupby(["Agents", "Scheduler"])["Speedup"]
        .agg(["mean", "max"])
        .reset_index()
        .sort_values(["Agents", "Scheduler"])
    )
    payoff_by_n_radius = (
        parallel.assign(PaysOff=parallel["Speedup"] > 1.0)
        .groupby(["Agents", "PerceptionRadius", "Scheduler"])
        .agg(Configs=("Speedup", "size"), PaysOff=("PaysOff", "sum"), MaxSpeedup=("Speedup", "max"))
        .reset_index()
        .sort_values(["Agents", "PerceptionRadius", "Scheduler"])
    )
    speedup_by_radius = (
        parallel.groupby(["PerceptionRadius", "Scheduler"])["Speedup"]
        .agg(["mean", "max"])
        .reset_index()
        .sort_values(["PerceptionRadius", "Scheduler"])
    )
    radius_summary = (
        df.groupby(["PerceptionRadius", "Scheduler"])["AvgFrameMs"]
        .mean()
        .reset_index()
        .sort_values(["PerceptionRadius", "Scheduler"])
    )
    grain_summary = (
        parallel.groupby(["GrainSize", "Scheduler", "Threads"])["AvgFrameMs"]
        .mean()
        .reset_index()
        .sort_values(["GrainSize", "Scheduler", "Threads"])
    )
    thread_summary = (
        thread_pairs.groupby("Scheduler")["Speedup_4_to_8"]
        .agg(["mean", "median", "min", "max"])
        .reset_index()
    )
    stability_by_radius = (
        df.groupby(["PerceptionRadius", "Scheduler"])["FrameRangeRatio"]
        .mean()
        .reset_index()
        .sort_values(["PerceptionRadius", "Scheduler"])
    )
    stability_by_grain = (
        df.groupby(["GrainSize", "Scheduler"])["FrameRangeRatio"]
        .mean()
        .reset_index()
        .sort_values(["GrainSize", "Scheduler"])
    )

    best_grain_counts = (
        parallel.loc[parallel.groupby(["Agents", "PerceptionRadius", "Scheduler", "Threads"])["AvgFrameMs"].idxmin()]
        .groupby(["Scheduler", "GrainSize"])
        .size()
        .reset_index(name="Wins")
        .sort_values(["Scheduler", "GrainSize"])
    )

    best_max_pairs = pairs.copy()
    best_max_pairs["AvgWinner"] = best_max_pairs.apply(
        lambda row: "ThreadPool"
        if row["AvgFrameMs_ThreadPool"] < row["AvgFrameMs_WorkStealing"]
        else ("WorkStealing" if row["AvgFrameMs_WorkStealing"] < row["AvgFrameMs_ThreadPool"] else "Equal"),
        axis=1,
    )
    best_max_pairs["WorstFrameWinner"] = best_max_pairs.apply(
        lambda row: "ThreadPool"
        if row["MaxFrameMs_ThreadPool"] < row["MaxFrameMs_WorkStealing"]
        else ("WorkStealing" if row["MaxFrameMs_WorkStealing"] < row["MaxFrameMs_ThreadPool"] else "Equal"),
        axis=1,
    )
    avg_wins_by_radius = (
        best_max_pairs.groupby(["PerceptionRadius", "AvgWinner"])
        .size()
        .reset_index(name="Wins")
        .sort_values(["PerceptionRadius", "AvgWinner"])
    )
    worst_frame_wins = best_max_pairs["WorstFrameWinner"].value_counts().sort_index()

    lines = [
        "Interaction snapshot analysis",
        "=============================",
        f"Rows: {len(df)}",
        f"Agents: {sorted(int(x) for x in df['Agents'].unique())}",
        f"PerceptionRadius: {sorted(int(x) for x in df['PerceptionRadius'].unique())}",
        f"GrainSize: {sorted(int(x) for x in df['GrainSize'].unique())}",
        f"Schedulers: {sorted(df['Scheduler'].unique())}",
        "",
        "Speedup by N (mean/max):",
        speedup_by_n.to_string(index=False),
        "",
        "Payoff by N, PerceptionRadius and Scheduler:",
        payoff_by_n_radius.to_string(index=False),
        "",
        "Speedup by PerceptionRadius (mean/max):",
        speedup_by_radius.to_string(index=False),
        "",
        "AvgFrameMs by PerceptionRadius and Scheduler:",
        radius_summary.to_string(index=False),
        "",
        "AvgFrameMs by GrainSize, Scheduler and Threads:",
        grain_summary.to_string(index=False),
        "",
        "ThreadPool vs WorkStealing:",
        f"pairs={pair_stats.pairs}",
        f"ThreadPool wins={pair_stats.thread_pool_wins}",
        f"WorkStealing wins={pair_stats.work_stealing_wins}",
        f"equal={pair_stats.equal}",
        f"mean TP/WS ratio={pair_stats.mean_ratio:.6f}",
        f"median TP/WS ratio={pair_stats.median_ratio:.6f}",
        f"max WorkStealing gain={pair_stats.max_work_stealing_gain:.6f}",
        f"max ThreadPool gain={pair_stats.max_thread_pool_gain:.6f}",
        "",
        "Speedup 4->8:",
        thread_summary.to_string(index=False),
        "",
        "Best grainSize counts among parallel configurations:",
        best_grain_counts.to_string(index=False),
        "",
        "Frame range ratio by PerceptionRadius and Scheduler:",
        stability_by_radius.to_string(index=False),
        "",
        "Frame range ratio by GrainSize and Scheduler:",
        stability_by_grain.to_string(index=False),
        "",
        "AvgFrameMs winner by PerceptionRadius:",
        avg_wins_by_radius.to_string(index=False),
        "",
        "Better MaxFrameMs in pairwise comparison:",
        worst_frame_wins.to_string(),
        "",
        "Best by N:",
        best_rows(df, ["Agents"])[
            ["Agents", "Scheduler", "Threads", "GrainSize", "PerceptionRadius", "AvgFrameMs", "Speedup"]
        ].to_string(index=False),
        "",
        "Best by PerceptionRadius:",
        best_rows(df, ["PerceptionRadius"])[
            ["PerceptionRadius", "Scheduler", "Threads", "Agents", "GrainSize", "AvgFrameMs", "Speedup"]
        ].to_string(index=False),
        "",
        "Global best:",
        df.loc[[df["AvgFrameMs"].idxmin()]][
            ["Agents", "PerceptionRadius", "Scheduler", "Threads", "GrainSize", "AvgFrameMs", "Speedup"]
        ].to_string(index=False),
        "",
    ]
    return "\n".join(lines)


def print_key_statistics(summary: str) -> None:
    print(summary)


def make_plots(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = scheduler_pairs(df)
    thread_pairs = thread_scaling_pairs(df)
    plot_scaling_by_agents(df)
    plot_radius_effect(df)
    plot_grain_effect(df)
    plot_scheduler_ratio(pairs)
    plot_thread_scaling(thread_pairs)
    plot_stability(df)
    return pairs, thread_pairs


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    pairs, thread_pairs = make_plots(df)
    write_best_config_table(df)
    summary = build_summary(df, pairs, thread_pairs)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print_key_statistics(summary)
    print(f"\nFigures written to: {FIGURE_DIR}")
    print(f"Summary written to: {SUMMARY_PATH}")
    print(f"LaTeX table written to: {BEST_TABLE_PATH}")


if __name__ == "__main__":
    main()
