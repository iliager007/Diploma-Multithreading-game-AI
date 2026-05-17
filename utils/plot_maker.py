# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
LIGHT_CSV = ROOT / "benchmark_results_light_overhead.csv"
HEAVY_CSV = ROOT / "benchmark_results_heavy_parallel.csv"
BASE_CSV = ROOT / "benchmark_results.csv"

FIG_DIR = ROOT / "figures" / "benchmarks"
TABLE_DIR = ROOT / "tables" / "benchmarks"


def savefig(name: str) -> None:
    """Сохраняет текущий график в PDF и PNG."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"{name}.{ext}"
        plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()


def prepare_light(light: pd.DataFrame) -> pd.DataFrame:
    """Возвращает таблицу парных сравнений ThreadPool и WorkStealing."""
    parallel = light[light["Scheduler"].isin(["ThreadPool", "WorkStealing"])].copy()
    pivot = parallel.pivot_table(
        index=["Agents", "Threads", "GrainSize", "HeavyRatio", "HeavyIterations"],
        columns="Scheduler",
        values="AvgFrameMs",
        aggfunc="first",
    ).reset_index()

    seq = light[light["Scheduler"] == "Sequential"][
        ["Agents", "GrainSize", "AvgFrameMs"]
    ].rename(columns={"AvgFrameMs": "Sequential"})

    paired = pivot.merge(seq, on=["Agents", "GrainSize"], how="left")
    paired["Speedup_ThreadPool"] = paired["Sequential"] / paired["ThreadPool"]
    paired["Speedup_WorkStealing"] = paired["Sequential"] / paired["WorkStealing"]
    paired["S_WS_TP"] = paired["ThreadPool"] / paired["WorkStealing"]
    paired["WS_TP_ratio"] = paired["WorkStealing"] / paired["ThreadPool"]
    return paired


def prepare_heavy(heavy: pd.DataFrame) -> pd.DataFrame:
    """Возвращает таблицу парных сравнений ThreadPool и WorkStealing."""
    pivot = heavy.pivot_table(
        index=["Agents", "Threads", "GrainSize", "HeavyRatio", "HeavyIterations", "HeavyLoad"],
        columns="Scheduler",
        values="AvgFrameMs",
        aggfunc="first",
    ).reset_index()
    pivot["HeavyRatioRounded"] = pivot["HeavyRatio"].round(2)
    pivot["S_WS_TP"] = pivot["ThreadPool"] / pivot["WorkStealing"]
    pivot["WS_TP_ratio"] = pivot["WorkStealing"] / pivot["ThreadPool"]
    return pivot


def prepare_heavy_stability(heavy: pd.DataFrame) -> pd.DataFrame:
    """Возвращает парную таблицу Avg/Min/Max для анализа стабильности."""
    index = ["Agents", "Threads", "GrainSize", "HeavyRatio", "HeavyIterations", "HeavyLoad"]
    frames = []
    for metric in ["AvgFrameMs", "MinFrameMs", "MaxFrameMs"]:
        part = heavy.pivot_table(
            index=index,
            columns="Scheduler",
            values=metric,
            aggfunc="first",
        ).reset_index()
        part = part.rename(columns={
            "ThreadPool": f"{metric}_ThreadPool",
            "WorkStealing": f"{metric}_WorkStealing",
        })
        frames.append(part)

    paired = frames[0]
    for part in frames[1:]:
        paired = paired.merge(part, on=index, how="inner")

    paired["HeavyRatioRounded"] = paired["HeavyRatio"].round(2)
    for scheduler in ["ThreadPool", "WorkStealing"]:
        paired[f"RangeFrameMs_{scheduler}"] = (
            paired[f"MaxFrameMs_{scheduler}"] - paired[f"MinFrameMs_{scheduler}"]
        )
        paired[f"RelativeRange_{scheduler}"] = (
            paired[f"RangeFrameMs_{scheduler}"] / paired[f"AvgFrameMs_{scheduler}"]
        )
    paired["MaxFrameRatio_TP_over_WS"] = (
        paired["MaxFrameMs_ThreadPool"] / paired["MaxFrameMs_WorkStealing"]
    )
    paired["RangeRatio_TP_over_WS"] = (
        paired["RangeFrameMs_ThreadPool"] / paired["RangeFrameMs_WorkStealing"]
    )
    paired["RelativeRangeRatio_TP_over_WS"] = (
        paired["RelativeRange_ThreadPool"] / paired["RelativeRange_WorkStealing"]
    )
    return paired


def plot_light_frame_by_agents(paired: pd.DataFrame) -> None:
    """Лучшее время кадра по числу агентов в фиксированной конфигурации."""
    threads = 8
    grain = 1024
    data = paired[(paired["Threads"] == threads) & (paired["GrainSize"] == grain)].sort_values("Agents")

    plt.figure(figsize=(7.2, 4.4))
    plt.plot(data["Agents"], data["Sequential"], marker="o", label="Sequential")
    plt.plot(data["Agents"], data["ThreadPool"], marker="o", label=f"ThreadPool, {threads} потоков")
    plt.plot(data["Agents"], data["WorkStealing"], marker="o", label=f"WorkStealing, {threads} потоков")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Число агентов")
    plt.ylabel("Среднее время кадра, мс")
    plt.title(f"Лёгкая нагрузка: время кадра при grainSize={grain}")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    savefig("light_fixed_config_frame_by_agents")


def plot_light_grain_effect(paired: pd.DataFrame) -> None:
    """Влияние grainSize при большом числе агентов."""
    agents = 1_000_000
    threads = 8
    data = paired[(paired["Agents"] == agents) & (paired["Threads"] == threads)].sort_values("GrainSize")

    plt.figure(figsize=(7.2, 4.4))
    plt.plot(data["GrainSize"], data["Sequential"], marker="o", label="Sequential")
    plt.plot(data["GrainSize"], data["ThreadPool"], marker="o", label=f"ThreadPool, {threads} потоков")
    plt.plot(data["GrainSize"], data["WorkStealing"], marker="o", label=f"WorkStealing, {threads} потоков")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Размер задачи grainSize, агентов")
    plt.ylabel("Среднее время кадра, мс")
    plt.title(f"Лёгкая нагрузка: влияние grainSize при N={agents:,}".replace(",", " "))
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    savefig("light_grain_effect_large_agents")


def plot_light_speedup_by_agents(paired: pd.DataFrame) -> None:
    """Speedup относительно Sequential при фиксированных threads и grainSize."""
    threads = 8
    grain = 1024
    data = paired[(paired["Threads"] == threads) & (paired["GrainSize"] == grain)].sort_values("Agents")

    plt.figure(figsize=(7.2, 4.4))
    plt.plot(data["Agents"], data["Speedup_ThreadPool"], marker="o", label="ThreadPool")
    plt.plot(data["Agents"], data["Speedup_WorkStealing"], marker="o", label="WorkStealing")
    plt.axhline(1.0, linestyle="--", linewidth=1.0, label="граница ускорения")
    plt.xscale("log")
    plt.xlabel("Число агентов")
    plt.ylabel("Speedup относительно Sequential")
    plt.title(f"Лёгкая нагрузка: speedup при {threads} потоках и grainSize={grain}")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    savefig("light_speedup_fixed_config_by_agents")


def plot_light_fine_grain_overhead(paired: pd.DataFrame) -> None:
    """Показывает накладные расходы слишком мелкого разбиения."""
    agents = 1_000_000
    threads = 8
    data = paired[(paired["Agents"] == agents) & (paired["Threads"] == threads)].sort_values("GrainSize").copy()
    data["ThreadPool_over_Seq"] = data["ThreadPool"] / data["Sequential"]
    data["WorkStealing_over_Seq"] = data["WorkStealing"] / data["Sequential"]

    plt.figure(figsize=(7.2, 4.4))
    plt.plot(data["GrainSize"], data["ThreadPool_over_Seq"], marker="o", label="ThreadPool / Sequential")
    plt.plot(data["GrainSize"], data["WorkStealing_over_Seq"], marker="o", label="WorkStealing / Sequential")
    plt.axhline(1.0, linestyle="--", linewidth=1.0, label="паритет с Sequential")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Размер задачи grainSize, агентов")
    plt.ylabel("Отношение времени к Sequential")
    plt.title(f"Накладные расходы мелких задач (N={agents:,}, 8 потоков)".replace(",", " "))
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    savefig("light_fine_grain_overhead")


def plot_light_ws_tp_ratio(paired: pd.DataFrame) -> None:
    """Относительное сравнение WorkStealing и ThreadPool при одинаковых параметрах."""
    agents = 1_000_000
    threads = 8
    data = paired[(paired["Agents"] == agents) & (paired["Threads"] == threads)].sort_values("GrainSize")

    plt.figure(figsize=(7.2, 4.4))
    plt.plot(data["GrainSize"], data["S_WS_TP"], marker="o", label=r"$S_{\mathrm{WS/TP}}=T_{\mathrm{TP}}/T_{\mathrm{WS}}$")
    plt.axhline(1.0, linestyle="--", linewidth=1.0, label="паритет")
    plt.xscale("log", base=2)
    plt.xlabel("Размер задачи grainSize, агентов")
    plt.ylabel(r"$S_{\mathrm{WS/TP}} = T_{\mathrm{TP}} / T_{\mathrm{WS}}$")
    plt.title(f"Лёгкая нагрузка: отношение TP/WS по grainSize (N={agents:,})".replace(",", " "))
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    savefig("light_ws_tp_relative_by_grain")


def plot_heavy_scaling_4_to_8(heavy: pd.DataFrame) -> None:
    """Масштабирование 4 -> 8 потоков в тяжёлой серии."""
    agents = 200_000
    grain = 1024
    iterations = 10_000

    plt.figure(figsize=(7.2, 4.4))
    for scheduler in ["ThreadPool", "WorkStealing"]:
        data = heavy[
            (heavy["Scheduler"] == scheduler)
            & (heavy["Agents"] == agents)
            & (heavy["GrainSize"] == grain)
            & (heavy["HeavyIterations"] == iterations)
        ]
        pivot = data.pivot_table(index="HeavyRatio", columns="Threads", values="AvgFrameMs", aggfunc="first").reset_index()
        pivot["HeavyRatioRounded"] = pivot["HeavyRatio"].round(2)
        pivot["Scale_4_to_8"] = pivot[4] / pivot[8]
        plt.plot(pivot["HeavyRatioRounded"], pivot["Scale_4_to_8"], marker="o", label=scheduler)

    plt.axhline(2.0, linestyle="--", linewidth=1.0, label="идеал 2x")
    plt.xlabel("Доля тяжёлых агентов HeavyRatio")
    plt.ylabel("Ускорение при переходе с 4 на 8 потоков")
    plt.title(f"Тяжёлая нагрузка: масштабирование 4→8 потоков, N={agents:,}".replace(",", " "))
    plt.grid(True, alpha=0.35)
    plt.legend()
    savefig("heavy_scaling_4_to_8")


def plot_heavy_avg_by_heavy_ratio(paired: pd.DataFrame) -> None:
    """Зависимость AvgFrameMs от HeavyRatio."""
    agents = 200_000
    threads = 8
    grain = 1024
    iterations = 10_000
    data = paired[
        (paired["Agents"] == agents)
        & (paired["Threads"] == threads)
        & (paired["GrainSize"] == grain)
        & (paired["HeavyIterations"] == iterations)
    ].sort_values("HeavyRatio")

    plt.figure(figsize=(7.2, 4.4))
    plt.plot(data["HeavyRatioRounded"], data["ThreadPool"], marker="o", label="ThreadPool")
    plt.plot(data["HeavyRatioRounded"], data["WorkStealing"], marker="o", label="WorkStealing")
    plt.xlabel("Доля тяжёлых агентов HeavyRatio")
    plt.ylabel("Среднее время кадра, мс")
    plt.title(f"Тяжёлая нагрузка: рост времени кадра, N={agents:,}, grainSize={grain}".replace(",", " "))
    plt.grid(True, alpha=0.35)
    plt.legend()
    savefig("heavy_avg_frame_by_heavy_ratio")


def plot_heavy_ws_tp_relative_speedup(paired: pd.DataFrame) -> None:
    """Относительное преимущество/проигрыш WorkStealing к ThreadPool."""
    agents = 200_000
    threads = 8
    grain = 1024
    iterations = 10_000
    data = paired[
        (paired["Agents"] == agents)
        & (paired["Threads"] == threads)
        & (paired["GrainSize"] == grain)
        & (paired["HeavyIterations"] == iterations)
    ].sort_values("HeavyRatio")

    plt.figure(figsize=(7.2, 4.4))
    plt.plot(data["HeavyRatioRounded"], data["S_WS_TP"], marker="o", label=r"$S_{\mathrm{WS/TP}}$")
    plt.axhline(1.0, linestyle="--", linewidth=1.0, label="паритет")
    plt.xlabel("Доля тяжёлых агентов HeavyRatio")
    plt.ylabel(r"$S_{\mathrm{WS/TP}} = T_{\mathrm{TP}} / T_{\mathrm{WS}}$")
    plt.title(f"Тяжёлая нагрузка: отношение TP/WS, N={agents:,}".replace(",", " "))
    plt.grid(True, alpha=0.35)
    plt.legend()
    savefig("heavy_ws_tp_relative_speedup")


def plot_heavy_min_max_stability(stability: pd.DataFrame) -> None:
    """Показывает диапазон MinFrameMs--MaxFrameMs для тяжёлой нагрузки."""
    agents = 200_000
    threads = 8
    grain = 1024
    iterations = 10_000
    data = stability[
        (stability["Agents"] == agents)
        & (stability["Threads"] == threads)
        & (stability["GrainSize"] == grain)
        & (stability["HeavyIterations"] == iterations)
    ].sort_values("HeavyRatio")

    plt.figure(figsize=(7.2, 4.4))
    for scheduler in ["ThreadPool", "WorkStealing"]:
        x = data["HeavyRatioRounded"]
        avg = data[f"AvgFrameMs_{scheduler}"]
        min_v = data[f"MinFrameMs_{scheduler}"]
        max_v = data[f"MaxFrameMs_{scheduler}"]
        plt.plot(x, avg, marker="o", label=f"{scheduler}: среднее")
        plt.fill_between(x, min_v, max_v, alpha=0.16, label=f"{scheduler}: Min--Max")

    plt.xlabel("Доля тяжёлых агентов HeavyRatio")
    plt.ylabel("Время кадра, мс")
    plt.title(f"Тяжёлая нагрузка: диапазон Min--Max, N={agents:,}, grainSize={grain}".replace(",", " "))
    plt.grid(True, alpha=0.35)
    plt.legend()
    savefig("heavy_min_max_stability")


def plot_heavy_relative_range(stability: pd.DataFrame) -> None:
    """Сравнивает относительный разброс (Max-Min)/Avg."""
    agents = 200_000
    threads = 8
    grain = 1024
    iterations = 10_000
    data = stability[
        (stability["Agents"] == agents)
        & (stability["Threads"] == threads)
        & (stability["GrainSize"] == grain)
        & (stability["HeavyIterations"] == iterations)
    ].sort_values("HeavyRatio")

    plt.figure(figsize=(7.2, 4.4))
    plt.plot(
        data["HeavyRatioRounded"],
        100.0 * data["RelativeRange_ThreadPool"],
        marker="o",
        label="ThreadPool",
    )
    plt.plot(
        data["HeavyRatioRounded"],
        100.0 * data["RelativeRange_WorkStealing"],
        marker="o",
        label="WorkStealing",
    )
    plt.xlabel("Доля тяжёлых агентов HeavyRatio")
    plt.ylabel(r"Относительный размах $(Max-Min)/Avg$, %")
    plt.title(f"Тяжёлая нагрузка: стабильность времени кадра, N={agents:,}".replace(",", " "))
    plt.grid(True, alpha=0.35)
    plt.legend()
    savefig("heavy_relative_range_by_heavy_ratio")


def plot_heavy_grain_effect(paired: pd.DataFrame) -> None:
    """Влияние grainSize при тяжёлой нагрузке."""
    agents = 200_000
    threads = 8
    ratio = 0.25
    iterations = 10_000
    data = paired[
        (paired["Agents"] == agents)
        & (paired["Threads"] == threads)
        & (paired["HeavyRatioRounded"] == ratio)
        & (paired["HeavyIterations"] == iterations)
    ].sort_values("GrainSize")

    plt.figure(figsize=(7.2, 4.4))
    plt.plot(data["GrainSize"], data["ThreadPool"], marker="o", label="ThreadPool")
    plt.plot(data["GrainSize"], data["WorkStealing"], marker="o", label="WorkStealing")
    plt.xscale("log", base=2)
    plt.xlabel("Размер задачи grainSize, агентов")
    plt.ylabel("Среднее время кадра, мс")
    plt.title(f"Тяжёлая нагрузка: влияние grainSize при HeavyRatio={ratio}")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    savefig("heavy_grain_effect")


def export_tables(light_paired: pd.DataFrame, heavy_paired: pd.DataFrame, heavy_stability: pd.DataFrame, light: pd.DataFrame, heavy: pd.DataFrame, base: pd.DataFrame) -> None:
    """Экспортирует краткие таблицы, используемые в тексте главы."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # Описание наборов данных
    datasets = pd.DataFrame([
        {
            "file": "benchmark_results.csv",
            "rows": len(base),
            "schedulers": ", ".join(sorted(base["Scheduler"].unique())),
            "agents": f'{base["Agents"].min()}–{base["Agents"].max()}',
            "threads": ", ".join(map(str, sorted(base["Threads"].unique()))),
            "grain_size": ", ".join(map(str, sorted(base["GrainSize"].unique()))),
            "heavy_ratio": ", ".join(map(lambda x: f"{x:.2g}", sorted(base["HeavyRatio"].unique()))),
            "heavy_iterations": ", ".join(map(str, sorted(base["HeavyIterations"].unique()))),
            "frames": ", ".join(map(str, sorted(base["Frames"].unique()))),
        },
        {
            "file": "benchmark_results_light_overhead.csv",
            "rows": len(light),
            "schedulers": ", ".join(sorted(light["Scheduler"].unique())),
            "agents": f'{light["Agents"].min()}–{light["Agents"].max()}',
            "threads": ", ".join(map(str, sorted(light["Threads"].unique()))),
            "grain_size": ", ".join(map(str, sorted(light["GrainSize"].unique()))),
            "heavy_ratio": ", ".join(map(lambda x: f"{x:.2g}", sorted(light["HeavyRatio"].unique()))),
            "heavy_iterations": ", ".join(map(str, sorted(light["HeavyIterations"].unique()))),
            "frames": ", ".join(map(str, sorted(light["Frames"].unique()))),
        },
        {
            "file": "benchmark_results_heavy_parallel.csv",
            "rows": len(heavy),
            "schedulers": ", ".join(sorted(heavy["Scheduler"].unique())),
            "agents": f'{heavy["Agents"].min()}–{heavy["Agents"].max()}',
            "threads": ", ".join(map(str, sorted(heavy["Threads"].unique()))),
            "grain_size": ", ".join(map(str, sorted(heavy["GrainSize"].unique()))),
            "heavy_ratio": ", ".join(map(lambda x: f"{x:.2g}", sorted(heavy["HeavyRatio"].unique()))),
            "heavy_iterations": ", ".join(map(str, sorted(heavy["HeavyIterations"].unique()))),
            "frames": ", ".join(map(str, sorted(heavy["Frames"].unique()))),
        },
    ])
    datasets.to_csv(TABLE_DIR / "datasets_summary.csv", index=False)

    # Лучшие парные конфигурации light: для каждого N выбирается минимум среди ThreadPool/WorkStealing,
    # а второй scheduler показывается при тех же Threads и GrainSize.
    light_records = []
    for agents, group in light_paired.groupby("Agents"):
        g = group.copy()
        g["BestParallelMs"] = g[["ThreadPool", "WorkStealing"]].min(axis=1)
        row = g.loc[g["BestParallelMs"].idxmin()]
        best_scheduler = "ThreadPool" if row["ThreadPool"] <= row["WorkStealing"] else "WorkStealing"
        light_records.append({
            "Agents": int(agents),
            "Threads": int(row["Threads"]),
            "GrainSize": int(row["GrainSize"]),
            "SequentialMs": row["Sequential"],
            "ThreadPoolMs": row["ThreadPool"],
            "WorkStealingMs": row["WorkStealing"],
            "BestScheduler": best_scheduler,
            "BestSpeedup": row["Sequential"] / row["BestParallelMs"],
            "S_WS_TP": row["ThreadPool"] / row["WorkStealing"],
        })
    pd.DataFrame(light_records).to_csv(TABLE_DIR / "light_best_paired_configurations.csv", index=False)

    # Лёгкая серия: детальный overhead при N=1e6, threads=8.
    light_overhead = light_paired[
        (light_paired["Agents"] == 1_000_000)
        & (light_paired["Threads"] == 8)
    ].sort_values("GrainSize").copy()
    light_overhead["ThreadPool_over_Seq"] = light_overhead["ThreadPool"] / light_overhead["Sequential"]
    light_overhead["WorkStealing_over_Seq"] = light_overhead["WorkStealing"] / light_overhead["Sequential"]
    light_overhead.to_csv(TABLE_DIR / "light_overhead_n1000000_threads8.csv", index=False)

    # Лучшие парные конфигурации heavy: для N=200000 по HeavyRatio и HeavyIterations.
    heavy_records = []
    for (ratio, iterations), group in heavy_paired[heavy_paired["Agents"] == 200_000].groupby(["HeavyRatioRounded", "HeavyIterations"]):
        g = group.copy()
        g["BestMs"] = g[["ThreadPool", "WorkStealing"]].min(axis=1)
        row = g.loc[g["BestMs"].idxmin()]
        best_scheduler = "ThreadPool" if row["ThreadPool"] <= row["WorkStealing"] else "WorkStealing"
        heavy_records.append({
            "HeavyRatio": ratio,
            "HeavyIterations": int(iterations),
            "Threads": int(row["Threads"]),
            "GrainSize": int(row["GrainSize"]),
            "ThreadPoolMs": row["ThreadPool"],
            "WorkStealingMs": row["WorkStealing"],
            "BestScheduler": best_scheduler,
            "BestMs": row["BestMs"],
            "S_WS_TP": row["ThreadPool"] / row["WorkStealing"],
        })
    pd.DataFrame(heavy_records).to_csv(TABLE_DIR / "heavy_best_paired_configurations_n200000.csv", index=False)

    # Показательные сценарии для сравнения ThreadPool и WorkStealing.
    comparison_rows = []

    light_cases = [
        (10_000, 8, 1, "light: мелкие задачи, мало агентов"),
        (1_000_000, 8, 1, "light: мелкие задачи, много агентов"),
        (1_000_000, 8, 1024, "light: рабочий grainSize"),
        (1_000_000, 8, 4096, "light: крупный grainSize"),
    ]
    for agents, threads, grain, label in light_cases:
        row = light_paired[
            (light_paired["Agents"] == agents)
            & (light_paired["Threads"] == threads)
            & (light_paired["GrainSize"] == grain)
        ].iloc[0]
        comparison_rows.append({
            "Scenario": label,
            "Agents": agents,
            "Threads": threads,
            "GrainSize": grain,
            "HeavyRatio": 0.0,
            "HeavyIterations": 0,
            "ThreadPoolMs": row["ThreadPool"],
            "WorkStealingMs": row["WorkStealing"],
            "S_WS_TP": row["S_WS_TP"],
        })

    heavy_cases = [
        (200_000, 8, 1024, 0.01, 10_000, "heavy: малая доля тяжёлых агентов"),
        (200_000, 8, 1024, 0.10, 10_000, "heavy: средняя доля тяжёлых агентов"),
        (200_000, 8, 1024, 0.25, 10_000, "heavy: высокая доля тяжёлых агентов"),
        (200_000, 8, 64, 0.25, 10_000, "heavy: высокая доля, мелкий grainSize"),
    ]
    for agents, threads, grain, ratio, iterations, label in heavy_cases:
        row = heavy_paired[
            (heavy_paired["Agents"] == agents)
            & (heavy_paired["Threads"] == threads)
            & (heavy_paired["GrainSize"] == grain)
            & (heavy_paired["HeavyRatioRounded"] == ratio)
            & (heavy_paired["HeavyIterations"] == iterations)
        ].iloc[0]
        comparison_rows.append({
            "Scenario": label,
            "Agents": agents,
            "Threads": threads,
            "GrainSize": grain,
            "HeavyRatio": ratio,
            "HeavyIterations": iterations,
            "ThreadPoolMs": row["ThreadPool"],
            "WorkStealingMs": row["WorkStealing"],
            "S_WS_TP": row["S_WS_TP"],
        })
    pd.DataFrame(comparison_rows).to_csv(TABLE_DIR / "scheduler_comparison_representative.csv", index=False)


    # Стабильность тяжёлой серии: Min/Avg/Max и относительный размах для фиксированного среза.
    stability_slice = heavy_stability[
        (heavy_stability["Agents"] == 200_000)
        & (heavy_stability["Threads"] == 8)
        & (heavy_stability["GrainSize"] == 1024)
        & (heavy_stability["HeavyIterations"] == 10_000)
    ].sort_values("HeavyRatioRounded").copy()
    stability_slice.to_csv(TABLE_DIR / "heavy_stability_n200000_threads8_grain1024_iter10000.csv", index=False)

    stability_summary = pd.DataFrame([
        {
            "Metric": "WorkStealing lower MaxFrameMs",
            "Count": int((heavy_stability["MaxFrameMs_WorkStealing"] < heavy_stability["MaxFrameMs_ThreadPool"]).sum()),
            "Total": int(len(heavy_stability)),
        },
        {
            "Metric": "WorkStealing lower absolute range",
            "Count": int((heavy_stability["RangeFrameMs_WorkStealing"] < heavy_stability["RangeFrameMs_ThreadPool"]).sum()),
            "Total": int(len(heavy_stability)),
        },
        {
            "Metric": "WorkStealing lower relative range",
            "Count": int((heavy_stability["RelativeRange_WorkStealing"] < heavy_stability["RelativeRange_ThreadPool"]).sum()),
            "Total": int(len(heavy_stability)),
        },
    ])
    stability_summary["Share"] = stability_summary["Count"] / stability_summary["Total"]
    stability_summary.to_csv(TABLE_DIR / "heavy_stability_summary.csv", index=False)


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    base = pd.read_csv(BASE_CSV)
    light = pd.read_csv(LIGHT_CSV)
    heavy = pd.read_csv(HEAVY_CSV)

    light_paired = prepare_light(light)
    heavy_paired = prepare_heavy(heavy)
    heavy_stability = prepare_heavy_stability(heavy)

    plot_light_frame_by_agents(light_paired)
    plot_light_grain_effect(light_paired)
    plot_light_speedup_by_agents(light_paired)
    plot_light_fine_grain_overhead(light_paired)
    plot_light_ws_tp_ratio(light_paired)

    plot_heavy_scaling_4_to_8(heavy)
    plot_heavy_avg_by_heavy_ratio(heavy_paired)
    plot_heavy_ws_tp_relative_speedup(heavy_paired)
    plot_heavy_min_max_stability(heavy_stability)
    plot_heavy_relative_range(heavy_stability)
    plot_heavy_grain_effect(heavy_paired)

    export_tables(light_paired, heavy_paired, heavy_stability, light, heavy, base)

    print(f"Figures saved to: {FIG_DIR}")
    print(f"Tables saved to: {TABLE_DIR}")


if __name__ == "__main__":
    main()
