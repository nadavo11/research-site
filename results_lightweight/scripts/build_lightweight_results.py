#!/usr/bin/env python3
"""Build a compact report-ready results bundle for DiffusionEdge RWTD LoRA runs."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
RUN_ROOT = SCRIPT_PATH.parents[2]
SUMMARY_DIR = RUN_ROOT / "summary"
MASTER_CSV = SUMMARY_DIR / "master_results.csv"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def clean_value(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    return value


def as_int_shot(value) -> Optional[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return int(value)


def domain_label(domain: str) -> str:
    labels = {
        "synthetic": "Synthetic",
        "bsds_wild": "Wild-BSDS",
    }
    return labels.get(domain, domain)


def load_master() -> pd.DataFrame:
    if not MASTER_CSV.exists():
        raise FileNotFoundError(f"Missing summary file: {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV)
    return df


def write_core_tables(df: pd.DataFrame, out_tables: Path) -> pd.DataFrame:
    ensure_dir(out_tables)

    core_columns = [
        "run_id",
        "status",
        "source_domain",
        "target_dataset",
        "lora_shots",
        "primary_metric",
        "primary_metric_value",
        "delta_from_zero_shot",
        "edge_ap",
        "edge_ods_f1",
        "edge_ois_f1",
        "metrics_source",
        "checkpoint_out",
    ]
    core_df = df.loc[:, [c for c in core_columns if c in df.columns]].copy()
    core_df = core_df.sort_values(["source_domain", "lora_shots", "run_id"], na_position="last")

    core_df.to_csv(out_tables / "core_results.csv", index=False)
    core_df.to_json(out_tables / "core_results.json", orient="records", indent=2)

    for name in ("master_results.csv", "master_results.json", "master_results.md"):
        copy_if_exists(SUMMARY_DIR / name, out_tables / name)

    return core_df


def detect_run_metric_files(run_dir: Path, row: pd.Series) -> Dict[str, Optional[Path]]:
    metrics_source = clean_value(row.get("metrics_source"))
    candidate_source = Path(metrics_source) if isinstance(metrics_source, str) and metrics_source else None
    source_name = candidate_source.name if candidate_source is not None else ""

    candidates = {
        "metrics_json": [
            run_dir / "eval" / "metrics.json",
            run_dir / "metrics.json",
            candidate_source if source_name == "metrics.json" else None,
        ],
        "eval_results_json": [
            run_dir / "eval" / "eval_results.json",
            candidate_source if source_name == "eval_results.json" else None,
        ],
        "performance_summary_json": [
            run_dir / "checkpoints" / "performance_summary.json",
            candidate_source if source_name == "performance_summary.json" else None,
        ],
    }

    resolved: Dict[str, Optional[Path]] = {}
    for key, opts in candidates.items():
        resolved[key] = next((p for p in opts if p is not None and p.exists()), None)

    edgeeval_dir = run_dir / "eval" / "edgeeval_json"
    resolved["edgeeval_dir"] = edgeeval_dir if edgeeval_dir.exists() else None
    return resolved


def copy_run_artifacts(df: pd.DataFrame, out_per_run: Path) -> Dict[str, Dict[str, List[str]]]:
    ensure_dir(out_per_run)
    copied: Dict[str, Dict[str, List[str]]] = {}

    for row in df.itertuples(index=False):
        run_id = row.run_id
        run_dir = RUN_ROOT / run_id
        if not run_dir.exists():
            continue

        row_series = pd.Series(row._asdict())
        files = detect_run_metric_files(run_dir, row_series)
        run_out = out_per_run / run_id
        ensure_dir(run_out)

        copied_paths: List[str] = []
        for artifact_name in ("metrics_json", "eval_results_json", "performance_summary_json"):
            src = files.get(artifact_name)
            if src is None:
                continue
            dst_name = src.name if artifact_name != "performance_summary_json" else "performance_summary.json"
            dst = run_out / dst_name
            if copy_if_exists(src, dst):
                copied_paths.append(str(dst.relative_to(out_per_run.parent)))

        edgeeval_dir = files.get("edgeeval_dir")
        if isinstance(edgeeval_dir, Path):
            edgeeval_out = run_out / "edgeeval"
            ensure_dir(edgeeval_out)
            for txt in ("eval_bdry.txt", "eval_bdry_thr.txt", "eval_bdry_img.txt"):
                src = edgeeval_dir / txt
                if copy_if_exists(src, edgeeval_out / txt):
                    copied_paths.append(str((edgeeval_out / txt).relative_to(out_per_run.parent)))

        if copied_paths:
            copied[run_id] = {"files": copied_paths}

    return copied


def load_edgeeval_thr(thr_path: Path) -> Optional[np.ndarray]:
    if not thr_path.exists():
        return None
    try:
        arr = np.loadtxt(thr_path)
    except Exception:
        return None
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 4:
        return None
    return arr


def load_edgeeval_img(img_path: Path) -> List[Tuple[str, float]]:
    if not img_path.exists():
        return []

    items: List[Tuple[str, float]] = []
    with img_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            sample_id = parts[0]
            f1 = float(parts[4])
            items.append((sample_id, f1))
    return items


def select_preview_ids(eval_bdry_img_path: Path, max_ids: int) -> List[str]:
    rows = load_edgeeval_img(eval_bdry_img_path)
    if not rows:
        return []

    rows_sorted = sorted(rows, key=lambda x: x[1])
    n = len(rows_sorted)
    idxs = [n - 1, int(0.75 * (n - 1)), int(0.50 * (n - 1)), int(0.25 * (n - 1)), 0]
    picks: List[str] = []
    for idx in idxs:
        sid = rows_sorted[idx][0]
        if sid not in picks:
            picks.append(sid)
        if len(picks) >= max_ids:
            break

    for sid, _ in rows_sorted:
        if sid not in picks:
            picks.append(sid)
        if len(picks) >= max_ids:
            break
    return picks


def style_plot(ax) -> None:
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)


def plot_metric_curves(df: pd.DataFrame, out_plots: Path) -> List[str]:
    ensure_dir(out_plots)
    plot_paths: List[str] = []

    plot_df = df[df["edge_ap"].notna()].copy()
    plot_df["lora_shots"] = plot_df["lora_shots"].astype(int)
    plot_df = plot_df.sort_values(["source_domain", "lora_shots"])

    if plot_df.empty:
        return plot_paths

    # AP vs shots
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for domain, group in plot_df.groupby("source_domain"):
        ax.plot(
            group["lora_shots"],
            group["edge_ap"],
            marker="o",
            linewidth=2.0,
            label=domain_label(domain),
        )
    ax.set_title("Edge AP vs LoRA Shots")
    ax.set_xlabel("LoRA shots")
    ax.set_ylabel("edge/AP")
    style_plot(ax)
    ax.legend(loc="best")
    ap_out = out_plots / "ap_vs_shots.png"
    fig.tight_layout()
    fig.savefig(ap_out, dpi=180)
    plt.close(fig)
    plot_paths.append(str(ap_out))

    # ODS and OIS F1 vs shots
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), sharex=True)
    metric_specs = [("edge_ods_f1", "edge/ODS_f1"), ("edge_ois_f1", "edge/OIS_f1")]
    for ax, (metric_col, metric_name) in zip(axes, metric_specs):
        for domain, group in plot_df.groupby("source_domain"):
            ax.plot(
                group["lora_shots"],
                group[metric_col],
                marker="o",
                linewidth=2.0,
                label=domain_label(domain),
            )
        ax.set_title(f"{metric_name} vs LoRA Shots")
        ax.set_xlabel("LoRA shots")
        ax.set_ylabel(metric_name)
        style_plot(ax)
    axes[0].legend(loc="best")
    f1_out = out_plots / "f1_vs_shots.png"
    fig.tight_layout()
    fig.savefig(f1_out, dpi=180)
    plt.close(fig)
    plot_paths.append(str(f1_out))

    # Delta vs zero-shot
    delta_df = df[df["delta_from_zero_shot"].notna()].copy()
    if not delta_df.empty:
        delta_df["lora_shots"] = delta_df["lora_shots"].astype(int)
        delta_df = delta_df.sort_values(["source_domain", "lora_shots"])

        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        for domain, group in delta_df.groupby("source_domain"):
            ax.plot(
                group["lora_shots"],
                group["delta_from_zero_shot"],
                marker="o",
                linewidth=2.0,
                label=domain_label(domain),
            )
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
        ax.set_title("Delta AP vs Zero-Shot Baseline")
        ax.set_xlabel("LoRA shots")
        ax.set_ylabel("delta_from_zero_shot")
        style_plot(ax)
        ax.legend(loc="best")
        delta_out = out_plots / "delta_vs_zero_shot.png"
        fig.tight_layout()
        fig.savefig(delta_out, dpi=180)
        plt.close(fig)
        plot_paths.append(str(delta_out))

    # Status overview
    status_tbl = (
        df.groupby(["source_domain", "status"]).size().reset_index(name="count").pivot(
            index="source_domain", columns="status", values="count"
        )
    ).fillna(0)
    if not status_tbl.empty:
        status_tbl = status_tbl.reindex(sorted(status_tbl.columns), axis=1)
        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        bottom = np.zeros(len(status_tbl.index))
        x = np.arange(len(status_tbl.index))
        for col in status_tbl.columns:
            vals = status_tbl[col].to_numpy()
            ax.bar(x, vals, bottom=bottom, label=col)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels([domain_label(d) for d in status_tbl.index], rotation=0)
        ax.set_ylabel("Run count")
        ax.set_title("Run Status Overview")
        style_plot(ax)
        ax.legend(loc="best")
        status_out = out_plots / "status_overview.png"
        fig.tight_layout()
        fig.savefig(status_out, dpi=180)
        plt.close(fig)
        plot_paths.append(str(status_out))

    return plot_paths


def plot_pr_curves(df: pd.DataFrame, out_plots: Path) -> List[str]:
    ensure_dir(out_plots)
    plot_paths: List[str] = []
    plot_df = df[df["edge_ap"].notna()].copy()
    if plot_df.empty:
        return plot_paths

    for domain, group in plot_df.groupby("source_domain"):
        group = group.sort_values("lora_shots")
        fig, ax = plt.subplots(figsize=(6.3, 5.3))
        plotted = 0
        for row in group.itertuples(index=False):
            run_dir = RUN_ROOT / row.run_id
            thr = run_dir / "eval" / "edgeeval_json" / "eval_bdry_thr.txt"
            arr = load_edgeeval_thr(thr)
            if arr is None:
                continue
            recall = arr[:, 1]
            precision = arr[:, 2]
            shot = as_int_shot(row.lora_shots)
            ap_val = row.edge_ap
            label = f"{shot}-shot (AP={ap_val:.3f})" if shot is not None else f"AP={ap_val:.3f}"
            ax.plot(recall, precision, linewidth=1.8, label=label)
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            continue

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR Curves ({domain_label(domain)})")
        style_plot(ax)
        ax.legend(loc="lower left", fontsize=8)
        out_path = out_plots / f"pr_curve_{domain}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        plot_paths.append(str(out_path))

    return plot_paths


def build_qualitative(
    df: pd.DataFrame, out_qual: Path, previews_per_domain: int
) -> Dict[str, Dict[str, Sequence[str]]]:
    ensure_dir(out_qual)
    preview_out = out_qual / "previews"
    ensure_dir(preview_out)

    manifest: Dict[str, Dict[str, Sequence[str]]] = {}
    metric_df = df[df["edge_ap"].notna()].copy()
    if metric_df.empty:
        return manifest

    for domain, group in metric_df.groupby("source_domain"):
        group = group.sort_values("lora_shots")

        zero_shot = group[group["lora_shots"] == 0]
        ref_row = zero_shot.iloc[0] if not zero_shot.empty else group.iloc[0]
        ref_run = RUN_ROOT / ref_row["run_id"]
        ref_img_stats = ref_run / "eval" / "edgeeval_json" / "eval_bdry_img.txt"
        selected_ids = select_preview_ids(ref_img_stats, previews_per_domain)
        if not selected_ids:
            continue

        rows = len(selected_ids)
        cols = len(group)
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(max(3.4 * cols, 8.0), max(1.3 * rows, 4.8)),
            squeeze=False,
        )

        included_files: List[str] = []
        run_ids: List[str] = []

        for col, run_row in enumerate(group.itertuples(index=False)):
            run_ids.append(run_row.run_id)
            shot = as_int_shot(run_row.lora_shots)
            shot_label = f"{shot}-shot" if shot is not None else "N/A-shot"
            ap_val = run_row.edge_ap
            metric_txt = f"AP={ap_val:.3f}" if ap_val == ap_val else "AP=n/a"
            axes[0, col].set_title(f"{shot_label}\n{metric_txt}", fontsize=8)

            for row_idx, sample_id in enumerate(selected_ids):
                ax = axes[row_idx, col]
                ax.axis("off")
                src = RUN_ROOT / run_row.run_id / "eval" / "previews" / f"{sample_id}_preview.png"
                if src.exists():
                    img = mpimg.imread(src)
                    ax.imshow(img)
                    dst = preview_out / run_row.run_id / f"{sample_id}_preview.png"
                    if copy_if_exists(src, dst):
                        included_files.append(str(dst.relative_to(out_qual)))
                else:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8)
                if col == 0:
                    ax.set_ylabel(f"id {sample_id}", fontsize=8)

        fig.suptitle(f"{domain_label(domain)} Qualitative Comparison", fontsize=11)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        grid_out = out_qual / f"{domain}_preview_grid.png"
        fig.savefig(grid_out, dpi=160)
        plt.close(fig)

        manifest[domain] = {
            "selected_preview_ids": selected_ids,
            "runs": run_ids,
            "copied_preview_files": sorted(set(included_files)),
            "grid_image": str(grid_out.relative_to(out_qual.parent)),
        }

    with (out_qual / "selected_preview_ids.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def write_report_md(
    df: pd.DataFrame,
    plots: Sequence[str],
    qualitative_manifest: Dict[str, Dict[str, Sequence[str]]],
    out_root: Path,
) -> None:
    report_path = out_root / "report.md"
    metric_df = df[df["edge_ap"].notna()].copy()
    metric_df = metric_df.sort_values(["source_domain", "lora_shots"])

    best_rows = (
        metric_df.sort_values("edge_ap", ascending=False).groupby("source_domain", as_index=False).head(1)
    )

    lines: List[str] = []
    lines.append("# Lightweight Report Pack")
    lines.append("")
    lines.append(f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Source run root: `{RUN_ROOT}`")
    lines.append(f"- Total runs in master table: {len(df)}")
    lines.append(f"- Runs with AP metric: {len(metric_df)}")
    lines.append("")
    lines.append("## Best AP per domain")
    lines.append("")
    for row in best_rows.itertuples(index=False):
        shot = as_int_shot(row.lora_shots)
        lines.append(
            f"- `{row.run_id}` ({domain_label(row.source_domain)}): AP={row.edge_ap:.6f}, shots={shot}, status={row.status}"
        )
    lines.append("")
    lines.append("## Included plots")
    lines.append("")
    for p in sorted(plots):
        rel = Path(p).relative_to(out_root)
        lines.append(f"- `{rel}`")
    lines.append("")
    lines.append("## Qualitative selections")
    lines.append("")
    for domain, payload in qualitative_manifest.items():
        ids = ", ".join(payload.get("selected_preview_ids", []))
        runs = ", ".join(payload.get("runs", []))
        lines.append(f"- {domain_label(domain)}:")
        lines.append(f"  - Preview ids: `{ids}`")
        lines.append(f"  - Runs: `{runs}`")
        lines.append(f"  - Grid: `{payload.get('grid_image', 'n/a')}`")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_sizes(path: Path) -> Dict[str, float]:
    total_bytes = 0
    file_count = 0
    for file in path.rglob("*"):
        if file.is_file():
            file_count += 1
            total_bytes += file.stat().st_size
    return {
        "file_count": file_count,
        "total_size_mb": round(total_bytes / (1024 * 1024), 3),
    }


def write_manifest(
    df: pd.DataFrame,
    copied_run_artifacts: Dict[str, Dict[str, List[str]]],
    plots: Sequence[str],
    pr_plots: Sequence[str],
    qualitative_manifest: Dict[str, Dict[str, Sequence[str]]],
    out_root: Path,
) -> None:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(RUN_ROOT),
        "rows_in_master_results": int(len(df)),
        "rows_with_metrics": int(df["edge_ap"].notna().sum()),
        "plots": [str(Path(p).relative_to(out_root)) for p in sorted(plots)],
        "pr_plots": [str(Path(p).relative_to(out_root)) for p in sorted(pr_plots)],
        "copied_run_artifacts": copied_run_artifacts,
        "qualitative": qualitative_manifest,
        "bundle_summary": summarize_sizes(out_root),
    }
    with (out_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUN_ROOT / "results_lightweight",
        help="Destination directory for the lightweight report bundle.",
    )
    parser.add_argument(
        "--previews-per-domain",
        type=int,
        default=5,
        help="How many preview sample IDs to include per domain qualitative grid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = args.output_dir.resolve()
    ensure_dir(out_root)

    for dirname in ("tables", "per_run", "plots", "qualitative"):
        dirpath = out_root / dirname
        if dirpath.exists():
            shutil.rmtree(dirpath)
    for filename in ("manifest.json", "report.md"):
        filepath = out_root / filename
        if filepath.exists():
            filepath.unlink()

    out_tables = out_root / "tables"
    out_plots = out_root / "plots"
    out_per_run = out_root / "per_run"
    out_qual = out_root / "qualitative"

    df = load_master()
    write_core_tables(df, out_tables)
    copied_run_artifacts = copy_run_artifacts(df, out_per_run)
    plots = plot_metric_curves(df, out_plots)
    pr_plots = plot_pr_curves(df, out_plots)
    qualitative_manifest = build_qualitative(df, out_qual, args.previews_per_domain)
    write_report_md(df, [*plots, *pr_plots], qualitative_manifest, out_root)
    write_manifest(df, copied_run_artifacts, plots, pr_plots, qualitative_manifest, out_root)

    print(f"Lightweight report bundle created at: {out_root}")
    summary = summarize_sizes(out_root)
    print(
        f"Bundle size: {summary['total_size_mb']} MB across {summary['file_count']} files "
        f"(previews_per_domain={args.previews_per_domain})"
    )


if __name__ == "__main__":
    main()
