#!/usr/bin/env python3
"""Build the SAM2-vs-SAM3 frozen-feature clustering report bundle."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import date
from html import escape
from pathlib import Path
from textwrap import dedent

from PIL import Image


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "experiments" / "Coarse Feature Clustering"
DEST_DIR = ROOT / "site" / "experiments" / "sam2-vs-sam3-frozen-feature-clustering"

PAGE_TITLE = "SAM2 vs SAM3 Frozen Feature Clustering for Binary Texture Partitioning"
PAGE_SUBTITLE = "Experiment Report"
PAGE_DATE = date.today().isoformat()
PAGE_DESCRIPTION = (
    "Same-flavor frozen-feature binary partitioning comparisons across RWTD, CAID, and STLD. "
    "The cleaner parity view is coarse-only first, flip-averaged second: SAM2 leads every published "
    "coarse-only row, keeps the flip-avg lead on STLD and CAID, and RWTD flip-avg remains the lone SAM3 holdout. "
    "On STLD, aggregate mIoU stays essentially flat around 0.5 and is better treated as a caution metric than as the headline signal."
)

PRIMARY_METRICS = ("eval_miou", "eval_ari")
CAUTION_METRIC = "miou_agg"

DATASET_ORDER = ["rwtd", "caid", "stld"]
DATASET_LABELS = {
    "rwtd": "RWTD",
    "caid": "CAID",
    "stld": "STLD",
}

FLAVOR_ORDER = ["coarse_only", "flip_averaged"]
FLAVOR_META = {
    "coarse_only": {
        "label": "Coarse Only",
        "section_title": "Coarse-Only Model Comparison",
        "summary": "Direct same-flavor parity on the original coarse-only clustering route.",
        "csv_name": "coarse_only_model_parity.csv",
        "dataset_takeaways": {
            "rwtd": "SAM2 wins coarse-only parity cleanly on RWTD.",
            "caid": "CAID shows a clear SAM2 lead in coarse-only parity.",
            "stld": "This is the largest same-flavor margin on the page.",
        },
    },
    "flip_averaged": {
        "label": "Flip Averaged",
        "section_title": "Flip-Averaged Model Comparison",
        "summary": "Same-flavor parity after the flip-averaged coarse clustering variant.",
        "csv_name": "flip_averaged_model_parity.csv",
        "dataset_takeaways": {
            "rwtd": "RWTD flip-avg remains the one place where SAM3 still holds a small edge.",
            "caid": "CAID flip-avg is close, but SAM2 still edges ahead.",
            "stld": "SAM2 keeps a clear flip-avg lead on STLD.",
        },
    },
}

MODEL_META = {
    "sam2": {
        "label": "SAM2",
        "model_id": "facebook/sam2-hiera-small",
        "color": "#005f73",
    },
    "sam3": {
        "label": "SAM3",
        "model_id": "facebook/sam3",
        "color": "#ca6702",
    },
}


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    dataset: str
    dataset_label: str
    split_label: str
    source_dir: str
    flavor: str
    model_key: str
    display_label: str
    short_label: str


RUN_SPECS = [
    RunSpec(
        run_id="rwtd_sam2_coarse",
        dataset="rwtd",
        dataset_label="aviadcohz/RWTD",
        split_label="test",
        source_dir="rwtd/sam2_vanilla_cfc",
        flavor="coarse_only",
        model_key="sam2",
        display_label="RWTD · SAM2 coarse_only",
        short_label="SAM2 coarse",
    ),
    RunSpec(
        run_id="rwtd_sam3_coarse",
        dataset="rwtd",
        dataset_label="aviadcohz/RWTD",
        split_label="test",
        source_dir="rwtd/default",
        flavor="coarse_only",
        model_key="sam3",
        display_label="RWTD · SAM3 coarse_only",
        short_label="SAM3 coarse",
    ),
    RunSpec(
        run_id="rwtd_sam2_flip",
        dataset="rwtd",
        dataset_label="aviadcohz/RWTD",
        split_label="test",
        source_dir="rwtd/sam2_flipavg_cfc",
        flavor="flip_averaged",
        model_key="sam2",
        display_label="RWTD · SAM2 flip_avg",
        short_label="SAM2 flip",
    ),
    RunSpec(
        run_id="rwtd_sam3_flip",
        dataset="rwtd",
        dataset_label="aviadcohz/RWTD",
        split_label="test",
        source_dir="rwtd/flip_averaged",
        flavor="flip_averaged",
        model_key="sam3",
        display_label="RWTD · SAM3 flip_avg",
        short_label="SAM3 flip",
    ),
    RunSpec(
        run_id="caid_sam2_coarse",
        dataset="caid",
        dataset_label="architexture:caid",
        split_label="caid",
        source_dir="caid/sam2_vanilla_cfc",
        flavor="coarse_only",
        model_key="sam2",
        display_label="CAID · SAM2 coarse_only",
        short_label="SAM2 coarse",
    ),
    RunSpec(
        run_id="caid_sam3_coarse",
        dataset="caid",
        dataset_label="architexture:caid",
        split_label="caid",
        source_dir="caid/default",
        flavor="coarse_only",
        model_key="sam3",
        display_label="CAID · SAM3 coarse_only",
        short_label="SAM3 coarse",
    ),
    RunSpec(
        run_id="caid_sam2_flip",
        dataset="caid",
        dataset_label="architexture:caid",
        split_label="caid",
        source_dir="caid/sam2_flipavg_cfc",
        flavor="flip_averaged",
        model_key="sam2",
        display_label="CAID · SAM2 flip_avg",
        short_label="SAM2 flip",
    ),
    RunSpec(
        run_id="caid_sam3_flip",
        dataset="caid",
        dataset_label="architexture:caid",
        split_label="caid",
        source_dir="caid/flip_averaged",
        flavor="flip_averaged",
        model_key="sam3",
        display_label="CAID · SAM3 flip_avg",
        short_label="SAM3 flip",
    ),
    RunSpec(
        run_id="stld_sam2_coarse",
        dataset="stld",
        dataset_label="architexture:stld",
        split_label="benchmark",
        source_dir="stld/sam2_vanilla_cfc",
        flavor="coarse_only",
        model_key="sam2",
        display_label="STLD · SAM2 coarse_only",
        short_label="SAM2 coarse",
    ),
    RunSpec(
        run_id="stld_sam3_coarse",
        dataset="stld",
        dataset_label="architexture:stld",
        split_label="benchmark",
        source_dir="stld/default",
        flavor="coarse_only",
        model_key="sam3",
        display_label="STLD · SAM3 coarse_only",
        short_label="SAM3 coarse",
    ),
    RunSpec(
        run_id="stld_sam2_flip",
        dataset="stld",
        dataset_label="architexture:stld",
        split_label="benchmark",
        source_dir="stld/sam2_flipavg_cfc",
        flavor="flip_averaged",
        model_key="sam2",
        display_label="STLD · SAM2 flip_avg",
        short_label="SAM2 flip",
    ),
    RunSpec(
        run_id="stld_sam3_flip",
        dataset="stld",
        dataset_label="architexture:stld",
        split_label="benchmark",
        source_dir="stld/flip_averaged",
        flavor="flip_averaged",
        model_key="sam3",
        display_label="STLD · SAM3 flip_avg",
        short_label="SAM3 flip",
    ),
]

RUN_SPEC_BY_ID = {spec.run_id: spec for spec in RUN_SPECS}
RUN_LOOKUP = {(spec.dataset, spec.model_key, spec.flavor): spec.run_id for spec in RUN_SPECS}

MAIN_STORIES = [
    {
        "story_id": "rwtd_coarse_parity",
        "dataset": "rwtd",
        "dataset_label": "RWTD",
        "sample_id": "240",
        "title": "RWTD: SAM2 Wins the Coarse-Only Parity Check",
        "summary": (
            "On this tile-and-grout crop, the same-flavor coarse-only comparison is visually clear. "
            "SAM2 coarse tracks the diagonal split cleanly, while SAM3 coarse breaks the scene into broad horizontal bands."
        ),
        "variant_order": ["rwtd_sam3_coarse", "rwtd_sam2_coarse"],
        "tag": "coarse parity",
        "show_on_index": True,
    },
    {
        "story_id": "stld_coarse_parity",
        "dataset": "stld",
        "dataset_label": "STLD",
        "sample_id": "66",
        "title": "STLD: Coarse-Only Still Favors SAM2 Decisively",
        "summary": (
            "This compact synthetic foreground is representative of the coarse-only gap on STLD. "
            "SAM2 coarse recovers the object cleanly, while SAM3 coarse overfills a much larger false-positive region."
        ),
        "variant_order": ["stld_sam3_coarse", "stld_sam2_coarse"],
        "tag": "coarse parity",
        "show_on_index": True,
    },
    {
        "story_id": "stld_flip_parity",
        "dataset": "stld",
        "dataset_label": "STLD",
        "sample_id": "41",
        "title": "STLD: Flip-Averaged Parity Still Favors SAM2",
        "summary": (
            "The thin oblique strip is harder, but the same-flavor flip-avg comparison still separates the models. "
            "SAM2 flip captures far more of the structure, while SAM3 flip nearly collapses the signal."
        ),
        "variant_order": ["stld_sam3_flip", "stld_sam2_flip"],
        "tag": "flip parity",
        "show_on_index": True,
    },
    {
        "story_id": "rwtd_flip_holdout",
        "dataset": "rwtd",
        "dataset_label": "RWTD",
        "sample_id": "219",
        "title": "RWTD: Flip-Averaged Is the Main Exception",
        "summary": (
            "RWTD flip-avg is the one same-flavor slice where SAM3 keeps a slight aggregate lead. "
            "This crop is included as a useful counterexample rather than hidden by the summary."
        ),
        "variant_order": ["rwtd_sam2_flip", "rwtd_sam3_flip"],
        "tag": "rwtd holdout",
        "show_on_index": False,
    },
]

REQUIRED_RUN_FILES = [
    "config.json",
    "experiment_terms.md",
    "summary.json",
    "summary.md",
    "per_sample_metrics.csv",
    "visuals_manifest.jsonl",
]


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path.relative_to(ROOT)}")
    return path


def load_json(path: Path) -> dict:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def leader_model_from_scores(sam2_miou: float, sam2_ari: float, sam3_miou: float, sam3_ari: float) -> str:
    if sam2_miou > sam3_miou:
        return "sam2"
    if sam3_miou > sam2_miou:
        return "sam3"
    if sam2_ari > sam3_ari:
        return "sam2"
    if sam3_ari > sam2_ari:
        return "sam3"
    return "tie"


def write_thumbnail(source_path: Path, dest_path: Path, width: int = 780, quality: int = 84) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        image = image.convert("RGB")
        if image.width > width:
            scale = width / image.width
            image = image.resize((width, max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
        image.save(dest_path, format="WEBP", quality=quality, method=6)


def load_visual_rows(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    with require_file(run_dir / "visuals_manifest.jsonl").open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            visual_rel = Path(payload["visual_path"])
            visual_path = require_file(run_dir / visual_rel)
            rows.append(
                {
                    "crop_name": str(payload.get("crop_name") or visual_rel.stem),
                    "sample_index": int(payload.get("sample_index", len(rows))),
                    "eval_miou": float(payload.get("eval_miou", payload.get("miou", 0.0))),
                    "eval_ari": float(payload.get("eval_ari", payload.get("ari", 0.0))),
                    "miou_agg": float(payload.get("miou_agg", 0.0)),
                    "caption": payload.get("caption", ""),
                    "source_visual_path": visual_path,
                }
            )
    rows.sort(key=lambda row: row["sample_index"])
    return rows


def load_run(spec: RunSpec) -> dict:
    run_dir = SOURCE_ROOT / spec.source_dir
    for file_name in REQUIRED_RUN_FILES:
        require_file(run_dir / file_name)

    summary = load_json(run_dir / "summary.json")
    metrics = summary["mean_metrics"]
    rows = load_visual_rows(run_dir)
    sample_lookup = {row["crop_name"]: row for row in rows}
    model_id = summary.get("model_id") or MODEL_META[spec.model_key]["model_id"]

    return {
        "run_id": spec.run_id,
        "dataset": spec.dataset,
        "dataset_slug": spec.dataset,
        "dataset_label": DATASET_LABELS[spec.dataset],
        "dataset_id": summary.get("dataset_id") or spec.dataset_label,
        "split_label": summary.get("split") or spec.split_label,
        "flavor": spec.flavor,
        "flavor_label": FLAVOR_META[spec.flavor]["label"],
        "model_key": spec.model_key,
        "model_label": MODEL_META[spec.model_key]["label"],
        "model_id": model_id,
        "display_label": spec.display_label,
        "short_label": spec.short_label,
        "exact_variant": summary.get("variant") or "",
        "num_samples": int(summary.get("num_evaluated_samples") or 0),
        "eval_miou": float(metrics.get("eval_miou", metrics.get("miou", 0.0))),
        "eval_ari": float(metrics.get("eval_ari", metrics.get("ari", 0.0))),
        "miou_agg": float(metrics.get("miou_agg", 0.0)),
        "source_dir": run_dir,
        "sample_lookup": sample_lookup,
    }


def enforce_alignment(runs: dict[str, dict]) -> None:
    for dataset in DATASET_ORDER:
        dataset_runs = [run for run in runs.values() if run["dataset"] == dataset]
        baseline = set(dataset_runs[0]["sample_lookup"])
        mismatches = []
        for run in dataset_runs[1:]:
            current = set(run["sample_lookup"])
            if current != baseline:
                mismatches.append(f"{run['run_id']} ({len(baseline.symmetric_difference(current))} mismatched ids)")
        if mismatches:
            raise RuntimeError(f"Sample alignment mismatch for {dataset}: {', '.join(mismatches)}")


def build_comparison_rows(runs: dict[str, dict]) -> dict[str, list[dict]]:
    comparisons: dict[str, list[dict]] = {}
    for flavor in FLAVOR_ORDER:
        rows: list[dict] = []
        for dataset in DATASET_ORDER:
            sam2 = runs[RUN_LOOKUP[(dataset, "sam2", flavor)]]
            sam3 = runs[RUN_LOOKUP[(dataset, "sam3", flavor)]]
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "dataset_id": sam2["dataset_id"],
                    "num_samples": sam2["num_samples"],
                    "flavor": flavor,
                    "flavor_label": FLAVOR_META[flavor]["label"],
                    "sam2_run_id": sam2["run_id"],
                    "sam3_run_id": sam3["run_id"],
                    "sam2_miou": sam2["eval_miou"],
                    "sam2_ari": sam2["eval_ari"],
                    "sam2_agg": sam2["miou_agg"],
                    "sam3_miou": sam3["eval_miou"],
                    "sam3_ari": sam3["eval_ari"],
                    "sam3_agg": sam3["miou_agg"],
                    "delta_miou": sam2["eval_miou"] - sam3["eval_miou"],
                    "delta_ari": sam2["eval_ari"] - sam3["eval_ari"],
                    "leader_model": leader_model_from_scores(sam2["eval_miou"], sam2["eval_ari"], sam3["eval_miou"], sam3["eval_ari"]),
                    "takeaway": FLAVOR_META[flavor]["dataset_takeaways"][dataset],
                }
            )
        comparisons[flavor] = rows
    return comparisons


def build_story_blocks(runs: dict[str, dict]) -> list[dict]:
    blocks: list[dict] = []
    for spec in MAIN_STORIES:
        images = []
        for run_id in spec["variant_order"]:
            run = runs[run_id]
            row = run["sample_lookup"].get(spec["sample_id"])
            if row is None:
                raise RuntimeError(f"Missing sample {spec['sample_id']} in {run_id}")
            asset_name = f"{spec['story_id']}__{run_id}.webp"
            asset_rel = Path("assets") / "gallery" / asset_name
            write_thumbnail(row["source_visual_path"], DEST_DIR / asset_rel)
            images.append(
                {
                    "run_id": run_id,
                    "label": run["short_label"],
                    "asset_rel": asset_rel.as_posix(),
                    "eval_miou": row["eval_miou"],
                    "eval_ari": row["eval_ari"],
                }
            )
        blocks.append({**spec, "images": images})
    return blocks


def render_grouped_parity_chart(title: str, subtitle: str, comparison_rows: dict[str, list[dict]], metric_key: str) -> str:
    groups = [row for flavor in FLAVOR_ORDER for row in comparison_rows[flavor]]
    width = 1120
    height = 500
    top = 96
    left = 70
    right = 28
    bottom = 130
    chart_height = height - top - bottom
    chart_width = width - left - right
    bar_width = 58
    intra_gap = 12
    group_gap = 34
    group_width = bar_width * 2 + intra_gap
    total_width = len(groups) * group_width + (len(groups) - 1) * group_gap
    start_x = left + max(0, (chart_width - total_width) / 2)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        '<rect width="100%" height="100%" fill="#ffffff" rx="24" />',
        f'<text x="28" y="40" font-family="Space Grotesk, sans-serif" font-size="24" font-weight="700" fill="#122033">{escape(title)}</text>',
        f'<text x="28" y="64" font-family="Space Grotesk, sans-serif" font-size="13" fill="#4c5d73">{escape(subtitle)}</text>',
        f'<rect x="{width - 190}" y="24" width="14" height="14" rx="4" fill="{MODEL_META["sam2"]["color"]}" />',
        f'<text x="{width - 168}" y="36" font-family="Space Grotesk, sans-serif" font-size="12" fill="#122033">SAM2</text>',
        f'<rect x="{width - 110}" y="24" width="14" height="14" rx="4" fill="{MODEL_META["sam3"]["color"]}" />',
        f'<text x="{width - 88}" y="36" font-family="Space Grotesk, sans-serif" font-size="12" fill="#122033">SAM3</text>',
    ]

    for tick in [0.0, 0.25, 0.50, 0.75, 1.0]:
        y = top + chart_height * (1.0 - tick)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e3ebf2" stroke-width="1" />')
        lines.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Space Grotesk, sans-serif" font-size="11" fill="#4c5d73">{tick:.2f}</text>'
        )

    cursor_x = start_x
    for row in groups:
        group_center = cursor_x + group_width / 2
        for model_index, model_key in enumerate(["sam2", "sam3"]):
            value = row[f"{model_key}_{metric_key}"]
            bar_height = chart_height * max(0.0, min(1.0, value))
            x = cursor_x + model_index * (bar_width + intra_gap)
            y = top + chart_height - bar_height
            lines.extend(
                [
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" rx="10" fill="{MODEL_META[model_key]["color"]}" />',
                    f'<text x="{x + bar_width / 2:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-family="Space Grotesk, sans-serif" font-size="11" font-weight="700" fill="#122033">{value:.3f}</text>',
                ]
            )
        lines.extend(
            [
                f'<text x="{group_center:.1f}" y="{height - 54}" text-anchor="middle" font-family="Outfit, sans-serif" font-size="14" font-weight="700" fill="#005f73">{escape(row["dataset_label"])}</text>',
                f'<text x="{group_center:.1f}" y="{height - 36}" text-anchor="middle" font-family="Space Grotesk, sans-serif" font-size="11" fill="#4c5d73">{escape(FLAVOR_META[row["flavor"]]["label"])}</text>',
            ]
        )
        cursor_x += group_width + group_gap

    lines.append("</svg>")
    return "\n".join(lines)


def write_plot_assets(comparison_rows: dict[str, list[dict]]) -> dict[str, str]:
    plots = {
        "eval_miou_parity.svg": render_grouped_parity_chart(
            "Same-Flavor eval_mIoU",
            "Each group is a dataset+flavor parity check with SAM2 and SAM3 side by side.",
            comparison_rows,
            "miou",
        ),
        "eval_ari_parity.svg": render_grouped_parity_chart(
            "Same-Flavor eval_ARI",
            "ARI tracks the same dataset-by-dataset story as mIoU, especially in coarse-only parity.",
            comparison_rows,
            "ari",
        ),
    }
    for file_name, content in plots.items():
        write_text(DEST_DIR / "assets" / "plots" / file_name, content)
    return {name: f"assets/plots/{name}" for name in plots}


def write_table_csvs(comparison_rows: dict[str, list[dict]]) -> dict[str, str]:
    tables_dir = DEST_DIR / "assets" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    fields = [
        "dataset_label",
        "dataset_id",
        "num_samples",
        "sam2_miou",
        "sam2_ari",
        "sam3_miou",
        "sam3_ari",
        "delta_miou",
        "delta_ari",
        "leader_model",
        "takeaway",
    ]
    for flavor, rows in comparison_rows.items():
        path = tables_dir / FLAVOR_META[flavor]["csv_name"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows([{field: row[field] for field in fields} for row in rows])
        paths[flavor] = path.relative_to(DEST_DIR).as_posix()
    return paths


def render_model_cell(miou: float, ari: float) -> str:
    return dedent(
        f"""\
        <strong>{miou:.3f}</strong>
        <div class="mini-note">ARI {ari:.3f}</div>
        """
    ).strip()


def render_comparison_table(rows: list[dict]) -> str:
    body_rows = []
    for row in rows:
        leader_label = "SAM2" if row["leader_model"] == "sam2" else "SAM3" if row["leader_model"] == "sam3" else "Tie"
        leader_class = "good" if row["leader_model"] == "sam2" else "warn" if row["leader_model"] == "sam3" else ""
        body_rows.append(
            dedent(
                f"""\
                <tr>
                  <td><strong>{escape(row['dataset_label'])}</strong></td>
                  <td>{render_model_cell(row['sam2_miou'], row['sam2_ari'])}</td>
                  <td>{render_model_cell(row['sam3_miou'], row['sam3_ari'])}</td>
                  <td><span class="delta-pill {'good' if row['delta_miou'] >= 0 else 'bad'}">{signed(row['delta_miou'], 3)}</span></td>
                  <td><span class="delta-pill {'good' if row['delta_ari'] >= 0 else 'bad'}">{signed(row['delta_ari'], 3)}</span></td>
                  <td><span class="tag {leader_class}">{leader_label}</span></td>
                </tr>
                """
            ).strip()
        )
    return "\n".join(body_rows)


def render_story_blocks(blocks: list[dict], *, index_only: bool) -> str:
    rendered = []
    for block in blocks:
        if index_only and not block["show_on_index"]:
            continue
        figures = []
        for image in block["images"]:
            figures.append(
                dedent(
                    f"""\
                    <figure>
                      <img src="{escape(image['asset_rel'])}" alt="{escape(block['title'])} — {escape(image['label'])}" data-zoom-src="{escape(image['asset_rel'])}" />
                      <figcaption>
                        <strong>{escape(image['label'])}</strong>
                        <div class="mini-note">mIoU {image['eval_miou']:.4f} · ARI {image['eval_ari']:.4f}</div>
                      </figcaption>
                    </figure>
                    """
                ).strip()
            )
        rendered.append(
            dedent(
                f"""\
                <article class="story-block">
                  <div class="story-head">
                    <span class="tag good">{escape(block['tag'])}</span>
                    <span class="pill">{escape(block['dataset_label'])} · crop {escape(block['sample_id'])}</span>
                  </div>
                  <h3>{escape(block['title'])}</h3>
                  <p class="story-summary">{escape(block['summary'])}</p>
                  <div class="story-grid">
                    {' '.join(figures)}
                  </div>
                </article>
                """
            ).strip()
        )
    return "\n".join(rendered)


def build_metrics_payload(runs: dict[str, dict], comparison_rows: dict[str, list[dict]]) -> dict:
    stld_agg_values = [
        runs["stld_sam2_coarse"]["miou_agg"],
        runs["stld_sam3_coarse"]["miou_agg"],
        runs["stld_sam2_flip"]["miou_agg"],
        runs["stld_sam3_flip"]["miou_agg"],
    ]
    all_rows = [row for flavor in FLAVOR_ORDER for row in comparison_rows[flavor]]
    largest_gain = max(all_rows, key=lambda row: (row["delta_miou"], row["delta_ari"]))
    rwtd_flip = next(row for row in comparison_rows["flip_averaged"] if row["dataset"] == "rwtd")
    return {
        "status": "reviewed",
        "page_title": PAGE_TITLE,
        "primary_metrics": list(PRIMARY_METRICS),
        "caution_metric": CAUTION_METRIC,
        "comparisons": comparison_rows,
        "headline": {
            "coarse_sam2_leads": sum(1 for row in comparison_rows["coarse_only"] if row["leader_model"] == "sam2"),
            "flip_sam2_leads": sum(1 for row in comparison_rows["flip_averaged"] if row["leader_model"] == "sam2"),
            "largest_gain": {
                "dataset": largest_gain["dataset_label"],
                "flavor": largest_gain["flavor"],
                "delta_miou": largest_gain["delta_miou"],
                "delta_ari": largest_gain["delta_ari"],
            },
            "rwtd_flip_exception": {
                "delta_miou": rwtd_flip["delta_miou"],
                "delta_ari": rwtd_flip["delta_ari"],
            },
            "stld_miou_agg_span": max(stld_agg_values) - min(stld_agg_values),
        },
        "notes": [
            "CAID is now included in both same-flavor tables.",
            "Legacy SAM3 coarse-only summaries on RWTD, CAID, and STLD publish `miou`/`ari` instead of `eval_miou`/`eval_ari`; the loader normalizes those fields directly from the bundle summaries.",
            "On STLD, miou_agg remains effectively flat across both flavors and both models.",
        ],
    }


def build_training_data(story_blocks: list[dict], comparison_rows: dict[str, list[dict]]) -> dict:
    return {
        "page": {
            "title": PAGE_TITLE,
            "date": PAGE_DATE,
            "status": "reviewed",
        },
        "comparisons": comparison_rows,
        "stories": [
            {
                "story_id": block["story_id"],
                "dataset": block["dataset"],
                "dataset_label": block["dataset_label"],
                "sample_id": block["sample_id"],
                "title": block["title"],
                "summary": block["summary"],
                "tag": block["tag"],
                "show_on_index": block["show_on_index"],
                "images": block["images"],
            }
            for block in story_blocks
        ],
    }


def build_links_payload(table_paths: dict[str, str]) -> dict:
    return {
        "page": "index.html",
        "gallery": "gallery.html",
        "metrics": "metrics.json",
        "training_data": "training_data.json",
        "summary": "summary.md",
        "manifest": "manifest.yaml",
        "tables": table_paths,
    }


def build_manifest(story_blocks: list[dict]) -> str:
    return dedent(
        f"""\
        title: "{PAGE_TITLE}"
        model_ids:
          - "{MODEL_META['sam2']['model_id']}"
          - "{MODEL_META['sam3']['model_id']}"
        dataset_ids:
          - "aviadcohz/RWTD"
          - "architexture:caid"
          - "architexture:stld"
        date: "{PAGE_DATE}"
        description: "Same-flavor SAM2-vs-SAM3 frozen-feature binary texture partitioning report for RWTD, CAID, and STLD."
        status: "reviewed"
        assets:
          story_block_count: {sum(1 for block in story_blocks if block['show_on_index'])}
          gallery_block_count: {len(story_blocks)}
          plot_count: 2
        evaluation:
          primary_metrics:
            - "eval_miou"
            - "eval_ari"
          caution_metric: "{CAUTION_METRIC}"
          metrics_file: "metrics.json"
        """
    )


def render_index_html(
    runs: dict[str, dict],
    comparison_rows: dict[str, list[dict]],
    story_blocks: list[dict],
    plot_paths: dict[str, str],
    table_paths: dict[str, str],
) -> str:
    coarse_rows = comparison_rows["coarse_only"]
    flip_rows = comparison_rows["flip_averaged"]
    stld_agg_span = max(
        runs["stld_sam2_coarse"]["miou_agg"],
        runs["stld_sam3_coarse"]["miou_agg"],
        runs["stld_sam2_flip"]["miou_agg"],
        runs["stld_sam3_flip"]["miou_agg"],
    ) - min(
        runs["stld_sam2_coarse"]["miou_agg"],
        runs["stld_sam3_coarse"]["miou_agg"],
        runs["stld_sam2_flip"]["miou_agg"],
        runs["stld_sam3_flip"]["miou_agg"],
    )
    coarse_leads = sum(1 for row in coarse_rows if row["leader_model"] == "sam2")
    flip_leads = sum(1 for row in flip_rows if row["leader_model"] == "sam2")
    largest_gain = max(coarse_rows + flip_rows, key=lambda row: (row["delta_miou"], row["delta_ari"]))
    rwtd_flip = next(row for row in flip_rows if row["dataset"] == "rwtd")

    headline_cards = dedent(
        f"""\
        <div class="metric-grid compact-grid">
          <article class="metric">
            <div class="k">Coarse-Only Lead Split</div>
            <div class="v">{coarse_leads}/3</div>
            <div class="mini-note">SAM2 leads all published coarse-only datasets.</div>
          </article>
          <article class="metric">
            <div class="k">Flip-Avg Lead Split</div>
            <div class="v">{flip_leads}/3</div>
            <div class="mini-note">SAM2 leads STLD and CAID; RWTD is the holdout.</div>
          </article>
          <article class="metric">
            <div class="k">Largest Same-Flavor Gain</div>
            <div class="v delta-positive">{signed(largest_gain['delta_miou'])}</div>
            <div class="mini-note">{largest_gain['dataset_label']} {FLAVOR_META[largest_gain['flavor']]['label']} · ARI {signed(largest_gain['delta_ari'])}</div>
          </article>
          <article class="metric">
            <div class="k">STLD miou_agg Span</div>
            <div class="v caution-readout">{stld_agg_span:.6f}</div>
            <div class="mini-note">All four STLD rows stay clustered around 0.5000.</div>
          </article>
        </div>
        """
    ).strip()

    dataset_cards = dedent(
        f"""\
        <div class="card-grid compact-grid">
          <article class="card">
            <h3>RWTD Takeaway</h3>
            <p class="interpretation-copy">RWTD is split by flavor. In <strong>coarse-only parity</strong>, SAM2 leads by <strong>{signed(coarse_rows[0]['delta_miou'], 6)}</strong> mIoU / <strong>{signed(coarse_rows[0]['delta_ari'], 6)}</strong> ARI. In <strong>flip-avg parity</strong>, SAM3 retains a small edge of <strong>{signed(-rwtd_flip['delta_miou'], 6)}</strong> / <strong>{signed(-rwtd_flip['delta_ari'], 6)}</strong>.</p>
          </article>
          <article class="card">
            <h3>CAID Takeaway</h3>
            <p class="interpretation-copy">CAID now has both flavors published for both models. SAM2 leads in <strong>coarse-only</strong> by <strong>{signed(coarse_rows[1]['delta_miou'], 6)}</strong> mIoU / <strong>{signed(coarse_rows[1]['delta_ari'], 6)}</strong> ARI and still keeps a smaller <strong>flip-avg</strong> edge of <strong>{signed(flip_rows[1]['delta_miou'], 6)}</strong> / <strong>{signed(flip_rows[1]['delta_ari'], 6)}</strong>.</p>
          </article>
          <article class="card">
            <h3>STLD Takeaway</h3>
            <p class="interpretation-copy">STLD is the clearest same-flavor win. SAM2 leads in <strong>coarse-only</strong> by <strong>{signed(coarse_rows[2]['delta_miou'], 6)}</strong> mIoU / <strong>{signed(coarse_rows[2]['delta_ari'], 6)}</strong> ARI and in <strong>flip-avg</strong> by <strong>{signed(flip_rows[2]['delta_miou'], 6)}</strong> / <strong>{signed(flip_rows[2]['delta_ari'], 6)}</strong>.</p>
            <p class="mini-note"><code>miou_agg</code> is nearly flat across all STLD rows, so it stays a caution metric rather than the headline metric.</p>
          </article>
        </div>
        """
    ).strip()

    caution_panel = dedent(
        f"""\
        <div class="caution-panel">
          <div>
            <span class="tag warn">metric caution</span>
            <h3>STLD <code>miou_agg</code> Is Effectively Flat</h3>
            <p class="interpretation-copy">Across all four STLD rows, the aggregate mIoU values are <code>{runs['stld_sam2_coarse']['miou_agg']:.6f}</code>, <code>{runs['stld_sam3_coarse']['miou_agg']:.6f}</code>, <code>{runs['stld_sam2_flip']['miou_agg']:.6f}</code>, and <code>{runs['stld_sam3_flip']['miou_agg']:.6f}</code>. The total spread is only <strong>{stld_agg_span:.6f}</strong>, so this aggregate view is low-signal for the STLD route.</p>
          </div>
          <div class="caution-list">
            <div><strong>Main readout</strong><span><code>eval_miou</code> and <code>eval_ari</code></span></div>
            <div><strong>Why</strong><span>Those metrics separate the models clearly where <code>miou_agg</code> barely moves.</span></div>
            <div><strong>Takeaway</strong><span>Do not over-interpret STLD aggregate mIoU for this route.</span></div>
          </div>
        </div>
        """
    ).strip()

    html = f"""<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{PAGE_TITLE}</title>
  <script>
    (() => {{
      const storageKey = 'research-site-theme';
      let theme = null;
      try {{
        const stored = window.localStorage.getItem(storageKey);
        if (stored === 'dark' || stored === 'light') theme = stored;
      }} catch (_) {{}}
      if (!theme && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {{
        theme = 'dark';
      }}
      document.documentElement.dataset.theme = theme || 'light';
      document.documentElement.style.colorScheme = theme || 'light';
    }})();
  </script>
  <link rel="stylesheet" href="../../assets/site.css" />
  <style>
    .compact-grid {{
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }}

    .caution-readout {{
      font-size: 1.55rem !important;
      line-height: 1.2;
    }}

    .chart-grid {{
      display: grid;
      gap: 20px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-top: 24px;
    }}

    .chart-card,
    .story-block,
    .caution-panel {{
      background: var(--surface-card);
      border: 1px solid var(--glass-border);
      border-radius: 20px;
      padding: 20px;
    }}

    .chart-card img {{
      width: 100%;
      display: block;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: var(--surface-strong);
    }}

    .comparison-stack {{
      display: grid;
      gap: 26px;
    }}

    .comparison-table table td,
    .comparison-table table th {{
      vertical-align: top;
    }}

    .delta-pill {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 6px 12px;
      font-size: 0.82rem;
      font-weight: 700;
      border: 1px solid var(--line);
      background: var(--surface-strong);
    }}

    .delta-pill.good {{
      color: var(--good);
    }}

    .delta-pill.bad {{
      color: var(--bad);
    }}

    .delta-positive {{
      color: var(--good) !important;
    }}

    .story-head {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 8px;
    }}

    .story-summary,
    .interpretation-copy {{
      margin: 0;
      color: var(--muted);
    }}

    .story-grid {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      margin-top: 18px;
    }}

    .story-grid img {{
      width: 100%;
      display: block;
      border-radius: 14px;
      border: 1px solid var(--line);
      cursor: zoom-in;
      background: var(--surface-strong);
    }}

    .mini-note {{
      font-size: 0.78rem;
      color: var(--muted);
      margin-top: 4px;
    }}

    .caution-panel {{
      display: grid;
      gap: 20px;
      grid-template-columns: 1.3fr 1fr;
      align-items: start;
    }}

    .caution-list {{
      display: grid;
      gap: 10px;
    }}

    .caution-list div {{
      display: grid;
      gap: 4px;
      padding: 14px;
      border: 1px solid var(--glass-border);
      border-radius: 16px;
      background: var(--surface-strong);
    }}

    .artifact-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 20px;
    }}

    .bullet-list {{
      margin: 16px 0 0;
      padding-left: 20px;
      color: var(--muted);
    }}

    .bullet-list li + li {{
      margin-top: 8px;
    }}

    @media (max-width: 960px) {{
      .chart-grid {{
        grid-template-columns: 1fr;
      }}

      .caution-panel {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>

<body>
  <div class="container animate-fade-in-down">
    <div class="page-tools">
      <button class="theme-toggle" type="button" data-theme-toggle aria-label="Toggle color theme" aria-pressed="false">
        <span data-theme-toggle-label>Dark mode</span>
      </button>
    </div>

    <header style="text-align: center; margin-bottom: 54px;">
      <div class="subtitle">{PAGE_SUBTITLE}</div>
      <h1 class="title-gradient">{PAGE_TITLE}</h1>
      <p style="color: var(--muted); margin-top: 16px; font-weight: 300; max-width: 940px; margin-inline: auto;">
        {escape(PAGE_DESCRIPTION)}
      </p>
      <div style="margin-top: 18px;">
        <span class="tag good">reviewed</span>
        <span class="tag">sam2</span>
        <span class="tag">sam3</span>
        <span class="tag">rwtd</span>
        <span class="tag">caid</span>
        <span class="tag">stld</span>
      </div>
    </header>

    <section class="executive-summary">
      <h2>Executive Synopsis</h2>
      <p>This page now uses the cleaner parity view: <strong>same dataset, same flavor, model columns side by side</strong>. Under that framing, SAM2 leads all three <strong>coarse-only</strong> rows and keeps the <strong>flip-avg</strong> lead on STLD and CAID, while RWTD flip-avg remains the lone SAM3 exception.</p>
      <p>The important caution is unchanged. On STLD, <code>miou_agg</code> stays clustered around <code>0.5</code> across both flavors and both models, so the main interpretable signal is still <code>eval_miou</code> and <code>eval_ari</code>.</p>
    </section>

    <section class="section">
      <h2>Headline Readout</h2>
      {headline_cards}
    </section>

    <section class="section comparison-table">
      <h2>Same-Flavor Model Comparison</h2>
      <p class="interpretation-copy">Dataset rows are intentionally minimal for readability. Each model cell shows <code>mIoU</code> on the first line and <code>ARI</code> below it, following the cross-dataset benchmark style.</p>
      <div class="comparison-stack" style="margin-top: 22px;">
        <div>
          <h3>{FLAVOR_META['coarse_only']['section_title']}</h3>
          <p class="mini-note">{FLAVOR_META['coarse_only']['summary']}</p>
          <div class="table-wrap" style="margin-top: 14px;">
            <table>
              <thead>
                <tr>
                  <th>Dataset</th>
                  <th>SAM2</th>
                  <th>SAM3</th>
                  <th>Δ mIoU</th>
                  <th>Δ ARI</th>
                  <th>Leader</th>
                </tr>
              </thead>
              <tbody>
                {render_comparison_table(coarse_rows)}
              </tbody>
            </table>
          </div>
          <div class="artifact-row">
            <a class="btn secondary" href="{table_paths['coarse_only']}">Open Coarse-Only CSV</a>
          </div>
        </div>

        <div>
          <h3>{FLAVOR_META['flip_averaged']['section_title']}</h3>
          <p class="mini-note">{FLAVOR_META['flip_averaged']['summary']}</p>
          <div class="table-wrap" style="margin-top: 14px;">
            <table>
              <thead>
                <tr>
                  <th>Dataset</th>
                  <th>SAM2</th>
                  <th>SAM3</th>
                  <th>Δ mIoU</th>
                  <th>Δ ARI</th>
                  <th>Leader</th>
                </tr>
              </thead>
              <tbody>
                {render_comparison_table(flip_rows)}
              </tbody>
            </table>
          </div>
          <div class="artifact-row">
            <a class="btn secondary" href="{table_paths['flip_averaged']}">Open Flip-Averaged CSV</a>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>Bigger-Picture Visuals</h2>
      <div class="chart-grid">
        <article class="chart-card">
          <img src="{plot_paths['eval_miou_parity.svg']}" alt="Same-flavor eval_mIoU grouped chart" />
        </article>
        <article class="chart-card">
          <img src="{plot_paths['eval_ari_parity.svg']}" alt="Same-flavor eval_ARI grouped chart" />
        </article>
      </div>
    </section>

    <section class="section">
      <h2>Metric Caution</h2>
      {caution_panel}
    </section>

    <section class="section">
      <h2>Dataset-Level Takeaways</h2>
      {dataset_cards}
    </section>

    <section class="section">
      <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 20px; flex-wrap: wrap;">
        <div>
          <h2>Qualitative Comparison</h2>
          <p class="interpretation-copy">The qualitative panels are still curated rather than exhaustive. They now follow the same-flavor parity framing used by the main tables.</p>
        </div>
        <a class="btn" href="gallery.html">Open Curated Gallery →</a>
      </div>
      <div style="margin-top: 24px;">
        {render_story_blocks(story_blocks, index_only=True)}
      </div>
    </section>

    <section class="section">
      <h2>Interpretation</h2>
      <p class="interpretation-copy">This still does <strong>not</strong> mean SAM2 is the better model overall. It does mean that, under same-flavor frozen-feature clustering parity, SAM2 appears more cluster-friendly than SAM3 in most of the published binary partition settings here. The exception matters too: RWTD flip-avg stays a slight SAM3 holdout, so the result is route-specific rather than absolute.</p>
    </section>

    <section class="section">
      <h2>Caveats / Notes</h2>
      <ul class="bullet-list">
        <li>Legacy SAM3 coarse-only summaries publish <code>miou</code>/<code>ari</code> instead of <code>eval_miou</code>/<code>eval_ari</code>; this report normalizes those fields directly from the summary bundles.</li>
        <li>CAID is now included in both same-flavor tables, but the qualitative section remains curated around the clearest RWTD/STLD visual examples.</li>
        <li>On STLD, <code>miou_agg</code> is visually marked as low-signal because it is nearly constant across both models and both flavors.</li>
        <li>This is a route-specific frozen-feature clustering comparison, not a claim about overall SAM2-vs-SAM3 project performance.</li>
      </ul>
      <div class="artifact-row">
        <a class="btn secondary" href="metrics.json">metrics.json</a>
        <a class="btn secondary" href="training_data.json">training_data.json</a>
        <a class="btn secondary" href="summary.md">summary.md</a>
      </div>
    </section>

    <footer>
      <p>SAM2 vs SAM3 route report | Rendered {PAGE_DATE}</p>
    </footer>
  </div>

  <dialog class="image-dialog" id="imageDialog">
    <img id="imageDialogImg" alt="Expanded preview" />
  </dialog>

  <script src="../../assets/site.js"></script>
</body>

</html>
"""
    return html


def render_gallery_html(story_blocks: list[dict]) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{PAGE_TITLE} — Gallery</title>
  <script>
    (() => {{
      const storageKey = 'research-site-theme';
      let theme = null;
      try {{
        const stored = window.localStorage.getItem(storageKey);
        if (stored === 'dark' || stored === 'light') theme = stored;
      }} catch (_) {{}}
      if (!theme && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {{
        theme = 'dark';
      }}
      document.documentElement.dataset.theme = theme || 'light';
      document.documentElement.style.colorScheme = theme || 'light';
    }})();
  </script>
  <link rel="stylesheet" href="../../assets/site.css" />
  <style>
    .story-block {{
      background: var(--surface-card);
      border: 1px solid var(--glass-border);
      border-radius: 20px;
      padding: 22px;
      margin-bottom: 20px;
    }}

    .story-head {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 8px;
    }}

    .story-summary {{
      margin: 0;
      color: var(--muted);
    }}

    .story-grid {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      margin-top: 18px;
    }}

    .story-grid img {{
      width: 100%;
      display: block;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: var(--surface-strong);
      cursor: zoom-in;
    }}

    .mini-note {{
      font-size: 0.78rem;
      color: var(--muted);
      margin-top: 4px;
    }}
  </style>
</head>

<body>
  <div class="container animate-fade-in-down">
    <div class="page-tools">
      <button class="theme-toggle" type="button" data-theme-toggle aria-label="Toggle color theme" aria-pressed="false">
        <span data-theme-toggle-label>Dark mode</span>
      </button>
    </div>

    <header style="text-align: center; margin-bottom: 44px;">
      <div class="subtitle">Curated Gallery</div>
      <h1 class="title-gradient">{PAGE_TITLE}</h1>
      <p style="color: var(--muted); margin-top: 16px; font-weight: 300; max-width: 860px; margin-inline: auto;">
        A lightweight qualitative subset built from the run-emitted visual triptychs. The gallery follows the same-flavor parity framing used by the main report.
      </p>
      <div style="margin-top: 18px;">
        <a class="btn secondary" href="index.html">← Back to Report</a>
      </div>
    </header>

    <section class="section">
      {render_story_blocks(story_blocks, index_only=False)}
    </section>

    <footer>
      <p>Curated comparison subset | Lightweight previews only</p>
    </footer>
  </div>

  <dialog class="image-dialog" id="imageDialog">
    <img id="imageDialogImg" alt="Expanded preview" />
  </dialog>

  <script src="../../assets/site.js"></script>
</body>

</html>
"""


def render_summary_md(comparison_rows: dict[str, list[dict]]) -> str:
    return dedent(
        f"""\
        # {PAGE_TITLE}

        - Date: `{PAGE_DATE}`
        - Scope: RWTD, CAID, and STLD same-flavor frozen-feature binary partitioning
        - Coarse-only lead split: `SAM2 {sum(1 for row in comparison_rows['coarse_only'] if row['leader_model'] == 'sam2')}/3`
        - Flip-avg lead split: `SAM2 {sum(1 for row in comparison_rows['flip_averaged'] if row['leader_model'] == 'sam2')}/3`
        - Coarse-only rows:
          - RWTD: `SAM2 {comparison_rows['coarse_only'][0]['sam2_miou']:.6f}/{comparison_rows['coarse_only'][0]['sam2_ari']:.6f}` vs `SAM3 {comparison_rows['coarse_only'][0]['sam3_miou']:.6f}/{comparison_rows['coarse_only'][0]['sam3_ari']:.6f}`
          - CAID: `SAM2 {comparison_rows['coarse_only'][1]['sam2_miou']:.6f}/{comparison_rows['coarse_only'][1]['sam2_ari']:.6f}` vs `SAM3 {comparison_rows['coarse_only'][1]['sam3_miou']:.6f}/{comparison_rows['coarse_only'][1]['sam3_ari']:.6f}`
          - STLD: `SAM2 {comparison_rows['coarse_only'][2]['sam2_miou']:.6f}/{comparison_rows['coarse_only'][2]['sam2_ari']:.6f}` vs `SAM3 {comparison_rows['coarse_only'][2]['sam3_miou']:.6f}/{comparison_rows['coarse_only'][2]['sam3_ari']:.6f}`
        - Flip-avg rows:
          - RWTD: `SAM2 {comparison_rows['flip_averaged'][0]['sam2_miou']:.6f}/{comparison_rows['flip_averaged'][0]['sam2_ari']:.6f}` vs `SAM3 {comparison_rows['flip_averaged'][0]['sam3_miou']:.6f}/{comparison_rows['flip_averaged'][0]['sam3_ari']:.6f}`
          - CAID: `SAM2 {comparison_rows['flip_averaged'][1]['sam2_miou']:.6f}/{comparison_rows['flip_averaged'][1]['sam2_ari']:.6f}` vs `SAM3 {comparison_rows['flip_averaged'][1]['sam3_miou']:.6f}/{comparison_rows['flip_averaged'][1]['sam3_ari']:.6f}`
          - STLD: `SAM2 {comparison_rows['flip_averaged'][2]['sam2_miou']:.6f}/{comparison_rows['flip_averaged'][2]['sam2_ari']:.6f}` vs `SAM3 {comparison_rows['flip_averaged'][2]['sam3_miou']:.6f}/{comparison_rows['flip_averaged'][2]['sam3_ari']:.6f}`
        - Caution: STLD `miou_agg` remains nearly flat across both models and both flavors.
        """
    )


def main() -> None:
    runs = {spec.run_id: load_run(spec) for spec in RUN_SPECS}
    enforce_alignment(runs)

    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    comparison_rows = build_comparison_rows(runs)
    story_blocks = build_story_blocks(runs)
    plot_paths = write_plot_assets(comparison_rows)
    table_paths = write_table_csvs(comparison_rows)
    metrics_payload = build_metrics_payload(runs, comparison_rows)
    training_data = build_training_data(story_blocks, comparison_rows)
    links_payload = build_links_payload(table_paths)

    write_text(DEST_DIR / "index.html", render_index_html(runs, comparison_rows, story_blocks, plot_paths, table_paths))
    write_text(DEST_DIR / "gallery.html", render_gallery_html(story_blocks))
    write_text(DEST_DIR / "metrics.json", json.dumps(metrics_payload, indent=2))
    write_text(DEST_DIR / "training_data.json", json.dumps(training_data, indent=2))
    write_text(DEST_DIR / "links.json", json.dumps(links_payload, indent=2))
    write_text(DEST_DIR / "summary.md", render_summary_md(comparison_rows))
    write_text(DEST_DIR / "manifest.yaml", build_manifest(story_blocks))


if __name__ == "__main__":
    main()
