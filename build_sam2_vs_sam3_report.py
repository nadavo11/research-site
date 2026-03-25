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
    "Automatic binary texture partitioning with frozen features and coarse clustering on RWTD and STLD. "
    "In this narrow unsupervised route, the SAM2 variants outperform the comparable SAM3 variants on "
    "the headline evaluation metrics, while STLD aggregate mIoU stays nearly flat around 0.5 and is "
    "better treated as a caution metric than as the main story."
)

PRIMARY_METRICS = ("eval_miou", "eval_ari")
CAUTION_METRIC = "miou_agg"

RUN_COLORS = {
    "rwtd_sam3_flip": "#ca6702",
    "rwtd_sam2_coarse": "#005f73",
    "stld_sam3_flip": "#ca6702",
    "stld_sam2_coarse": "#005f73",
    "stld_sam2_flip": "#0a9396",
}

MODEL_LABELS = {
    "facebook/sam2-hiera-small": "SAM2",
    "facebook/sam3": "SAM3",
}


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    route: str
    route_label: str
    dataset_label: str
    split_label: str
    source_dir: str
    display_label: str
    short_label: str
    note: str


RUN_SPECS = [
    RunSpec(
        run_id="rwtd_sam2_coarse",
        route="rwtd",
        route_label="RWTD",
        dataset_label="aviadcohz/RWTD",
        split_label="test",
        source_dir="rwtd/sam2_vanilla_cfc",
        display_label="RWTD · SAM2 coarse_only",
        short_label="SAM2 coarse",
        note="SAM2 coarse-only on RWTD is the requested cross-model comparator.",
    ),
    RunSpec(
        run_id="rwtd_sam3_flip",
        route="rwtd",
        route_label="RWTD",
        dataset_label="aviadcohz/RWTD",
        split_label="test",
        source_dir="rwtd/flip_averaged",
        display_label="RWTD · SAM3 flip_avg",
        short_label="SAM3 flip",
        note="SAM3 flip-averaged coarse-only baseline on RWTD.",
    ),
    RunSpec(
        run_id="stld_sam2_coarse",
        route="stld",
        route_label="STLD",
        dataset_label="architexture:stld",
        split_label="benchmark",
        source_dir="stld/sam2_vanilla_cfc",
        display_label="STLD · SAM2 coarse_only",
        short_label="SAM2 coarse",
        note="SAM2 coarse-only on STLD.",
    ),
    RunSpec(
        run_id="stld_sam3_flip",
        route="stld",
        route_label="STLD",
        dataset_label="architexture:stld",
        split_label="benchmark",
        source_dir="stld/flip_averaged",
        display_label="STLD · SAM3 flip_avg",
        short_label="SAM3 flip",
        note="SAM3 flip-averaged coarse-only baseline on STLD.",
    ),
    RunSpec(
        run_id="stld_sam2_flip",
        route="stld",
        route_label="STLD",
        dataset_label="architexture:stld",
        split_label="benchmark",
        source_dir="stld/sam2_flipavg_cfc",
        display_label="STLD · SAM2 flip_avg",
        short_label="SAM2 flip",
        note="STLD SAM2 flip-averaging improves further over the SAM3 flip-avg baseline.",
    ),
]

RUN_SPEC_BY_ID = {spec.run_id: spec for spec in RUN_SPECS}
RUN_ORDER = [spec.run_id for spec in RUN_SPECS]
ROUTE_ORDER = ["rwtd", "stld"]
BASELINE_BY_ROUTE = {"rwtd": "rwtd_sam3_flip", "stld": "stld_sam3_flip"}
CHART_GROUP_ORDER = {
    "rwtd": ["rwtd_sam3_flip", "rwtd_sam2_coarse"],
    "stld": ["stld_sam3_flip", "stld_sam2_coarse", "stld_sam2_flip"],
}

DELTA_ROWS = [
    {
        "delta_id": "rwtd_sam2_coarse_vs_sam3_flip",
        "route": "rwtd",
        "title": "RWTD: SAM2 coarse vs SAM3 flip",
        "left_run_id": "rwtd_sam2_coarse",
        "right_run_id": "rwtd_sam3_flip",
        "summary": "Positive direction, but smaller than STLD and still worth reading cautiously until parity questions are locked down.",
    },
    {
        "delta_id": "stld_sam2_coarse_vs_sam3_flip",
        "route": "stld",
        "title": "STLD: SAM2 coarse vs SAM3 flip",
        "left_run_id": "stld_sam2_coarse",
        "right_run_id": "stld_sam3_flip",
        "summary": "Clear STLD uplift in both headline metrics.",
    },
    {
        "delta_id": "stld_sam2_flip_vs_sam3_flip",
        "route": "stld",
        "title": "STLD: SAM2 flip vs SAM3 flip",
        "left_run_id": "stld_sam2_flip",
        "right_run_id": "stld_sam3_flip",
        "summary": "Largest in-page gain. The SAM2 flip-avg variant is the strongest STLD row in this slice.",
    },
]

MAIN_STORIES = [
    {
        "story_id": "rwtd_granular_bands",
        "route": "rwtd",
        "route_label": "RWTD",
        "sample_id": "237",
        "title": "RWTD: SAM2 Preserves the Three-Band Split",
        "summary": (
            "This granular crop has a narrow central light band flanked by two different coarse textures. "
            "SAM2 coarse tracks the split cleanly, while the comparable SAM3 flip-avg run collapses the center into a broad blob."
        ),
        "variant_order": ["rwtd_sam3_flip", "rwtd_sam2_coarse"],
        "tag": "clear win",
        "show_on_index": True,
    },
    {
        "story_id": "stld_compact_foreground",
        "route": "stld",
        "route_label": "STLD",
        "sample_id": "66",
        "title": "STLD: Both SAM2 Variants Recover the Object",
        "summary": (
            "On this compact synthetic foreground, both SAM2 variants recover the object cleanly. "
            "The SAM3 flip baseline expands into a large false-positive disk instead."
        ),
        "variant_order": ["stld_sam3_flip", "stld_sam2_coarse", "stld_sam2_flip"],
        "tag": "stld gain",
        "show_on_index": True,
    },
    {
        "story_id": "stld_thin_strip",
        "route": "stld",
        "route_label": "STLD",
        "sample_id": "41",
        "title": "STLD: Thin Structure Favors the SAM2 Variants",
        "summary": (
            "The thin oblique strip is a harder shape. SAM2 coarse recovers part of it, SAM2 flip captures more of it, "
            "and the SAM3 flip baseline nearly collapses the signal."
        ),
        "variant_order": ["stld_sam3_flip", "stld_sam2_coarse", "stld_sam2_flip"],
        "tag": "thin structure",
        "show_on_index": True,
    },
    {
        "story_id": "rwtd_concrete_gravel",
        "route": "rwtd",
        "route_label": "RWTD",
        "sample_id": "261",
        "title": "RWTD: Another Strong Texture Split",
        "summary": (
            "SAM2 coarse separates smooth concrete from coarse gravel and the neighboring red slab cleanly; "
            "the SAM3 flip baseline merges the scene into two broad low-fidelity regions."
        ),
        "variant_order": ["rwtd_sam3_flip", "rwtd_sam2_coarse"],
        "tag": "gallery",
        "show_on_index": False,
    },
    {
        "story_id": "stld_diagonal_strip",
        "route": "stld",
        "route_label": "STLD",
        "sample_id": "2",
        "title": "STLD: Another Synthetic Foreground Win",
        "summary": (
            "A second thin foreground example. Both SAM2 rows remain stable, while the SAM3 flip baseline overfills a large surrounding region."
        ),
        "variant_order": ["stld_sam3_flip", "stld_sam2_coarse", "stld_sam2_flip"],
        "tag": "gallery",
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


def model_family(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id)


def slugify(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text)


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
                    "eval_miou": float(payload["eval_miou"]),
                    "eval_ari": float(payload["eval_ari"]),
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
    return {
        "run_id": spec.run_id,
        "route": spec.route,
        "route_label": spec.route_label,
        "dataset_label": summary.get("dataset_id") or spec.dataset_label,
        "split_label": summary.get("split") or spec.split_label,
        "display_label": spec.display_label,
        "short_label": spec.short_label,
        "exact_variant": summary.get("variant") or "",
        "model_id": summary.get("model_id") or "",
        "model_label": model_family(summary.get("model_id") or ""),
        "note": spec.note,
        "num_samples": int(summary.get("num_evaluated_samples") or 0),
        "eval_miou": float(metrics["eval_miou"]),
        "eval_ari": float(metrics["eval_ari"]),
        "miou_agg": float(metrics["miou_agg"]),
        "source_dir": run_dir,
        "sample_lookup": sample_lookup,
        "rows": rows,
    }


def enforce_alignment(runs: dict[str, dict]) -> None:
    for route in ROUTE_ORDER:
        route_runs = [run for run in runs.values() if run["route"] == route]
        baseline_set = set(route_runs[0]["sample_lookup"])
        mismatches = []
        for run in route_runs[1:]:
            current = set(run["sample_lookup"])
            if current != baseline_set:
                mismatches.append(f"{run['run_id']} ({len(baseline_set.symmetric_difference(current))} mismatched ids)")
        if mismatches:
            raise RuntimeError(f"Sample alignment mismatch for {route}: {', '.join(mismatches)}")


def build_run_rows(runs: dict[str, dict]) -> list[dict]:
    rows = []
    for run_id in RUN_ORDER:
        run = runs[run_id]
        rows.append(
            {
                "route_label": run["route_label"],
                "dataset_label": run["dataset_label"],
                "split_label": run["split_label"],
                "display_label": run["display_label"],
                "short_label": run["short_label"],
                "exact_variant": run["exact_variant"],
                "model_id": run["model_id"],
                "num_samples": run["num_samples"],
                "eval_miou": run["eval_miou"],
                "eval_ari": run["eval_ari"],
                "miou_agg": run["miou_agg"],
                "note": run["note"],
                "run_id": run_id,
            }
        )
    return rows


def build_delta_rows(runs: dict[str, dict]) -> list[dict]:
    rows = []
    for item in DELTA_ROWS:
        left = runs[item["left_run_id"]]
        right = runs[item["right_run_id"]]
        rows.append(
            {
                **item,
                "left_label": left["display_label"],
                "right_label": right["display_label"],
                "delta_miou": left["eval_miou"] - right["eval_miou"],
                "delta_ari": left["eval_ari"] - right["eval_ari"],
            }
        )
    return rows


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
                    "label": run["display_label"],
                    "short_label": run["short_label"],
                    "model_label": run["model_label"],
                    "asset_rel": asset_rel.as_posix(),
                    "eval_miou": row["eval_miou"],
                    "eval_ari": row["eval_ari"],
                }
            )
        blocks.append(
            {
                **spec,
                "images": images,
            }
        )
    return blocks


def render_grouped_bar_chart(title: str, subtitle: str, runs: dict[str, dict], metric_key: str) -> str:
    width = 980
    height = 470
    top = 92
    left = 78
    right = 34
    bottom = 122
    chart_height = height - top - bottom
    chart_width = width - left - right
    bar_width = 82
    gap = 18
    group_gap = 74
    total_inner_width = 0
    for route in ROUTE_ORDER:
        group_ids = CHART_GROUP_ORDER[route]
        total_inner_width += len(group_ids) * bar_width + max(0, len(group_ids) - 1) * gap
    total_inner_width += group_gap * (len(ROUTE_ORDER) - 1)
    start_x = left + max(0, (chart_width - total_inner_width) / 2)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        '<rect width="100%" height="100%" fill="#ffffff" rx="24" />',
        f'<text x="28" y="40" font-family="Space Grotesk, sans-serif" font-size="24" font-weight="700" fill="#122033">{escape(title)}</text>',
        f'<text x="28" y="64" font-family="Space Grotesk, sans-serif" font-size="13" fill="#4c5d73">{escape(subtitle)}</text>',
        '<rect x="720" y="24" width="14" height="14" rx="4" fill="#005f73" />',
        '<text x="742" y="36" font-family="Space Grotesk, sans-serif" font-size="12" fill="#122033">SAM2</text>',
        '<rect x="804" y="24" width="14" height="14" rx="4" fill="#ca6702" />',
        '<text x="826" y="36" font-family="Space Grotesk, sans-serif" font-size="12" fill="#122033">SAM3</text>',
    ]

    for tick in [0.0, 0.25, 0.50, 0.75, 1.0]:
        y = top + chart_height * (1.0 - tick)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e3ebf2" stroke-width="1" />')
        lines.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Space Grotesk, sans-serif" font-size="11" fill="#4c5d73">{tick:.2f}</text>'
        )

    cursor_x = start_x
    for route in ROUTE_ORDER:
        group_ids = CHART_GROUP_ORDER[route]
        group_start = cursor_x
        for run_id in group_ids:
            run = runs[run_id]
            value = run[metric_key]
            bar_height = chart_height * max(0.0, min(1.0, value))
            x = cursor_x
            y = top + chart_height - bar_height
            lines.extend(
                [
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" rx="12" fill="{RUN_COLORS[run_id]}" />',
                    f'<text x="{x + bar_width / 2:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-family="Space Grotesk, sans-serif" font-size="12" font-weight="700" fill="#122033">{value:.4f}</text>',
                    f'<text x="{x + bar_width / 2:.1f}" y="{height - 64}" text-anchor="middle" font-family="Space Grotesk, sans-serif" font-size="11" font-weight="600" fill="#122033">{escape(run["short_label"])}</text>',
                    f'<text x="{x + bar_width / 2:.1f}" y="{height - 48}" text-anchor="middle" font-family="Space Grotesk, sans-serif" font-size="10" fill="#4c5d73">{escape(run["model_label"])}</text>',
                ]
            )
            cursor_x += bar_width + gap
        cursor_x -= gap
        group_center = (group_start + cursor_x + bar_width) / 2 if len(group_ids) == 1 else (group_start + cursor_x - gap) / 2
        lines.append(
            f'<text x="{group_center:.1f}" y="{height - 20}" text-anchor="middle" font-family="Outfit, sans-serif" font-size="15" font-weight="700" fill="#005f73">{escape(runs[group_ids[0]]["route_label"])}</text>'
        )
        cursor_x += group_gap

    lines.append("</svg>")
    return "\n".join(lines)


def write_plot_assets(runs: dict[str, dict]) -> dict[str, str]:
    plots = {
        "eval_miou_grouped.svg": render_grouped_bar_chart(
            "Grouped eval_mIoU",
            "Grouped by route. The SAM2 bars are consistently higher than the comparable SAM3 bars in this slice.",
            runs,
            "eval_miou",
        ),
        "eval_ari_grouped.svg": render_grouped_bar_chart(
            "Grouped eval_ARI",
            "ARI tells the same route-level story as mIoU, especially on STLD.",
            runs,
            "eval_ari",
        ),
    }
    for file_name, content in plots.items():
        write_text(DEST_DIR / "assets" / "plots" / file_name, content)
    return {name: f"assets/plots/{name}" for name in plots}


def write_table_csvs(run_rows: list[dict], delta_rows: list[dict]) -> dict[str, str]:
    tables_dir = DEST_DIR / "assets" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    compared_path = tables_dir / "what_was_compared.csv"
    compared_fields = [
        "route_label",
        "dataset_label",
        "split_label",
        "display_label",
        "exact_variant",
        "model_id",
        "num_samples",
        "eval_miou",
        "eval_ari",
        "miou_agg",
        "note",
    ]
    with compared_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=compared_fields)
        writer.writeheader()
        writer.writerows([{field: row[field] for field in compared_fields} for row in run_rows])

    delta_path = tables_dir / "delta_vs_sam3_baseline.csv"
    delta_fields = [
        "delta_id",
        "route",
        "title",
        "left_label",
        "right_label",
        "delta_miou",
        "delta_ari",
        "summary",
    ]
    with delta_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=delta_fields)
        writer.writeheader()
        writer.writerows([{field: row[field] for field in delta_fields} for row in delta_rows])

    return {
        "what_was_compared": compared_path.relative_to(DEST_DIR).as_posix(),
        "delta_vs_sam3": delta_path.relative_to(DEST_DIR).as_posix(),
    }


def render_run_rows(run_rows: list[dict]) -> str:
    html_rows = []
    for row in run_rows:
        caution = row["route_label"] == "STLD"
        agg_cell = (
            f'<strong>{row["miou_agg"]:.6f}</strong><div class="mini-note">flat / low-signal on STLD</div>'
            if caution
            else f'<strong>{row["miou_agg"]:.6f}</strong>'
        )
        html_rows.append(
            dedent(
                f"""\
                <tr>
                  <td><strong>{escape(row['route_label'])}</strong></td>
                  <td><code>{escape(row['dataset_label'])}</code><div class="mini-note">{escape(row['split_label'])}</div></td>
                  <td><code>{escape(row['exact_variant'])}</code></td>
                  <td><code>{escape(row['model_id'])}</code></td>
                  <td>{row['num_samples']}</td>
                  <td><strong>{row['eval_miou']:.6f}</strong></td>
                  <td><strong>{row['eval_ari']:.6f}</strong></td>
                  <td>{agg_cell}</td>
                </tr>
                """
            ).strip()
        )
    return "\n".join(html_rows)


def render_delta_cards(delta_rows: list[dict]) -> str:
    cards = []
    for row in delta_rows:
        cards.append(
            dedent(
                f"""\
                <article class="metric">
                  <div class="k">{escape(row['title'])}</div>
                  <div class="v delta-positive">{signed(row['delta_miou'])}</div>
                  <div class="mini-note">mIoU uplift · ARI {signed(row['delta_ari'])}</div>
                </article>
                """
            ).strip()
        )
    return "\n".join(cards)


def render_delta_table(delta_rows: list[dict]) -> str:
    rows = []
    for row in delta_rows:
        rows.append(
            dedent(
                f"""\
                <tr>
                  <td><strong>{escape(row['title'])}</strong></td>
                  <td><code>{escape(row['left_label'])}</code></td>
                  <td><code>{escape(row['right_label'])}</code></td>
                  <td><span class="delta-pill good">{signed(row['delta_miou'], 6)}</span></td>
                  <td><span class="delta-pill good">{signed(row['delta_ari'], 6)}</span></td>
                  <td>{escape(row['summary'])}</td>
                </tr>
                """
            ).strip()
        )
    return "\n".join(rows)


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
                    <span class="pill">{escape(block['route_label'])} · crop {escape(block['sample_id'])}</span>
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


def build_metrics_payload(runs: dict[str, dict], delta_rows: list[dict]) -> dict:
    stld_agg_values = [runs["stld_sam2_coarse"]["miou_agg"], runs["stld_sam3_flip"]["miou_agg"], runs["stld_sam2_flip"]["miou_agg"]]
    return {
        "status": "reviewed",
        "page_title": PAGE_TITLE,
        "primary_metrics": list(PRIMARY_METRICS),
        "caution_metric": CAUTION_METRIC,
        "runs": [
            {
                "run_id": run_id,
                "route": runs[run_id]["route"],
                "route_label": runs[run_id]["route_label"],
                "display_label": runs[run_id]["display_label"],
                "exact_variant": runs[run_id]["exact_variant"],
                "model_id": runs[run_id]["model_id"],
                "num_samples": runs[run_id]["num_samples"],
                "eval_miou": runs[run_id]["eval_miou"],
                "eval_ari": runs[run_id]["eval_ari"],
                "miou_agg": runs[run_id]["miou_agg"],
            }
            for run_id in RUN_ORDER
        ],
        "delta_vs_sam3": [
            {
                "delta_id": row["delta_id"],
                "route": row["route"],
                "title": row["title"],
                "delta_miou": row["delta_miou"],
                "delta_ari": row["delta_ari"],
            }
            for row in delta_rows
        ],
        "headline": {
            "rwtd_winner": {
                "run_id": "rwtd_sam2_coarse",
                "eval_miou": runs["rwtd_sam2_coarse"]["eval_miou"],
                "eval_ari": runs["rwtd_sam2_coarse"]["eval_ari"],
            },
            "stld_winner": {
                "run_id": "stld_sam2_flip",
                "eval_miou": runs["stld_sam2_flip"]["eval_miou"],
                "eval_ari": runs["stld_sam2_flip"]["eval_ari"],
            },
            "stld_miou_agg_span": max(stld_agg_values) - min(stld_agg_values),
        },
        "notes": [
            "This page is intentionally scoped to the requested RWTD/STLD comparison rows.",
            "CAID is not included because the matching CAID SAM2 flip-avg row is still missing for this focused page.",
            "On STLD, miou_agg is nearly constant around 0.5 and should not be treated as the main signal.",
        ],
    }


def build_training_data(story_blocks: list[dict], delta_rows: list[dict]) -> dict:
    return {
        "page": {
            "title": PAGE_TITLE,
            "date": PAGE_DATE,
            "status": "reviewed",
        },
        "stories": [
            {
                "story_id": block["story_id"],
                "route": block["route"],
                "route_label": block["route_label"],
                "sample_id": block["sample_id"],
                "title": block["title"],
                "summary": block["summary"],
                "tag": block["tag"],
                "show_on_index": block["show_on_index"],
                "images": block["images"],
            }
            for block in story_blocks
        ],
        "delta_rows": [
            {
                "delta_id": row["delta_id"],
                "route": row["route"],
                "title": row["title"],
                "delta_miou": row["delta_miou"],
                "delta_ari": row["delta_ari"],
            }
            for row in delta_rows
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
          - "facebook/sam2-hiera-small"
          - "facebook/sam3"
        dataset_ids:
          - "aviadcohz/RWTD"
          - "architexture:stld"
        date: "{PAGE_DATE}"
        description: "SAM2-vs-SAM3 frozen-feature binary texture partitioning report for RWTD and STLD."
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
    run_rows: list[dict],
    delta_rows: list[dict],
    story_blocks: list[dict],
    plot_paths: dict[str, str],
    table_paths: dict[str, str],
) -> str:
    rwtd_delta = delta_rows[0]
    stld_coarse_delta = delta_rows[1]
    stld_flip_delta = delta_rows[2]
    stld_agg_span = max(runs["stld_sam2_coarse"]["miou_agg"], runs["stld_sam3_flip"]["miou_agg"], runs["stld_sam2_flip"]["miou_agg"]) - min(
        runs["stld_sam2_coarse"]["miou_agg"],
        runs["stld_sam3_flip"]["miou_agg"],
        runs["stld_sam2_flip"]["miou_agg"],
    )

    headline_cards = dedent(
        f"""\
        <div class="metric-grid compact-grid">
          <article class="metric">
            <div class="k">RWTD Winner</div>
            <div class="v">{runs['rwtd_sam2_coarse']['eval_miou']:.4f}</div>
            <div class="mini-note">SAM2 coarse · ARI {runs['rwtd_sam2_coarse']['eval_ari']:.4f}</div>
          </article>
          <article class="metric">
            <div class="k">STLD Winner</div>
            <div class="v">{runs['stld_sam2_flip']['eval_miou']:.4f}</div>
            <div class="mini-note">SAM2 flip · ARI {runs['stld_sam2_flip']['eval_ari']:.4f}</div>
          </article>
          <article class="metric">
            <div class="k">Largest Uplift vs SAM3</div>
            <div class="v delta-positive">{signed(stld_flip_delta['delta_miou'])}</div>
            <div class="mini-note">STLD SAM2 flip · ARI {signed(stld_flip_delta['delta_ari'])}</div>
          </article>
          <article class="metric">
            <div class="k">STLD miou_agg Span</div>
            <div class="v caution-readout">{stld_agg_span:.6f}</div>
            <div class="mini-note">All three STLD rows stay ~0.5000</div>
          </article>
        </div>
        """
    ).strip()

    route_cards = dedent(
        f"""\
        <div class="card-grid compact-grid">
          <article class="card">
            <h3>RWTD Takeaway</h3>
            <p class="interpretation-copy">On the requested RWTD pair, <strong>SAM2 coarse_only</strong> beats the comparable <strong>SAM3 flip_avg</strong> row by <strong>{signed(rwtd_delta['delta_miou'], 6)}</strong> mIoU and <strong>{signed(rwtd_delta['delta_ari'], 6)}</strong> ARI across 227 test crops.</p>
            <p class="mini-note">Direction is favorable, but this route should still be read carefully until evaluator and preprocessing parity are fully locked.</p>
          </article>
          <article class="card">
            <h3>STLD Takeaway</h3>
            <p class="interpretation-copy">STLD shows the clearer separation. SAM2 coarse beats the SAM3 flip baseline by <strong>{signed(stld_coarse_delta['delta_miou'], 6)}</strong> mIoU / <strong>{signed(stld_coarse_delta['delta_ari'], 6)}</strong> ARI, and SAM2 flip extends that to <strong>{signed(stld_flip_delta['delta_miou'], 6)}</strong> / <strong>{signed(stld_flip_delta['delta_ari'], 6)}</strong>.</p>
            <p class="mini-note"><code>miou_agg</code> stays nearly flat around 0.5, so <code>eval_miou</code> and <code>eval_ari</code> are the operative signal here.</p>
          </article>
        </div>
        """
    ).strip()

    caution_panel = dedent(
        f"""\
        <div class="caution-panel">
          <div>
            <span class="tag warn">metric caution</span>
            <h3>STLD <code>miou_agg</code> is Essentially Flat</h3>
            <p class="interpretation-copy">The STLD aggregate mIoU values are <code>{runs['stld_sam2_coarse']['miou_agg']:.6f}</code>, <code>{runs['stld_sam3_flip']['miou_agg']:.6f}</code>, and <code>{runs['stld_sam2_flip']['miou_agg']:.6f}</code>. That spread is only <strong>{stld_agg_span:.6f}</strong>, so this aggregate view is low-signal for the STLD route and should not carry the interpretation.</p>
          </div>
          <div class="caution-list">
            <div><strong>Main readout</strong><span><code>eval_miou</code> and <code>eval_ari</code></span></div>
            <div><strong>Why</strong><span>They separate the variants clearly where <code>miou_agg</code> barely moves.</span></div>
            <div><strong>Takeaway</strong><span>Treat <code>miou_agg</code> as a caution metric here, not the headline metric.</span></div>
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

    .delta-grid {{
      display: grid;
      gap: 20px;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      margin-top: 22px;
    }}

    .delta-positive {{
      color: var(--good) !important;
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
      color: var(--good);
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
      <p style="color: var(--muted); margin-top: 16px; font-weight: 300; max-width: 900px; margin-inline: auto;">
        {escape(PAGE_DESCRIPTION)}
      </p>
      <div style="margin-top: 18px;">
        <span class="tag good">reviewed</span>
        <span class="tag">sam2</span>
        <span class="tag">sam3</span>
        <span class="tag">unsupervised</span>
        <span class="tag">coarse clustering</span>
      </div>
    </header>

    <section class="executive-summary">
      <h2>Executive Synopsis</h2>
      <p>This page compares automatic binary partitioning from frozen SAM features on <strong>RWTD</strong> and <strong>STLD</strong>. The compared route is deliberately narrow: no prompts, no supervised decoder, and no learned downstream segmentation head, just coarse clustering over frozen features.</p>
      <p>Within that route, the requested SAM2 rows outperform the comparable SAM3 rows on the main evaluation metrics. The page also marks a caution explicitly: on STLD, <code>miou_agg</code> stays nearly flat around <code>0.5</code>, so it should be treated as a low-signal aggregate view rather than the main story.</p>
    </section>

    <section class="section">
      <h2>Headline Readout</h2>
      {headline_cards}
    </section>

    <section class="section">
      <h2>What Was Compared</h2>
      <p class="interpretation-copy">This table uses the exact published bundle metrics for the five requested rows. CAID is intentionally left out of this page because the matching CAID SAM2 flip-avg row is still missing for the focused SAM2-vs-SAM3 story.</p>
      <div class="table-wrap" style="margin-top: 22px;">
        <table>
          <thead>
            <tr>
              <th>Route</th>
              <th>Dataset / Split</th>
              <th>Variant</th>
              <th>Model</th>
              <th>Evaluated Samples</th>
              <th>eval_miou</th>
              <th>eval_ari</th>
              <th>miou_agg</th>
            </tr>
          </thead>
          <tbody>
            {render_run_rows(run_rows)}
          </tbody>
        </table>
      </div>
      <div class="artifact-row">
        <a class="btn secondary" href="{table_paths['what_was_compared']}">Open Table CSV</a>
      </div>
    </section>

    <section class="section">
      <h2>Bigger-Picture Visuals</h2>
      <div class="chart-grid">
        <article class="chart-card">
          <img src="{plot_paths['eval_miou_grouped.svg']}" alt="Grouped eval_mIoU chart" />
        </article>
        <article class="chart-card">
          <img src="{plot_paths['eval_ari_grouped.svg']}" alt="Grouped eval_ARI chart" />
        </article>
      </div>

      <div class="delta-grid">
        {render_delta_cards(delta_rows)}
      </div>

      <div class="table-wrap" style="margin-top: 24px;">
        <table>
          <thead>
            <tr>
              <th>Comparison</th>
              <th>Improved Row</th>
              <th>SAM3 Baseline</th>
              <th>Δ mIoU</th>
              <th>Δ ARI</th>
              <th>Reading</th>
            </tr>
          </thead>
          <tbody>
            {render_delta_table(delta_rows)}
          </tbody>
        </table>
      </div>
      <div class="artifact-row">
        <a class="btn secondary" href="{table_paths['delta_vs_sam3']}">Open Delta CSV</a>
      </div>
    </section>

    <section class="section">
      <h2>Metric Caution</h2>
      {caution_panel}
    </section>

    <section class="section">
      <h2>Route-Level Takeaways</h2>
      {route_cards}
    </section>

    <section class="section">
      <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 20px; flex-wrap: wrap;">
        <div>
          <h2>Qualitative Comparison</h2>
          <p class="interpretation-copy">The panels below reuse the run-emitted triptychs. They are chosen to make the route-level story visible quickly rather than to present an exhaustive gallery.</p>
        </div>
        <a class="btn" href="gallery.html">Open Curated Gallery →</a>
      </div>
      <div style="margin-top: 24px;">
        {render_story_blocks(story_blocks, index_only=True)}
      </div>
    </section>

    <section class="section">
      <h2>Interpretation</h2>
      <p class="interpretation-copy">This does <strong>not</strong> mean SAM2 is the stronger model overall. It does suggest that, for this specific prompt-free frozen-feature clustering route, the SAM2 features are more cluster-friendly than the comparable SAM3 features. STLD is controlled and synthetic, so its advantage should not be over-weighted against RWTD, but the result is still strong enough to keep as a serious ablation and baseline.</p>
    </section>

    <section class="section">
      <h2>Caveats / Notes</h2>
      <ul class="bullet-list">
        <li>The report intentionally focuses on RWTD and STLD because that is the requested parity-closest slice for this page.</li>
        <li>On STLD, <code>miou_agg</code> is visually marked as low-signal because it is nearly constant across variants.</li>
        <li>CAID already has a new SAM2 coarse bundle on disk, but the matching CAID SAM2 flip-avg row is still missing, so CAID is not included here yet.</li>
        <li>This is a frozen-feature clustering comparison, not a claim about overall SAM2-vs-SAM3 project performance.</li>
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
        A lightweight qualitative subset built from the run-emitted visual triptychs. This gallery is intentionally curated to show the route-level story, not to publish every sample.
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


def render_summary_md(runs: dict[str, dict], delta_rows: list[dict]) -> str:
    return dedent(
        f"""\
        # {PAGE_TITLE}

        - Date: `{PAGE_DATE}`
        - Scope: RWTD and STLD frozen-feature automatic binary partitioning
        - RWTD: `rwtd_sam2_coarse` = `eval_miou={runs['rwtd_sam2_coarse']['eval_miou']:.6f}`, `eval_ari={runs['rwtd_sam2_coarse']['eval_ari']:.6f}` vs `rwtd_sam3_flip` = `eval_miou={runs['rwtd_sam3_flip']['eval_miou']:.6f}`, `eval_ari={runs['rwtd_sam3_flip']['eval_ari']:.6f}`
        - STLD: `stld_sam2_coarse` = `eval_miou={runs['stld_sam2_coarse']['eval_miou']:.6f}`, `eval_ari={runs['stld_sam2_coarse']['eval_ari']:.6f}`
        - STLD: `stld_sam2_flip` = `eval_miou={runs['stld_sam2_flip']['eval_miou']:.6f}`, `eval_ari={runs['stld_sam2_flip']['eval_ari']:.6f}`
        - STLD SAM3 baseline: `stld_sam3_flip` = `eval_miou={runs['stld_sam3_flip']['eval_miou']:.6f}`, `eval_ari={runs['stld_sam3_flip']['eval_ari']:.6f}`
        - Key deltas:
          - RWTD SAM2 coarse vs SAM3 flip: `Δ mIoU={delta_rows[0]['delta_miou']:.6f}`, `Δ ARI={delta_rows[0]['delta_ari']:.6f}`
          - STLD SAM2 coarse vs SAM3 flip: `Δ mIoU={delta_rows[1]['delta_miou']:.6f}`, `Δ ARI={delta_rows[1]['delta_ari']:.6f}`
          - STLD SAM2 flip vs SAM3 flip: `Δ mIoU={delta_rows[2]['delta_miou']:.6f}`, `Δ ARI={delta_rows[2]['delta_ari']:.6f}`
        - Caution: STLD `miou_agg` stays near `0.5`, so the page treats `eval_miou` and `eval_ari` as the main interpretable signal.
        """
    )


def main() -> None:
    runs = {spec.run_id: load_run(spec) for spec in RUN_SPECS}
    enforce_alignment(runs)

    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    run_rows = build_run_rows(runs)
    delta_rows = build_delta_rows(runs)
    story_blocks = build_story_blocks(runs)
    plot_paths = write_plot_assets(runs)
    table_paths = write_table_csvs(run_rows, delta_rows)
    metrics_payload = build_metrics_payload(runs, delta_rows)
    training_data = build_training_data(story_blocks, delta_rows)
    links_payload = build_links_payload(table_paths)

    write_text(DEST_DIR / "index.html", render_index_html(runs, run_rows, delta_rows, story_blocks, plot_paths, table_paths))
    write_text(DEST_DIR / "gallery.html", render_gallery_html(story_blocks))
    write_text(DEST_DIR / "metrics.json", json.dumps(metrics_payload, indent=2))
    write_text(DEST_DIR / "training_data.json", json.dumps(training_data, indent=2))
    write_text(DEST_DIR / "links.json", json.dumps(links_payload, indent=2))
    write_text(DEST_DIR / "summary.md", render_summary_md(runs, delta_rows))
    write_text(DEST_DIR / "manifest.yaml", build_manifest(story_blocks))


if __name__ == "__main__":
    main()
