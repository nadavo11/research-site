#!/usr/bin/env python3
"""Build the Stage-2 coarse-vs-fine SAM scale report page."""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from datetime import date
from html import escape
from pathlib import Path
from textwrap import dedent

from PIL import Image


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "experiments" / "stage2"
DEST_DIR = ROOT / "site" / "experiments" / "sam3-stage2-coarse-vs-fine-probe"

PAGE_TITLE = "Coarse-vs-Fine SAM Scales — Stage 2 Probe"
PAGE_SUBTITLE = "Experiment Report"
PAGE_DESCRIPTION = (
    "Stage-2 scale ablation on RWTD binary texture partitioning with a shared tiny probe. "
    "The readout is coarse-dominant: fpn_2 carries the main signal, fpn_1 is a useful "
    "complement, and fpn_0 is weak enough to be harmful in this setting."
)
PAGE_DATE = date.today().isoformat()
PRIMARY_METRIC = "eval_miou"
SECONDARY_METRIC = "eval_ari"

LEVEL_ORDER = {"fpn_2": 0, "fpn_1": 1, "fpn_0": 2}

VARIANT_COLORS = {
    "learned_global_gates_coarse_plus_next_finer": "#005f73",
    "fixed_uniform_gates_all_scales": "#0a9396",
    "learned_global_gates_all_scales": "#94d2bd",
    "fpn_2_only": "#ca6702",
    "fpn_1_only": "#ee9b00",
    "fpn_0_only_aligned": "#bb3e03",
    "fpn_0_only_native": "#9b2226",
}


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    label: str
    source_dir: str
    levels_label: str
    gate_type: str
    family: str
    table_note: str
    gallery_enabled: bool = True
    naming_note: str = ""


RUN_SPECS = [
    RunSpec(
        run_id="learned_global_gates_coarse_plus_next_finer",
        label="fpn_2 + fpn_1 learned gates",
        source_dir="train_learned_global_gates_coarse_plus_next_finer_test",
        levels_label="fpn_2 + fpn_1",
        gate_type="learned global softmax",
        family="best_subset",
        table_note="Best overall. Coarse signal stays dominant while fpn_1 adds complementary detail.",
    ),
    RunSpec(
        run_id="fixed_uniform_gates_all_scales",
        label="all scales, fixed uniform",
        source_dir="train_fixed_uniform_gates_all_scales_test",
        levels_label="fpn_2 + fpn_1 + fpn_0",
        gate_type="fixed uniform",
        family="all_scale_fusion",
        table_note="All-scale fusion helps, but the simpler uniform fusion still trails the best coarse+mid subset.",
    ),
    RunSpec(
        run_id="learned_global_gates_all_scales",
        label="all scales, learned gates",
        source_dir="smoke_learned_global_gates_all_scales",
        levels_label="fpn_2 + fpn_1 + fpn_0",
        gate_type="learned global softmax",
        family="all_scale_fusion",
        table_note="Learns coarse > mid > fine, but still underperforms fixed uniform all-scale fusion.",
        naming_note=(
            "The only extant full 227-sample bundle for this variant lives under "
            "`smoke_learned_global_gates_all_scales/`."
        ),
    ),
    RunSpec(
        run_id="fpn_2_only",
        label="fpn_2 only",
        source_dir="train_fpn_2_only_test",
        levels_label="fpn_2",
        gate_type="single-scale fixed",
        family="single_scale",
        table_note="Strongest single scale. Coarsest SAM features already recover most of the binary partition.",
    ),
    RunSpec(
        run_id="fpn_1_only",
        label="fpn_1 only",
        source_dir="train_fpn_1_only_test",
        levels_label="fpn_1",
        gate_type="single-scale fixed",
        family="single_scale",
        table_note="Weaker alone than fpn_2, but useful when paired with the coarsest scale.",
    ),
    RunSpec(
        run_id="fpn_0_only_aligned",
        label="fpn_0 only, aligned",
        source_dir="train_fpn_0_only_test",
        levels_label="fpn_0",
        gate_type="single-scale fixed",
        family="single_scale",
        table_note="Weak even after alignment down to the coarse 72x72 grid.",
    ),
    RunSpec(
        run_id="fpn_0_only_native",
        label="fpn_0 only, native",
        source_dir="smoke_train_fpn_0_only_native",
        levels_label="fpn_0",
        gate_type="single-scale fixed",
        family="native_control",
        table_note="Worst control. Native 288x288 fine-scale clustering degrades further instead of helping.",
        naming_note=(
            "The only extant full 227-sample native-grid control lives under "
            "`smoke_train_fpn_0_only_native/`."
        ),
    ),
]

RUN_SPEC_BY_ID = {spec.run_id: spec for spec in RUN_SPECS}

STORY_BASE_VARIANTS = [
    "fpn_2_only",
    "fpn_1_only",
    "fpn_0_only_aligned",
    "learned_global_gates_all_scales",
    "learned_global_gates_coarse_plus_next_finer",
]

GALLERY_VARIANTS = [spec.run_id for spec in RUN_SPECS if spec.gallery_enabled]

STORY_SAMPLE_PREFERENCES = {
    "coarse_mid_rescue": "350",
    "subset_choice": "357",
    "coarse_already_strong": "213",
    "native_fine_collapse": "210",
}


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path.relative_to(ROOT)}")
    return path


def load_json(path: Path) -> dict:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def level_sort_key(name: str) -> tuple[int, str]:
    return (LEVEL_ORDER.get(name, 99), name)


def format_weight_dict(weights: dict[str, float]) -> str:
    ordered = sorted(weights.items(), key=lambda item: level_sort_key(item[0]))
    return " · ".join(f"{name}={value:.3f}" for name, value in ordered)


def slugify_sample(sample_id: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in sample_id)


def load_visual_rows(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    manifest_path = require_file(run_dir / "visuals_manifest.jsonl")
    with manifest_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            visual_rel = Path(payload["visual_path"])
            visual_path = require_file(run_dir / visual_rel)
            crop_name = str(payload.get("crop_name") or visual_rel.stem)
            rows.append(
                {
                    "crop_name": crop_name,
                    "sample_index": int(payload.get("sample_index", len(rows))),
                    "eval_miou": float(payload["eval_miou"]),
                    "eval_ari": float(payload["eval_ari"]),
                    "caption": payload.get("caption", ""),
                    "final_gate_weights_compact": payload.get("final_gate_weights_compact", ""),
                    "coarsest_grid_height": int(payload.get("coarsest_grid_height", 0)),
                    "coarsest_grid_width": int(payload.get("coarsest_grid_width", 0)),
                    "source_visual_path": visual_path,
                }
            )
    rows.sort(key=lambda item: item["sample_index"])
    return rows


def infer_grid_label(rows: list[dict]) -> str:
    first = rows[0]
    return f"{first['coarsest_grid_width']}x{first['coarsest_grid_height']}"


def load_run(spec: RunSpec) -> dict:
    run_dir = SOURCE_ROOT / spec.source_dir
    require_file(run_dir / "summary.json")
    require_file(run_dir / "summary.csv")
    require_file(run_dir / "summary.md")
    require_file(run_dir / "config.json")
    require_file(run_dir / "experiment_terms.md")
    require_file(run_dir / "per_sample_metrics.csv")
    require_file(run_dir / "per_sample_metrics.jsonl")
    rows = load_visual_rows(run_dir)
    summary = load_json(run_dir / "summary.json")
    metrics = summary["mean_metrics"]
    weights = {key: float(value) for key, value in summary["final_gate_weights"].items()}
    sample_lookup = {row["crop_name"]: row for row in rows}
    return {
        "run_id": spec.run_id,
        "label": spec.label,
        "source_dir": run_dir,
        "levels_label": spec.levels_label,
        "gate_type": spec.gate_type,
        "family": spec.family,
        "table_note": spec.table_note,
        "naming_note": spec.naming_note,
        "grid_label": infer_grid_label(rows),
        "weights": weights,
        "weights_compact": format_weight_dict(weights),
        "mean_miou": float(metrics["eval_miou"]),
        "mean_ari": float(metrics["eval_ari"]),
        "num_samples": int(summary["num_evaluated_samples"]),
        "checkpoint_path": summary.get("checkpoint_path") or "",
        "dataset_id": summary.get("dataset_id") or "",
        "model_id": summary.get("model_id") or "",
        "generated_at_utc": summary.get("generated_at_utc") or "",
        "learnable_gates": bool(summary.get("learnable_gates")),
        "selected_level_names": list(summary.get("selected_level_names", [])),
        "rows": rows,
        "sample_lookup": sample_lookup,
        "summary_json": summary,
    }


def enforce_sample_alignment(runs: dict[str, dict]) -> list[str]:
    sample_sets = {run_id: set(run["sample_lookup"].keys()) for run_id, run in runs.items()}
    baseline = sample_sets["fpn_2_only"]
    mismatches = {
        run_id: sorted(baseline.symmetric_difference(sample_ids))
        for run_id, sample_ids in sample_sets.items()
        if sample_ids != baseline
    }
    if mismatches:
        details = ", ".join(f"{run_id}: {len(diff)} mismatched samples" for run_id, diff in mismatches.items())
        raise RuntimeError(f"Stage-2 sample alignment mismatch detected: {details}")
    return sorted(baseline, key=lambda sample_id: runs["fpn_2_only"]["sample_lookup"][sample_id]["sample_index"])


def pick_unique_sample(candidates: list[str], used: set[str]) -> str:
    for sample_id in candidates:
        if sample_id not in used:
            used.add(sample_id)
            return sample_id
    raise RuntimeError("Could not choose a unique story sample from candidate set.")


def pick_story_sample(story_id: str, candidates: list[str], used: set[str]) -> str:
    preferred = STORY_SAMPLE_PREFERENCES.get(story_id)
    if preferred and preferred not in used and preferred in candidates:
        used.add(preferred)
        return preferred
    return pick_unique_sample(candidates, used)


def choose_story_samples(runs: dict[str, dict], sample_ids: list[str]) -> list[dict]:
    used: set[str] = set()

    best = runs["learned_global_gates_coarse_plus_next_finer"]["sample_lookup"]
    fpn2 = runs["fpn_2_only"]["sample_lookup"]
    fpn1 = runs["fpn_1_only"]["sample_lookup"]
    fpn0_aligned = runs["fpn_0_only_aligned"]["sample_lookup"]
    fpn0_native = runs["fpn_0_only_native"]["sample_lookup"]
    learned_all = runs["learned_global_gates_all_scales"]["sample_lookup"]
    uniform_all = runs["fixed_uniform_gates_all_scales"]["sample_lookup"]

    rescue_candidates = sorted(
        sample_ids,
        key=lambda sample_id: best[sample_id]["eval_miou"] - fpn2[sample_id]["eval_miou"],
        reverse=True,
    )
    subset_candidates = sorted(
        sample_ids,
        key=lambda sample_id: uniform_all[sample_id]["eval_miou"] - learned_all[sample_id]["eval_miou"],
        reverse=True,
    )
    coarse_dominant_candidates = sorted(
        [
            sample_id
            for sample_id in sample_ids
            if fpn2[sample_id]["eval_miou"] >= 0.97
            and abs(best[sample_id]["eval_miou"] - fpn2[sample_id]["eval_miou"]) <= 0.01
            and fpn1[sample_id]["eval_miou"] <= fpn2[sample_id]["eval_miou"] - 0.10
        ],
        key=lambda sample_id: (fpn2[sample_id]["eval_miou"], -abs(best[sample_id]["eval_miou"] - fpn2[sample_id]["eval_miou"])),
        reverse=True,
    )
    native_collapse_candidates = sorted(
        [
            sample_id
            for sample_id in sample_ids
            if fpn2[sample_id]["eval_miou"] >= 0.90
        ],
        key=lambda sample_id: best[sample_id]["eval_miou"] - fpn0_native[sample_id]["eval_miou"],
        reverse=True,
    )

    rescue_sample = pick_story_sample("coarse_mid_rescue", rescue_candidates, used)
    subset_sample = pick_story_sample("subset_choice", subset_candidates, used)
    coarse_sample = pick_story_sample("coarse_already_strong", coarse_dominant_candidates, used)
    native_sample = pick_story_sample("native_fine_collapse", native_collapse_candidates, used)

    return [
        {
            "story_id": "coarse_mid_rescue",
            "title": "Mid-Scale Rescue on a Coarse Boundary",
            "sample_id": rescue_sample,
            "tag": "complementarity",
            "summary": (
                "fpn_2 misses the narrow bark-versus-moss seam here. fpn_1 alone is noisy, "
                "but adding it to fpn_2 restores the thin vertical partition cleanly."
            ),
            "variant_order": STORY_BASE_VARIANTS,
        },
        {
            "story_id": "subset_choice",
            "title": "Subset Choice Beats All-Scale Learned Weighting",
            "sample_id": subset_sample,
            "tag": "subset matters",
            "summary": (
                "The learned all-scale gate already favors coarse > mid > fine, yet the explicit "
                "coarse+mid subset still edges it out. Even fixed-uniform all-scale fusion beats the learned all-scale variant here."
            ),
            "variant_order": [
                "fpn_2_only",
                "fixed_uniform_gates_all_scales",
                "learned_global_gates_all_scales",
                "learned_global_gates_coarse_plus_next_finer",
            ],
        },
        {
            "story_id": "coarse_already_strong",
            "title": "Coarse Signal Already Carries the Partition",
            "sample_id": coarse_sample,
            "tag": "coarse dominant",
            "summary": (
                "On this crop the coarsest scale is already nearly optimal. The best two-scale fusion "
                "barely moves the score, while the finer-only controls are still distinctly weaker."
            ),
            "variant_order": STORY_BASE_VARIANTS,
        },
        {
            "story_id": "native_fine_collapse",
            "title": "Fine-Scale Native Resolution Collapses",
            "sample_id": native_sample,
            "tag": "fine failure",
            "summary": (
                "The finest scale is weak even after alignment to 72x72, and becomes worse at its native "
                "288x288 grid. Coarse features still recover the semantic partition cleanly."
            ),
            "variant_order": STORY_BASE_VARIANTS + ["fpn_0_only_native"],
        },
    ]


def write_thumbnail(source_path: Path, dest_path: Path, width: int, quality: int) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        image = image.convert("RGB")
        scale = width / image.width
        resized = image.resize((width, max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
        resized.save(dest_path, format="WEBP", quality=quality, method=6)


def create_story_assets(runs: dict[str, dict], story_samples: list[dict]) -> list[dict]:
    created: list[dict] = []
    for story in story_samples:
        sample_id = story["sample_id"]
        story["images"] = []
        for variant_id in story["variant_order"]:
            run = runs[variant_id]
            row = run["sample_lookup"][sample_id]
            asset_name = f"{story['story_id']}__{variant_id}.webp"
            asset_path = DEST_DIR / "assets" / "gallery" / asset_name
            write_thumbnail(row["source_visual_path"], asset_path, width=640, quality=82)
            story["images"].append(
                {
                    "variant_id": variant_id,
                    "variant_label": RUN_SPEC_BY_ID[variant_id].label,
                    "asset_rel": f"assets/gallery/{asset_name}",
                    "miou": row["eval_miou"],
                    "ari": row["eval_ari"],
                }
            )
            created.append({"story_id": story["story_id"], "variant_id": variant_id, "asset_rel": asset_name})
    return created


def create_qc_assets(runs: dict[str, dict], story_samples: list[dict]) -> list[dict]:
    qc_specs = [
        ("fine_native_warning", story_samples[-1]["sample_id"], "fpn_0_only_native", "Worst-case native fine-scale control"),
        ("subset_gap_warning", story_samples[1]["sample_id"], "learned_global_gates_all_scales", "Learned all-scale lagging a simpler alternative"),
        ("coarse_rescue_warning", story_samples[0]["sample_id"], "fpn_2_only", "Coarse-only under-segmentation before the fpn_1 complement"),
    ]
    created: list[dict] = []
    for asset_stub, sample_id, variant_id, caption in qc_specs:
        row = runs[variant_id]["sample_lookup"][sample_id]
        asset_name = f"{asset_stub}.webp"
        asset_path = DEST_DIR / "assets" / "qc" / asset_name
        write_thumbnail(row["source_visual_path"], asset_path, width=420, quality=78)
        created.append(
            {
                "asset_rel": f"assets/qc/{asset_name}",
                "caption": caption,
                "sample_id": sample_id,
                "variant_label": RUN_SPEC_BY_ID[variant_id].label,
                "miou": row["eval_miou"],
                "ari": row["eval_ari"],
            }
        )
    return created


def create_gallery_previews(runs: dict[str, dict], sample_ids: list[str]) -> list[dict]:
    records: list[dict] = []
    for variant_id in GALLERY_VARIANTS:
        run = runs[variant_id]
        for sample_id in sample_ids:
            row = run["sample_lookup"][sample_id]
            sample_slug = slugify_sample(sample_id)
            asset_rel = Path("assets") / "all_previews" / variant_id / f"{sample_slug}.webp"
            write_thumbnail(row["source_visual_path"], DEST_DIR / asset_rel, width=360, quality=76)
            delta_vs_fpn2 = row["eval_miou"] - runs["fpn_2_only"]["sample_lookup"][sample_id]["eval_miou"]
            records.append(
                {
                    "sample_id": sample_id,
                    "sample_index": row["sample_index"],
                    "variant_id": variant_id,
                    "variant_label": RUN_SPEC_BY_ID[variant_id].label,
                    "family": RUN_SPEC_BY_ID[variant_id].family,
                    "miou": row["eval_miou"],
                    "ari": row["eval_ari"],
                    "delta_vs_fpn2": delta_vs_fpn2,
                    "thumb_path": asset_rel.as_posix(),
                }
            )
    return records


def copy_repro_bundle(spec: RunSpec) -> dict:
    source_dir = SOURCE_ROOT / spec.source_dir
    dest_dir = DEST_DIR / "repro" / spec.run_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    for file_name in [
        "config.json",
        "experiment_terms.md",
        "summary.json",
        "summary.csv",
        "summary.md",
        "visuals_manifest.jsonl",
        "per_sample_metrics.csv",
        "per_sample_metrics.jsonl",
    ]:
        shutil.copy2(source_dir / file_name, dest_dir / file_name)
    return {
        "run_id": spec.run_id,
        "label": spec.label,
        "source_dir": source_dir.relative_to(ROOT).as_posix(),
        "repro_dir": dest_dir.relative_to(DEST_DIR).as_posix(),
    }


def metric_delta(a: dict[str, float], b: dict[str, float], key: str) -> float:
    return a[key] - b[key]


def render_horizontal_bar_chart(title: str, subtitle: str, items: list[dict], metric_key: str, value_format: str = "{:.4f}") -> str:
    width = 860
    row_height = 42
    top = 80
    left = 210
    right = 120
    chart_width = width - left - right
    height = top + row_height * len(items) + 40
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        '<rect width="100%" height="100%" fill="#ffffff" rx="24" />',
        f'<text x="32" y="38" font-family="Space Grotesk, sans-serif" font-size="24" font-weight="700" fill="#122033">{escape(title)}</text>',
        f'<text x="32" y="62" font-family="Space Grotesk, sans-serif" font-size="13" fill="#4c5d73">{escape(subtitle)}</text>',
    ]
    for tick in [0.0, 0.25, 0.50, 0.75, 1.0]:
        x = left + chart_width * tick
        lines.append(f'<line x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" y2="{height - 24}" stroke="#e3ebf2" stroke-width="1" />')
        lines.append(
            f'<text x="{x:.1f}" y="{height - 8}" text-anchor="middle" font-family="Space Grotesk, sans-serif" '
            f'font-size="12" fill="#4c5d73">{tick:.2f}</text>'
        )
    for index, item in enumerate(items):
        y = top + index * row_height
        value = float(item[metric_key])
        bar_width = chart_width * max(0.0, min(1.0, value))
        color = VARIANT_COLORS[item["run_id"]]
        label = escape(item["label"].replace("`", ""))
        value_text = escape(value_format.format(value))
        lines.extend(
            [
                f'<text x="32" y="{y + 20}" font-family="Space Grotesk, sans-serif" font-size="13" font-weight="600" fill="#122033">{label}</text>',
                f'<text x="32" y="{y + 34}" font-family="Space Grotesk, sans-serif" font-size="11" fill="#4c5d73">{escape(item["levels_label"].replace("`", ""))}</text>',
                f'<rect x="{left}" y="{y + 6}" width="{chart_width}" height="16" rx="8" fill="#edf3f7" />',
                f'<rect x="{left}" y="{y + 6}" width="{bar_width:.1f}" height="16" rx="8" fill="{color}" />',
                f'<text x="{left + chart_width + 12}" y="{y + 19}" font-family="Space Grotesk, sans-serif" font-size="12" font-weight="700" fill="#122033">{value_text}</text>',
            ]
        )
    lines.append("</svg>")
    return "\n".join(lines)


def render_single_scale_chart(items: list[dict]) -> str:
    width = 860
    height = 420
    left = 130
    chart_width = 250
    row_height = 64
    top = 84
    gap = 130
    second_left = left + chart_width + gap
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Single-scale comparison">',
        '<rect width="100%" height="100%" fill="#ffffff" rx="24" />',
        '<text x="32" y="38" font-family="Space Grotesk, sans-serif" font-size="24" font-weight="700" fill="#122033">Single-Scale Readout</text>',
        '<text x="32" y="62" font-family="Space Grotesk, sans-serif" font-size="13" fill="#4c5d73">The ranking is clear: fpn_2 >> fpn_1 >> fpn_0, and the native-grid fine control is worst.</text>',
        f'<text x="{left}" y="78" font-family="Space Grotesk, sans-serif" font-size="13" font-weight="700" fill="#005f73">Mean mIoU</text>',
        f'<text x="{second_left}" y="78" font-family="Space Grotesk, sans-serif" font-size="13" font-weight="700" fill="#005f73">Mean ARI</text>',
    ]
    for tick in [0.0, 0.25, 0.50, 0.75, 1.0]:
        x1 = left + chart_width * tick
        x2 = second_left + chart_width * tick
        lines.extend(
            [
                f'<line x1="{x1:.1f}" y1="{top - 6}" x2="{x1:.1f}" y2="{height - 32}" stroke="#e3ebf2" stroke-width="1" />',
                f'<line x1="{x2:.1f}" y1="{top - 6}" x2="{x2:.1f}" y2="{height - 32}" stroke="#e3ebf2" stroke-width="1" />',
            ]
        )
    for index, item in enumerate(items):
        y = top + index * row_height
        color = VARIANT_COLORS[item["run_id"]]
        miou_width = chart_width * item["mean_miou"]
        ari_width = chart_width * item["mean_ari"]
        label = escape(item["label"].replace("`", ""))
        lines.extend(
            [
                f'<text x="32" y="{y + 16}" font-family="Space Grotesk, sans-serif" font-size="13" font-weight="600" fill="#122033">{label}</text>',
                f'<text x="32" y="{y + 34}" font-family="Space Grotesk, sans-serif" font-size="11" fill="#4c5d73">{escape(item["grid_label"])}</text>',
                f'<rect x="{left}" y="{y}" width="{chart_width}" height="14" rx="7" fill="#edf3f7" />',
                f'<rect x="{left}" y="{y}" width="{miou_width:.1f}" height="14" rx="7" fill="{color}" />',
                f'<text x="{left + chart_width + 10}" y="{y + 11}" font-family="Space Grotesk, sans-serif" font-size="11" font-weight="700" fill="#122033">{item["mean_miou"]:.4f}</text>',
                f'<rect x="{second_left}" y="{y}" width="{chart_width}" height="14" rx="7" fill="#edf3f7" />',
                f'<rect x="{second_left}" y="{y}" width="{ari_width:.1f}" height="14" rx="7" fill="{color}" />',
                f'<text x="{second_left + chart_width + 10}" y="{y + 11}" font-family="Space Grotesk, sans-serif" font-size="11" font-weight="700" fill="#122033">{item["mean_ari"]:.4f}</text>',
            ]
        )
    lines.append("</svg>")
    return "\n".join(lines)


def render_gate_chart(title: str, subtitle: str, weights: dict[str, float], emphasis_variant: str) -> str:
    ordered = sorted(weights.items(), key=lambda item: level_sort_key(item[0]))
    width = 560
    height = 220 + len(ordered) * 28
    left = 120
    chart_width = 360
    color = VARIANT_COLORS[emphasis_variant]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        '<rect width="100%" height="100%" fill="#ffffff" rx="24" />',
        f'<text x="28" y="38" font-family="Space Grotesk, sans-serif" font-size="22" font-weight="700" fill="#122033">{escape(title)}</text>',
        f'<text x="28" y="60" font-family="Space Grotesk, sans-serif" font-size="12" fill="#4c5d73">{escape(subtitle)}</text>',
    ]
    for idx, tick in enumerate([0.0, 0.25, 0.50, 0.75, 1.0]):
        x = left + chart_width * tick
        lines.append(f'<line x1="{x:.1f}" y1="76" x2="{x:.1f}" y2="{height - 32}" stroke="#e3ebf2" stroke-width="1" />')
        lines.append(
            f'<text x="{x:.1f}" y="{height - 10}" text-anchor="middle" font-family="Space Grotesk, sans-serif" font-size="11" fill="#4c5d73">{tick:.2f}</text>'
        )
    for index, (name, value) in enumerate(ordered):
        y = 96 + index * 34
        lines.extend(
            [
                f'<text x="28" y="{y + 12}" font-family="Space Grotesk, sans-serif" font-size="13" font-weight="600" fill="#122033">{escape(name)}</text>',
                f'<rect x="{left}" y="{y}" width="{chart_width}" height="16" rx="8" fill="#edf3f7" />',
                f'<rect x="{left}" y="{y}" width="{chart_width * value:.1f}" height="16" rx="8" fill="{color}" />',
                f'<text x="{left + chart_width + 12}" y="{y + 12}" font-family="Space Grotesk, sans-serif" font-size="12" font-weight="700" fill="#122033">{value:.3f}</text>',
            ]
        )
    lines.append("</svg>")
    return "\n".join(lines)


def histogram(values: list[float], bins: int) -> tuple[list[int], float, float]:
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return [len(values)], lo, hi + 1.0
    counts = [0 for _ in range(bins)]
    width = (hi - lo) / bins
    for value in values:
        bucket = min(bins - 1, int((value - lo) / width))
        counts[bucket] += 1
    return counts, lo, hi


def render_histogram_chart(title: str, subtitle: str, values: list[float], highlight_color: str, mean_label: str) -> str:
    counts, lo, hi = histogram(values, bins=18)
    width = 860
    height = 320
    left = 72
    chart_width = 740
    chart_height = 170
    bottom = 250
    max_count = max(counts) if counts else 1
    mean_value = sum(values) / len(values)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        '<rect width="100%" height="100%" fill="#ffffff" rx="24" />',
        f'<text x="32" y="38" font-family="Space Grotesk, sans-serif" font-size="24" font-weight="700" fill="#122033">{escape(title)}</text>',
        f'<text x="32" y="62" font-family="Space Grotesk, sans-serif" font-size="13" fill="#4c5d73">{escape(subtitle)}</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{left + chart_width}" y2="{bottom}" stroke="#cfd9e4" stroke-width="1.5" />',
        f'<line x1="{left}" y1="{bottom - chart_height}" x2="{left}" y2="{bottom}" stroke="#cfd9e4" stroke-width="1.5" />',
    ]
    if counts:
        bar_gap = 4
        total_bar_width = chart_width / len(counts)
        for index, count in enumerate(counts):
            x = left + index * total_bar_width + bar_gap / 2
            bar_width = total_bar_width - bar_gap
            bar_height = 0 if max_count == 0 else chart_height * (count / max_count)
            y = bottom - bar_height
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="4" fill="{highlight_color}" opacity="0.92" />'
            )
        mean_x = left + chart_width * ((mean_value - lo) / (hi - lo if not math.isclose(lo, hi) else 1.0))
        lines.extend(
            [
                f'<line x1="{mean_x:.1f}" y1="{bottom - chart_height - 6}" x2="{mean_x:.1f}" y2="{bottom + 4}" stroke="#122033" stroke-width="2" stroke-dasharray="6 4" />',
                f'<text x="{mean_x:.1f}" y="{bottom - chart_height - 12}" text-anchor="middle" font-family="Space Grotesk, sans-serif" font-size="12" font-weight="700" fill="#122033">{escape(mean_label)} {mean_value:+.4f}</text>',
            ]
        )
    lines.extend(
        [
            f'<text x="{left}" y="{bottom + 22}" font-family="Space Grotesk, sans-serif" font-size="12" fill="#4c5d73">{lo:+.3f}</text>',
            f'<text x="{left + chart_width}" y="{bottom + 22}" text-anchor="end" font-family="Space Grotesk, sans-serif" font-size="12" fill="#4c5d73">{hi:+.3f}</text>',
            f'<text x="{left}" y="{bottom - chart_height - 12}" font-family="Space Grotesk, sans-serif" font-size="12" fill="#4c5d73">count</text>',
        ]
    )
    lines.append("</svg>")
    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_plot_assets(runs: dict[str, dict]) -> dict[str, str]:
    ranked = rank_runs(runs)
    single_scale = [runs[run_id] for run_id in ["fpn_2_only", "fpn_1_only", "fpn_0_only_aligned", "fpn_0_only_native"]]
    plots = {
        "mean_miou_by_run.svg": render_horizontal_bar_chart(
            title="Mean mIoU by Run",
            subtitle="The coarse+next-finer learned probe is the current leader, with fpn_2 the best single scale.",
            items=ranked,
            metric_key="mean_miou",
        ),
        "mean_ari_by_run.svg": render_horizontal_bar_chart(
            title="Mean ARI by Run",
            subtitle="ARI tells the same story as mIoU: coarse+mid wins, all-scale learned gating does not.",
            items=ranked,
            metric_key="mean_ari",
        ),
        "single_scale_comparison.svg": render_single_scale_chart(single_scale),
        "gates_all_scales.svg": render_gate_chart(
            title="All-Scale Learned Gates",
            subtitle="The learned ordering is coarse > mid > fine, but this still loses to fixed uniform all-scale fusion.",
            weights=runs["learned_global_gates_all_scales"]["weights"],
            emphasis_variant="learned_global_gates_all_scales",
        ),
        "gates_coarse_plus_next_finer.svg": render_gate_chart(
            title="Coarse + Next-Finer Gates",
            subtitle="The winning two-scale probe still leans coarse, but keeps a substantial mid-scale contribution.",
            weights=runs["learned_global_gates_coarse_plus_next_finer"]["weights"],
            emphasis_variant="learned_global_gates_coarse_plus_next_finer",
        ),
        "delta_best_vs_fpn2.svg": render_histogram_chart(
            title="Per-Sample Delta: Best Run vs fpn_2",
            subtitle="Positive values mean the coarse+mid learned probe improved over the best single scale.",
            values=[
                runs["learned_global_gates_coarse_plus_next_finer"]["sample_lookup"][sample_id]["eval_miou"]
                - runs["fpn_2_only"]["sample_lookup"][sample_id]["eval_miou"]
                for sample_id in runs["fpn_2_only"]["sample_lookup"]
            ],
            highlight_color=VARIANT_COLORS["learned_global_gates_coarse_plus_next_finer"],
            mean_label="mean Δ",
        ),
        "delta_fpn0_native_vs_aligned.svg": render_histogram_chart(
            title="Per-Sample Delta: fpn_0 Native vs Aligned",
            subtitle="Negative values dominate: native fine-scale clustering is generally worse than the aligned control.",
            values=[
                runs["fpn_0_only_native"]["sample_lookup"][sample_id]["eval_miou"]
                - runs["fpn_0_only_aligned"]["sample_lookup"][sample_id]["eval_miou"]
                for sample_id in runs["fpn_2_only"]["sample_lookup"]
            ],
            highlight_color=VARIANT_COLORS["fpn_0_only_native"],
            mean_label="mean Δ",
        ),
    }
    for file_name, content in plots.items():
        write_text(DEST_DIR / "assets" / "plots" / file_name, content)
    return {name: f"assets/plots/{name}" for name in plots}


def rank_runs(runs: dict[str, dict]) -> list[dict]:
    return sorted(runs.values(), key=lambda run: (run["mean_miou"], run["mean_ari"]), reverse=True)


def build_summary_rows(runs: dict[str, dict]) -> list[dict]:
    rows = []
    for rank, run in enumerate(rank_runs(runs), start=1):
        rows.append(
            {
                "rank": rank,
                "run_id": run["run_id"],
                "variant": run["label"],
                "levels_label": run["levels_label"],
                "grid_label": run["grid_label"],
                "gate_type": run["gate_type"],
                "weights_compact": run["weights_compact"],
                "mean_miou": run["mean_miou"],
                "mean_ari": run["mean_ari"],
                "note": run["table_note"],
            }
        )
    return rows


def write_summary_table_csv(rows: list[dict]) -> str:
    path = DEST_DIR / "assets" / "tables" / "summary_table.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "run_id",
                "variant",
                "levels_label",
                "grid_label",
                "gate_type",
                "weights_compact",
                "mean_miou",
                "mean_ari",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path.relative_to(DEST_DIR).as_posix()


def build_manifest(runs: dict[str, dict], gallery_count: int, qc_count: int, all_previews_count: int) -> str:
    return dedent(
        f"""\
        title: "{PAGE_TITLE}"
        model_id: "facebook/sam3"
        dataset_id: "aviadcohz/RWTD"
        split: "test"
        date: "{PAGE_DATE}"
        description: "Stage-2 scale ablation for SAM 3 binary texture partitioning on RWTD."
        status: "reviewed"
        assets:
          gallery_count: {gallery_count}
          qc_count: {qc_count}
          all_previews_count: {all_previews_count}
        evaluation:
          primary_metric: "{PRIMARY_METRIC}"
          secondary_metric: "{SECONDARY_METRIC}"
          metrics_file: "metrics.json"
        """
    )


def build_metrics_payload(runs: dict[str, dict], summary_rows: list[dict]) -> dict:
    best_run = runs["learned_global_gates_coarse_plus_next_finer"]
    best_single = runs["fpn_2_only"]
    all_scale_learned = runs["learned_global_gates_all_scales"]
    uniform_all = runs["fixed_uniform_gates_all_scales"]
    fine_aligned = runs["fpn_0_only_aligned"]
    fine_native = runs["fpn_0_only_native"]
    return {
        "status": "reviewed",
        "primary_metric": PRIMARY_METRIC,
        "secondary_metric": SECONDARY_METRIC,
        "best_variant": {
            "run_id": best_run["run_id"],
            "label": best_run["label"],
            "mean_miou": best_run["mean_miou"],
            "mean_ari": best_run["mean_ari"],
            "weights": best_run["weights"],
        },
        "best_single_scale": {
            "run_id": best_single["run_id"],
            "label": best_single["label"],
            "mean_miou": best_single["mean_miou"],
            "mean_ari": best_single["mean_ari"],
        },
        "all_scale_learned_gate_order": all_scale_learned["weights"],
        "all_scale_uniform_minus_learned": {
            "delta_miou": uniform_all["mean_miou"] - all_scale_learned["mean_miou"],
            "delta_ari": uniform_all["mean_ari"] - all_scale_learned["mean_ari"],
        },
        "fine_native_minus_aligned": {
            "delta_miou": fine_native["mean_miou"] - fine_aligned["mean_miou"],
            "delta_ari": fine_native["mean_ari"] - fine_aligned["mean_ari"],
        },
        "runs": [
            {
                "run_id": row["run_id"],
                "label": row["variant"],
                "mean_miou": row["mean_miou"],
                "mean_ari": row["mean_ari"],
                "grid": row["grid_label"],
                "weights": runs[row["run_id"]]["weights"],
            }
            for row in summary_rows
        ],
    }


def build_training_data(
    runs: dict[str, dict],
    summary_rows: list[dict],
    story_samples: list[dict],
    gallery_records: list[dict],
    qc_assets: list[dict],
    repro_mapping: list[dict],
) -> dict:
    return {
        "page": {
            "title": PAGE_TITLE,
            "date": PAGE_DATE,
            "status": "reviewed",
            "sample_count": runs["fpn_2_only"]["num_samples"],
        },
        "runs": [
            {
                "run_id": row["run_id"],
                "label": row["variant"],
                "family": RUN_SPEC_BY_ID[row["run_id"]].family,
                "mean_miou": row["mean_miou"],
                "mean_ari": row["mean_ari"],
                "grid_label": row["grid_label"],
            }
            for row in summary_rows
        ],
        "story_samples": [
            {
                "story_id": story["story_id"],
                "title": story["title"],
                "sample_id": story["sample_id"],
                "tag": story["tag"],
                "summary": story["summary"],
                "images": story["images"],
            }
            for story in story_samples
        ],
        "gallery": {
            "sample_count": runs["fpn_2_only"]["num_samples"],
            "variants": [
                {"id": run_id, "label": RUN_SPEC_BY_ID[run_id].label, "family": RUN_SPEC_BY_ID[run_id].family}
                for run_id in GALLERY_VARIANTS
            ],
            "records": gallery_records,
        },
        "qc": qc_assets,
        "repro": repro_mapping,
    }


def signed(value: float) -> str:
    return f"{value:+.4f}"


def render_summary_table(summary_rows: list[dict]) -> str:
    body_rows = []
    for row in summary_rows:
        highlight = '<span class="tag good">best</span>' if row["rank"] == 1 else ""
        body_rows.append(
            dedent(
                f"""\
                <tr>
                  <td><strong>{row['rank']}</strong> {highlight}</td>
                  <td><strong>{escape(row['variant'])}</strong></td>
                  <td><code>{escape(row['levels_label'])}</code></td>
                  <td>{escape(row['grid_label'])}</td>
                  <td>{escape(row['gate_type'])}</td>
                  <td><code>{escape(row['weights_compact'])}</code></td>
                  <td><strong>{row['mean_miou']:.4f}</strong></td>
                  <td><strong>{row['mean_ari']:.4f}</strong></td>
                  <td>{escape(row['note'])}</td>
                </tr>
                """
            ).strip()
        )
    return "\n".join(body_rows)


def render_story_blocks(story_samples: list[dict]) -> str:
    blocks = []
    for story in story_samples:
        figures = []
        for image in story["images"]:
            figures.append(
                dedent(
                    f"""\
                    <figure>
                      <img src="{escape(image['asset_rel'])}" alt="{escape(story['title'])} — {escape(image['variant_label'])}" data-zoom-src="{escape(image['asset_rel'])}" />
                      <figcaption>
                        <strong>{escape(image['variant_label'])}</strong>
                        <div class="mini-note">mIoU {image['miou']:.4f} · ARI {image['ari']:.4f}</div>
                      </figcaption>
                    </figure>
                    """
                ).strip()
            )
        blocks.append(
            dedent(
                f"""\
                <article class="story-block">
                  <div class="story-head">
                    <span class="tag good">{escape(story['tag'])}</span>
                    <span class="pill">crop {escape(story['sample_id'])}</span>
                  </div>
                  <h3>{escape(story['title'])}</h3>
                  <p class="story-summary">{escape(story['summary'])}</p>
                  <div class="story-grid">
                    {' '.join(figures)}
                  </div>
                </article>
                """
            ).strip()
        )
    return "\n".join(blocks)


def render_qc_cards(qc_assets: list[dict]) -> str:
    cards = []
    for item in qc_assets:
        cards.append(
            dedent(
                f"""\
                <figure>
                  <img src="{escape(item['asset_rel'])}" alt="{escape(item['caption'])}" data-zoom-src="{escape(item['asset_rel'])}" />
                  <figcaption>
                    <strong>{escape(item['caption'])}</strong>
                    <div class="mini-note">crop {escape(item['sample_id'])} · {escape(item['variant_label'])} · mIoU {item['miou']:.4f} · ARI {item['ari']:.4f}</div>
                  </figcaption>
                </figure>
                """
            ).strip()
        )
    return "\n".join(cards)


def render_index_html(
    runs: dict[str, dict],
    summary_rows: list[dict],
    story_samples: list[dict],
    qc_assets: list[dict],
    plot_paths: dict[str, str],
    links_payload: dict,
    git_commit: str,
    git_author: str,
) -> str:
    best_run = runs["learned_global_gates_coarse_plus_next_finer"]
    best_single = runs["fpn_2_only"]
    all_scale_learned = runs["learned_global_gates_all_scales"]
    uniform_all = runs["fixed_uniform_gates_all_scales"]
    fine_aligned = runs["fpn_0_only_aligned"]
    fine_native = runs["fpn_0_only_native"]

    top_cards = dedent(
        f"""\
        <div class="metric-grid compact-grid">
          <article class="metric">
            <div class="k">Best Overall</div>
            <div class="v">{best_run['mean_miou']:.4f}</div>
            <div class="mini-note"><code>fpn_2 + fpn_1</code> learned · ARI {best_run['mean_ari']:.4f}</div>
          </article>
          <article class="metric">
            <div class="k">Best Single Scale</div>
            <div class="v">{best_single['mean_miou']:.4f}</div>
            <div class="mini-note"><code>fpn_2</code> only · ARI {best_single['mean_ari']:.4f}</div>
          </article>
          <article class="metric">
            <div class="k">All-Scale Learned Gates</div>
            <div class="v gate-readout">0.430 / 0.315 / 0.256</div>
            <div class="mini-note"><code>fpn_2 &gt; fpn_1 &gt; fpn_0</code></div>
          </article>
          <article class="metric">
            <div class="k">Fine Native Penalty</div>
            <div class="v delta-negative">{signed(fine_native['mean_miou'] - fine_aligned['mean_miou'])}</div>
            <div class="mini-note">mIoU vs aligned <code>fpn_0</code> · ARI {signed(fine_native['mean_ari'] - fine_aligned['mean_ari'])}</div>
          </article>
        </div>
        """
    ).strip()

    plot_cards = dedent(
        f"""\
        <div class="chart-grid">
          <article class="chart-card">
            <img src="{plot_paths['mean_miou_by_run.svg']}" alt="Mean mIoU by run" />
          </article>
          <article class="chart-card">
            <img src="{plot_paths['mean_ari_by_run.svg']}" alt="Mean ARI by run" />
          </article>
          <article class="chart-card chart-card-wide">
            <img src="{plot_paths['single_scale_comparison.svg']}" alt="Single-scale comparison" />
          </article>
          <article class="chart-card">
            <img src="{plot_paths['gates_all_scales.svg']}" alt="Learned gate weights across all scales" />
          </article>
          <article class="chart-card">
            <img src="{plot_paths['gates_coarse_plus_next_finer.svg']}" alt="Learned gate weights for coarse plus next-finer" />
          </article>
        </div>
        """
    ).strip()

    provenance_grid = dedent(
        f"""\
        <div class="identity-grid">
          <article class="card">
            <div class="k">Run Date</div>
            <div class="identity-value">{PAGE_DATE}</div>
            <div class="mini-note">Rendered from Stage-2 bundles dated 2026-03-24 to 2026-03-25.</div>
          </article>
          <article class="card">
            <div class="k">Git Commit</div>
            <div class="identity-value"><code>{escape(git_commit)}</code></div>
            <div class="mini-note">Author: {escape(git_author)}</div>
          </article>
          <article class="card">
            <div class="k">Dataset / Split</div>
            <div class="identity-value">RWTD test</div>
            <div class="mini-note"><code>aviadcohz/RWTD</code> · 227 evaluated crops</div>
          </article>
          <article class="card">
            <div class="k">Model / Task</div>
            <div class="identity-value">facebook/sam3</div>
            <div class="mini-note">Binary texture partition, Stage-2 scale fusion probe</div>
          </article>
        </div>
        """
    ).strip()

    config_grid = dedent(
        """\
        <div class="card-grid">
          <article class="card">
            <h3>Shared Probe</h3>
            <p class="config-copy">Projection dim <code>64</code>, embedding dim <code>32</code>, AdamW, 5 epochs, lr <code>1e-3</code>, weight decay <code>1e-4</code>.</p>
          </article>
          <article class="card">
            <h3>Clustering Contract</h3>
            <p class="config-copy">Deterministic K-means, cosine distance, <code>K=2</code>, bilinear feature alignment, 512 pair samples per image.</p>
          </article>
          <article class="card">
            <h3>Variant Space</h3>
            <p class="config-copy">3 single-scale controls, 2 all-scale fusion variants, 1 coarse+mid learned fusion, and 1 native fine-grid control.</p>
          </article>
          <article class="card">
            <h3>Missing Next Control</h3>
            <p class="config-copy"><code>fixed_uniform_gates_coarse_plus_next_finer</code> is still absent, so the strongest subset has no fixed-weight companion yet.</p>
          </article>
        </div>
        """
    ).strip()

    qc_warnings = [
        "fpn_1_only runs at its native 144x144 grid, while fpn_2_only runs at 72x72.",
        "fpn_0_only was evaluated both aligned to 72x72 and at native 288x288; native was worse on both headline metrics.",
        "The only available full all-scale learned bundle is stored under smoke_learned_global_gates_all_scales/, even though it covers the full 227-sample test set.",
        "The next useful control is still the missing fixed_uniform_gates_coarse_plus_next_finer run.",
    ]

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

    .identity-grid {{
      display: grid;
      gap: 20px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }}

    .identity-value {{
      font-size: 1.35rem;
      font-weight: 700;
      margin-top: 8px;
      color: var(--ink);
    }}

    .config-copy,
    .story-summary,
    .interpretation-copy {{
      margin: 0;
      color: var(--muted);
    }}

    .gate-readout {{
      font-size: 1.5rem !important;
      line-height: 1.2;
    }}

    .chart-grid {{
      display: grid;
      gap: 20px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-top: 26px;
    }}

    .chart-card {{
      background: var(--surface-card);
      border: 1px solid var(--glass-border);
      border-radius: 20px;
      padding: 18px;
    }}

    .chart-card img {{
      width: 100%;
      display: block;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: var(--surface-strong);
    }}

    .chart-card-wide {{
      grid-column: span 2;
    }}

    .story-block {{
      border: 1px solid var(--glass-border);
      border-radius: 20px;
      background: var(--surface-card);
      padding: 22px;
      margin-bottom: 20px;
    }}

    .story-head {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 8px;
    }}

    .story-block h3 {{
      margin: 0 0 10px;
    }}

    .story-grid {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      margin-top: 18px;
    }}

    .mini-note {{
      font-size: 0.78rem;
      color: var(--muted);
      margin-top: 4px;
    }}

    .plot-gallery {{
      display: grid;
      gap: 20px;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      margin-top: 20px;
    }}

    .artifact-grid {{
      display: grid;
      gap: 20px;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    }}

    .artifact-card {{
      border: 1px solid var(--glass-border);
      border-radius: 18px;
      background: var(--surface-card);
      padding: 20px;
    }}

    .artifact-card code {{
      font-size: 0.82rem;
    }}

    .warning-list {{
      margin: 16px 0 0;
      padding-left: 20px;
      color: var(--muted);
    }}

    .warning-list li + li {{
      margin-top: 8px;
    }}

    .delta-negative {{
      color: var(--bad) !important;
    }}

    @media (max-width: 900px) {{
      .chart-grid {{
        grid-template-columns: 1fr;
      }}

      .chart-card-wide {{
        grid-column: auto;
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
        <span class="tag">sam3</span>
        <span class="tag">stage2</span>
        <span class="tag">scale study</span>
      </div>
    </header>

    <section class="executive-summary">
      <h2>Executive Synopsis</h2>
      <p>Stage-2 asks which SAM FPN scale actually carries the binary texture partition signal once everything else stays fixed. The answer is <strong>coarse-dominant</strong>: <code>fpn_2</code> is the strongest single scale, <code>fpn_1</code> is clearly weaker alone but improves the best run when paired with <code>fpn_2</code>, and <code>fpn_0</code> is weak enough to hurt.</p>
      <p>That matters because the best result is not “coarse only.” The winning configuration is the learned coarse+next-finer probe (<code>fpn_2 + fpn_1</code>, mIoU {best_run['mean_miou']:.4f}, ARI {best_run['mean_ari']:.4f}), while learned all-scale gating only partly recovers the same insight: it does learn <code>fpn_2 &gt; fpn_1 &gt; fpn_0</code>, but still loses to both fixed-uniform all-scale fusion and the simpler coarse+mid subset.</p>
    </section>

    <section class="section">
      <h2>Run Identity / Provenance</h2>
      {provenance_grid}
    </section>

    <section class="section">
      <h2>Configuration</h2>
      {config_grid}
      <div style="margin-top: 20px; display: flex; gap: 12px; flex-wrap: wrap;">
        <a class="btn secondary" href="repro/source_runs.json">Open Run Map</a>
        <a class="btn secondary" href="assets/tables/summary_table.csv">Open Summary CSV</a>
      </div>
    </section>

    <section class="section">
      <h2>Headline Metrics</h2>
      {top_cards}
      <div class="table-wrap" style="margin-top: 26px;">
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Variant</th>
              <th>Levels Used</th>
              <th>Clustering Grid</th>
              <th>Gate Type</th>
              <th>Final Gate Weights</th>
              <th>Mean mIoU</th>
              <th>Mean ARI</th>
              <th>Takeaway</th>
            </tr>
          </thead>
          <tbody>
            {render_summary_table(summary_rows)}
          </tbody>
        </table>
      </div>
      {plot_cards}
    </section>

    <section class="section">
      <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 20px; flex-wrap: wrap;">
        <div>
          <h2>Visual Comparison Gallery</h2>
          <p class="interpretation-copy">Each published panel is the run-emitted triptych: input, GT, and prediction. The selected crops below are chosen to make the coarse-dominant story easy to see at a glance.</p>
        </div>
        <a class="btn" href="gallery.html">Open Full Lightweight Gallery →</a>
      </div>
      <div style="margin-top: 24px;">
        {render_story_blocks(story_samples)}
      </div>
    </section>

    <section class="section">
      <h2>QC / Diagnostics</h2>
      <div class="plot-gallery">
        <article class="chart-card">
          <img src="{plot_paths['delta_best_vs_fpn2.svg']}" alt="Per-sample delta between best run and fpn_2 only" />
        </article>
        <article class="chart-card">
          <img src="{plot_paths['delta_fpn0_native_vs_aligned.svg']}" alt="Per-sample delta between native and aligned fpn_0 controls" />
        </article>
      </div>
      <div class="gallery" style="margin-top: 22px;">
        {render_qc_cards(qc_assets)}
      </div>
      <ul class="warning-list">
        {"".join(f"<li>{escape(item)}</li>" for item in qc_warnings)}
      </ul>
    </section>

    <section class="section">
      <h2>Interpretation</h2>
      <p class="interpretation-copy">The coarsest SAM scale carries the main binary texture partition signal in this Stage-2 setting. The next-finer scale adds useful complementary structure, while the finest scale is not beneficial and can be actively detrimental, especially when clustered at its native resolution. The cleanest reading is therefore coarse-dominant rather than coarse-only: subset choice matters more than simply learning a softmax over all scales.</p>
      <div class="artifact-grid" style="margin-top: 20px;">
        <article class="artifact-card">
          <h3>What Improved</h3>
          <p class="config-copy">The best learned coarse+mid probe gains {signed(best_run['mean_miou'] - best_single['mean_miou'])} mIoU and {signed(best_run['mean_ari'] - best_single['mean_ari'])} ARI over <code>fpn_2</code> alone.</p>
        </article>
        <article class="artifact-card">
          <h3>What Regressed</h3>
          <p class="config-copy">The native fine-scale control drops {signed(fine_native['mean_miou'] - fine_aligned['mean_miou'])} mIoU and {signed(fine_native['mean_ari'] - fine_aligned['mean_ari'])} ARI relative to aligned <code>fpn_0_only</code>.</p>
        </article>
        <article class="artifact-card">
          <h3>What Is Suspicious</h3>
          <p class="config-copy">Even though learned all-scale gating settles on coarse > mid > fine, it still trails fixed-uniform all-scale fusion by {signed(uniform_all['mean_miou'] - all_scale_learned['mean_miou'])} mIoU and {signed(uniform_all['mean_ari'] - all_scale_learned['mean_ari'])} ARI.</p>
        </article>
      </div>
    </section>

    <section class="section">
      <h2>Artifact Links</h2>
      <div class="artifact-grid">
        <article class="artifact-card">
          <h3>Page Bundle</h3>
          <p class="config-copy"><a class="btn secondary" href="gallery.html">Gallery</a></p>
          <p class="mini-note"><code>metrics.json</code>, <code>summary.md</code>, <code>training_data.json</code>, and <code>assets/tables/summary_table.csv</code> are published beside this page.</p>
        </article>
        <article class="artifact-card">
          <h3>Repro Bundles</h3>
          <p class="config-copy">Each source run is copied into <code>repro/&lt;variant&gt;/</code> with <code>config.json</code>, <code>summary.json</code>, <code>summary.csv</code>, and the per-sample manifests.</p>
          <p style="margin-top: 16px;"><a class="btn secondary" href="repro/source_runs.json">Inspect Repro Map</a></p>
        </article>
        <article class="artifact-card">
          <h3>Source Mapping</h3>
          <p class="config-copy"><code>{escape(links_payload['source_run_map'])}</code></p>
          <p class="mini-note">Includes source experiment roots, checkpoint paths, and page-local repro folders.</p>
          <p style="margin-top: 16px;"><a class="btn secondary" href="repro/source_runs.json">Open JSON</a></p>
        </article>
      </div>
      <div style="margin-top: 28px; text-align: center;">
        <a class="btn secondary" href="../../index.html">← Back to Dashboard</a>
      </div>
    </section>

    <footer>
      <p>Report generated on {PAGE_DATE} | Powered by <strong>SAM 3 Intelligence Hub</strong></p>
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


def render_gallery_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Coarse-vs-Fine SAM Scales — Stage 2 Gallery</title>
  <script>
    (() => {
      const storageKey = 'research-site-theme';
      let theme = null;
      try {
        const stored = window.localStorage.getItem(storageKey);
        if (stored === 'dark' || stored === 'light') theme = stored;
      } catch (_) {}
      if (!theme && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        theme = 'dark';
      }
      document.documentElement.dataset.theme = theme || 'light';
      document.documentElement.style.colorScheme = theme || 'light';
    })();
  </script>
  <link rel="stylesheet" href="../../assets/site.css" />
  <style>
    .controls {
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      margin-bottom: 20px;
    }

    .control {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .control label {
      font-size: 0.8rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-weight: 700;
      color: var(--muted);
    }

    .control select,
    .control input {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 14px;
      background: var(--surface-strong);
      color: var(--ink);
      font: inherit;
    }

    .gallery-card {
      border-radius: 18px;
      overflow: hidden;
      background: var(--surface-strong);
      border: 1px solid var(--line);
      display: flex;
      flex-direction: column;
    }

    .gallery-card img {
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: cover;
      display: block;
      cursor: zoom-in;
    }

    .gallery-card .body {
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .gallery-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }

    .status-row {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
      margin-bottom: 18px;
      color: var(--muted);
    }

    .load-more-wrap {
      margin-top: 24px;
      text-align: center;
    }
  </style>
</head>

<body>
  <div class="container animate-fade-in-down">
    <div class="page-tools">
      <button class="theme-toggle" type="button" data-theme-toggle aria-label="Toggle color theme" aria-pressed="false">
        <span data-theme-toggle-label>Dark mode</span>
      </button>
    </div>

    <header style="text-align: center; margin-bottom: 42px;">
      <div class="subtitle">Full Gallery</div>
      <h1 class="title-gradient">Coarse-vs-Fine SAM Scales — Lightweight Gallery</h1>
      <p style="color: var(--muted); margin-top: 16px; font-weight: 300; max-width: 860px; margin-inline: auto;">
        Published web previews for every Stage-2 crop across the available variants. Metrics come from the full run bundles; the published images are downscaled previews to keep the Pages payload lightweight.
      </p>
      <div style="margin-top: 18px;">
        <a class="btn secondary" href="index.html">← Back to Report</a>
      </div>
    </header>

    <section class="section">
      <div class="controls">
        <div class="control">
          <label for="variantFilter">Variant</label>
          <select id="variantFilter"></select>
        </div>
        <div class="control">
          <label for="familyFilter">Family</label>
          <select id="familyFilter">
            <option value="all">All families</option>
            <option value="single_scale">Single scale</option>
            <option value="all_scale_fusion">All-scale fusion</option>
            <option value="best_subset">Best subset</option>
            <option value="native_control">Native fine control</option>
          </select>
        </div>
        <div class="control">
          <label for="sortBy">Sort</label>
          <select id="sortBy">
            <option value="sample">Sample order</option>
            <option value="miou_desc">mIoU ↓</option>
            <option value="miou_asc">mIoU ↑</option>
            <option value="ari_desc">ARI ↓</option>
            <option value="delta_desc">Δ vs fpn_2 ↓</option>
            <option value="delta_asc">Δ vs fpn_2 ↑</option>
          </select>
        </div>
        <div class="control">
          <label for="pageSize">Per page</label>
          <select id="pageSize">
            <option value="24">24</option>
            <option value="48">48</option>
            <option value="96">96</option>
          </select>
        </div>
      </div>

      <div class="status-row">
        <div id="gallerySummary">Loading gallery…</div>
        <div id="galleryHint">Click any preview to expand.</div>
      </div>

      <div class="gallery" id="fullGallery"></div>

      <div class="load-more-wrap">
        <button id="loadMoreButton" type="button" class="btn">Load More</button>
      </div>
    </section>

    <footer>
      <p>Stage-2 gallery bundle | Lightweight previews only</p>
    </footer>
  </div>

  <dialog class="image-dialog" id="imageDialog">
    <img id="imageDialogImg" alt="Expanded preview" />
  </dialog>

  <script src="../../assets/site.js"></script>
  <script>
    (async () => {
      const response = await fetch('training_data.json');
      if (!response.ok) {
        throw new Error(`Failed to load training_data.json: ${response.status}`);
      }
      const data = await response.json();
      const records = data.gallery.records.slice();
      const variantFilter = document.getElementById('variantFilter');
      const familyFilter = document.getElementById('familyFilter');
      const sortBy = document.getElementById('sortBy');
      const pageSize = document.getElementById('pageSize');
      const gallery = document.getElementById('fullGallery');
      const summary = document.getElementById('gallerySummary');
      const loadMoreButton = document.getElementById('loadMoreButton');
      const params = new URLSearchParams(window.location.search);

      const variants = data.gallery.variants;
      variantFilter.innerHTML = [
        '<option value="all">All variants</option>',
        ...variants.map((variant) => `<option value="${variant.id}">${variant.label}</option>`)
      ].join('');

      variantFilter.value = params.get('variant') || 'all';
      familyFilter.value = params.get('family') || 'all';
      sortBy.value = params.get('sort') || 'sample';
      pageSize.value = params.get('page_size') || '24';
      let shown = Number(params.get('shown')) || Number(pageSize.value);

      function syncUrl() {
        const next = new URLSearchParams();
        if (variantFilter.value !== 'all') next.set('variant', variantFilter.value);
        if (familyFilter.value !== 'all') next.set('family', familyFilter.value);
        if (sortBy.value !== 'sample') next.set('sort', sortBy.value);
        if (pageSize.value !== '24') next.set('page_size', pageSize.value);
        if (shown > Number(pageSize.value)) next.set('shown', String(shown));
        const query = next.toString();
        history.replaceState(null, '', query ? `?${query}` : 'gallery.html');
      }

      function filteredRecords() {
        let items = records.slice();
        if (variantFilter.value !== 'all') {
          items = items.filter((item) => item.variant_id === variantFilter.value);
        }
        if (familyFilter.value !== 'all') {
          items = items.filter((item) => item.family === familyFilter.value);
        }
        if (sortBy.value === 'miou_desc') {
          items.sort((a, b) => b.miou - a.miou || a.sample_index - b.sample_index);
        } else if (sortBy.value === 'miou_asc') {
          items.sort((a, b) => a.miou - b.miou || a.sample_index - b.sample_index);
        } else if (sortBy.value === 'ari_desc') {
          items.sort((a, b) => b.ari - a.ari || a.sample_index - b.sample_index);
        } else if (sortBy.value === 'delta_desc') {
          items.sort((a, b) => b.delta_vs_fpn2 - a.delta_vs_fpn2 || a.sample_index - b.sample_index);
        } else if (sortBy.value === 'delta_asc') {
          items.sort((a, b) => a.delta_vs_fpn2 - b.delta_vs_fpn2 || a.sample_index - b.sample_index);
        } else {
          items.sort((a, b) => a.sample_index - b.sample_index || a.variant_id.localeCompare(b.variant_id));
        }
        return items;
      }

      function card(item) {
        const deltaTag = item.delta_vs_fpn2 > 0.0005
          ? `<span class="tag good">Δ {item.delta_vs_fpn2.toFixed(4)}</span>`
          : item.delta_vs_fpn2 < -0.0005
            ? `<span class="tag bad">Δ ${item.delta_vs_fpn2.toFixed(4)}</span>`
            : '<span class="tag">Δ 0.0000</span>';
        return `
          <article class="gallery-card" data-score="${item.miou}">
            <img src="${item.thumb_path}" alt="crop ${item.sample_id} — ${item.variant_label}" data-zoom-src="${item.thumb_path}" />
            <div class="body">
              <div class="gallery-meta">
                <span class="pill">crop ${item.sample_id}</span>
                <span class="tag">${item.variant_label}</span>
                ${deltaTag}
              </div>
              <div><strong>mIoU ${item.miou.toFixed(4)}</strong> · ARI ${item.ari.toFixed(4)}</div>
              <div class="mini-note">${item.family.replaceAll('_', ' ')}</div>
            </div>
          </article>
        `;
      }

      function render() {
        const items = filteredRecords();
        const limit = Math.min(shown, items.length);
        gallery.innerHTML = items.slice(0, limit).map(card).join('');
        summary.textContent = `Showing ${limit} of ${items.length} published previews across ${data.gallery.sample_count} test crops.`;
        loadMoreButton.hidden = limit >= items.length;
        loadMoreButton.disabled = limit >= items.length;
        syncUrl();
      }

      variantFilter.addEventListener('change', () => {
        shown = Number(pageSize.value);
        render();
      });
      familyFilter.addEventListener('change', () => {
        shown = Number(pageSize.value);
        render();
      });
      sortBy.addEventListener('change', () => {
        shown = Number(pageSize.value);
        render();
      });
      pageSize.addEventListener('change', () => {
        shown = Number(pageSize.value);
        render();
      });
      loadMoreButton.addEventListener('click', () => {
        shown += Number(pageSize.value);
        render();
      });

      render();
    })().catch((error) => {
      document.getElementById('gallerySummary').textContent = error.message;
      document.getElementById('loadMoreButton').hidden = true;
    });
  </script>
</body>

</html>
"""


def render_summary_md(runs: dict[str, dict]) -> str:
    best = runs["learned_global_gates_coarse_plus_next_finer"]
    fpn2 = runs["fpn_2_only"]
    fine_native = runs["fpn_0_only_native"]
    fine_aligned = runs["fpn_0_only_aligned"]
    return dedent(
        f"""\
        # {PAGE_TITLE}

        - Dataset: `aviadcohz/RWTD` test split
        - Best run: `learned_global_gates_coarse_plus_next_finer` with mean `eval_miou={best['mean_miou']:.6f}` and `eval_ari={best['mean_ari']:.6f}`
        - Best single scale: `fpn_2_only` with mean `eval_miou={fpn2['mean_miou']:.6f}` and `eval_ari={fpn2['mean_ari']:.6f}`
        - Main conclusion: coarse-dominant, not coarse-only. `fpn_1` is weaker alone but improves the best run when added to `fpn_2`.
        - Fine control: `fpn_0_only_native` drops `{fine_native['mean_miou'] - fine_aligned['mean_miou']:+.6f}` mIoU and `{fine_native['mean_ari'] - fine_aligned['mean_ari']:+.6f}` ARI versus aligned `fpn_0_only`.
        - Missing next control: `fixed_uniform_gates_coarse_plus_next_finer`
        """
    )


def build_links_payload(repro_mapping: list[dict], summary_csv_rel: str) -> dict:
    return {
        "dashboard": "../../index.html",
        "report": "index.html",
        "gallery": "gallery.html",
        "metrics": "metrics.json",
        "training_data": "training_data.json",
        "summary": "summary.md",
        "summary_csv": summary_csv_rel,
        "source_run_map": "repro/source_runs.json",
        "repro_runs": repro_mapping,
    }


def build_all() -> None:
    runs = {spec.run_id: load_run(spec) for spec in RUN_SPECS}
    sample_ids = enforce_sample_alignment(runs)
    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = build_summary_rows(runs)
    story_samples = choose_story_samples(runs, sample_ids)
    create_story_assets(runs, story_samples)
    qc_assets = create_qc_assets(runs, story_samples)
    gallery_records = create_gallery_previews(runs, sample_ids)
    plot_paths = write_plot_assets(runs)
    summary_csv_rel = write_summary_table_csv(summary_rows)
    repro_mapping = [copy_repro_bundle(spec) for spec in RUN_SPECS]

    metrics_payload = build_metrics_payload(runs, summary_rows)
    training_data = build_training_data(runs, summary_rows, story_samples, gallery_records, qc_assets, repro_mapping)
    links_payload = build_links_payload(repro_mapping, summary_csv_rel)

    write_text(DEST_DIR / "metrics.json", json.dumps(metrics_payload, indent=2))
    write_text(DEST_DIR / "training_data.json", json.dumps(training_data, indent=2))
    write_text(DEST_DIR / "links.json", json.dumps(links_payload, indent=2))
    write_text(DEST_DIR / "summary.md", render_summary_md(runs))
    write_text(
        DEST_DIR / "manifest.yaml",
        build_manifest(
            runs=runs,
            gallery_count=sum(len(story["images"]) for story in story_samples),
            qc_count=len(qc_assets),
            all_previews_count=len(gallery_records),
        ),
    )

    source_run_map = []
    for spec in RUN_SPECS:
        run = runs[spec.run_id]
        source_run_map.append(
            {
                "run_id": spec.run_id,
                "label": spec.label,
                "source_dir": run["source_dir"].relative_to(ROOT).as_posix(),
                "checkpoint_path": run["checkpoint_path"],
                "naming_note": spec.naming_note,
            }
        )
    write_text(DEST_DIR / "repro" / "source_runs.json", json.dumps(source_run_map, indent=2))

    git_commit = git_output("rev-parse", "--short", "HEAD")
    git_author = git_output("config", "user.name") or "unknown"
    write_text(
        DEST_DIR / "index.html",
        render_index_html(
            runs=runs,
            summary_rows=summary_rows,
            story_samples=story_samples,
            qc_assets=qc_assets,
            plot_paths=plot_paths,
            links_payload=links_payload,
            git_commit=git_commit,
            git_author=git_author,
        ),
    )
    write_text(DEST_DIR / "gallery.html", render_gallery_html())


if __name__ == "__main__":
    build_all()
