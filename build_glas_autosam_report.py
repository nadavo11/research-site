#!/usr/bin/env python3
"""Build the GlaS + MoNuSeg vs AutoSAM frozen-SAM3 readout report bundle."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import shutil
from dataclasses import dataclass
from datetime import date
from html import escape
from pathlib import Path
from textwrap import dedent

from PIL import Image


ROOT = Path(__file__).resolve().parent
TEXTURE_REPO_ROOT = ROOT.parent / "texture representations"
GLAS_OUTPUT_ROOT = TEXTURE_REPO_ROOT / "outputs" / "glas_binary"
MONUSEG_OUTPUT_ROOT = TEXTURE_REPO_ROOT / "outputs" / "monuseg_binary" / "frozen_sam_mask_head"
DEST_DIR = ROOT / "site" / "experiments" / "glas-vs-autosam-frozen-sam3-readout-study"
METHOD_README = ROOT / "experiments" / "autosam head to head" / "glas_supervised_frozen_sam_phase2" / "method" / "README.md"
GRAPHVIZ_DOT_CANDIDATES = [
    Path.home() / ".local" / "graphviz-env" / "bin" / "dot",
]

PAGE_TITLE = "GlaS + MoNuSeg vs AutoSAM: Frozen SAM3 Readout Study"
PAGE_SUBTITLE = "Experiment Report"
PAGE_DATE = date.today().isoformat()
PAGE_DESCRIPTION = (
    "Across GlaS and MoNuSeg, the current evidence points much more strongly to the readout bottleneck than to missing "
    "frozen-feature signal. GlaS still shows that a tiny supervised head on frozen SAM3 features is already strong, while "
    "the new MoNuSeg runs show that a slightly richer coarse-plus-mid residual readout can get very close to the cited AutoSAM "
    "foreground numbers. AutoSAM values on this page remain reported paper references, not reproduced runs."
)

GLAS_AUTOSAM_REPORTED_FG_IOU = 0.8708
GLAS_AUTOSAM_REPORTED_DICE = 0.9282
MONUSEG_AUTOSAM_REPORTED_FG_IOU = 0.7017
MONUSEG_AUTOSAM_REPORTED_DICE = 0.8243
GALLERY_PREVIEW_WIDTH = 1680
GALLERY_PREVIEW_QUALITY = 80

AUTOSAM_COMPARISON_README = TEXTURE_REPO_ROOT / "outputs" / "glas_autosam_comparison_phase1" / "README.md"
PHASE1_README = TEXTURE_REPO_ROOT / "outputs" / "glas_supervised_frozen_sam_phase1" / "README.md"
PHASE2_README = TEXTURE_REPO_ROOT / "outputs" / "glas_supervised_frozen_sam_phase2" / "README.md"

REQUIRED_RUN_FILES = [
    "config.json",
    "experiment_terms.md",
    "summary.json",
    "summary.md",
    "per_sample_metrics.csv",
    "visuals_manifest.jsonl",
]

METRIC_COLORS = {
    "fg_iou": "#005f73",
    "dice": "#ca6702",
}


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    label: str
    short_label: str
    source_dir: str
    supervision: str
    frozen_backbone: str
    learned_prompt_generator: str
    notes: str


GLAS_RUN_SPECS = [
    RunSpec(
        run_id="cfc_baseline",
        label="CFC training-free baseline",
        short_label="CFC baseline",
        source_dir="glas_autosam_phase1_flipavg_baseline",
        supervision="none",
        frozen_backbone="yes",
        learned_prompt_generator="no",
        notes="Exact GlaS flip-avg coarse-only baseline from the phase-1 AutoSAM comparison note.",
    ),
    RunSpec(
        run_id="all_scales_d64",
        label="Frozen mask head all_scales d64",
        short_label="all_scales d64",
        source_dir="frozen_sam_mask_head/train_all_scales_test_e20_visuals",
        supervision="dense binary gland masks",
        frozen_backbone="yes",
        learned_prompt_generator="no",
        notes="Legacy tiny multiscale decoder; summary bundle records 40 epochs despite the older directory name.",
    ),
    RunSpec(
        run_id="fpn_0_only_d64",
        label="Frozen mask head fpn_0_only d64",
        short_label="fpn_0_only d64",
        source_dir="frozen_sam_mask_head/train_fpn_0_only_test_e40",
        supervision="dense binary gland masks",
        frozen_backbone="yes",
        learned_prompt_generator="no",
        notes="Finest single-scale control.",
    ),
    RunSpec(
        run_id="mid_plus_fine_d64",
        label="Frozen mask head mid_plus_fine d64",
        short_label="mid_plus_fine d64",
        source_dir="frozen_sam_mask_head/train_mid_plus_fine_test_e40",
        supervision="dense binary gland masks",
        frozen_backbone="yes",
        learned_prompt_generator="no",
        notes="Drops the coarsest scale and fuses only the two finer levels.",
    ),
    RunSpec(
        run_id="fpn_2_only_d64",
        label="Frozen mask head fpn_2_only d64",
        short_label="fpn_2_only d64",
        source_dir="frozen_sam_mask_head/train_fpn_2_only_test_e40",
        supervision="dense binary gland masks",
        frozen_backbone="yes",
        learned_prompt_generator="no",
        notes="Default-width coarse-only run; this is the ambiguous `fpn_2_only` directory that resolves to d64.",
    ),
    RunSpec(
        run_id="fpn_2_only_d32",
        label="Frozen mask head fpn_2_only d32",
        short_label="fpn_2_only d32",
        source_dir="frozen_sam_mask_head/train_fpn_2_only_d32_test_e40",
        supervision="dense binary gland masks",
        frozen_backbone="yes",
        learned_prompt_generator="no",
        notes="Lower-capacity coarse-only control.",
    ),
    RunSpec(
        run_id="fpn_2_only_d128",
        label="Frozen mask head fpn_2_only d128",
        short_label="fpn_2_only d128",
        source_dir="frozen_sam_mask_head/train_fpn_2_only_d128_test_e40",
        supervision="dense binary gland masks",
        frozen_backbone="yes",
        learned_prompt_generator="no",
        notes="Best current frozen-feature run and the current internal headline comparison point.",
    ),
    RunSpec(
        run_id="fpn_2_only_d128_224_autosamaug",
        label="Frozen mask head fpn_2_only d128 @ 224 + autosam aug",
        short_label="fpn_2_only d128 @224",
        source_dir="frozen_sam_mask_head/train_fpn_2_only_d128_224_autosamaug_e200",
        supervision="dense binary gland masks",
        frozen_backbone="yes",
        learned_prompt_generator="no",
        notes="Closest current GlaS AutoSAM-style endpoint: explicit 224x224 resize, AutoSAM-style dense augmentation, final epoch checkpoint.",
    ),
]

GLAS_RESULTS_ORDER = [
    "cfc_baseline",
    "all_scales_d64",
    "fpn_2_only_d64",
    "fpn_2_only_d32",
    "fpn_2_only_d128",
    "fpn_2_only_d128_224_autosamaug",
]

GLAS_SCALE_ABLATION_IDS = ["fpn_0_only_d64", "mid_plus_fine_d64", "all_scales_d64", "fpn_2_only_d64"]
GLAS_CAPACITY_SWEEP_IDS = ["fpn_2_only_d32", "fpn_2_only_d64", "fpn_2_only_d128"]

MONUSEG_RUN_SPECS = [
    RunSpec(
        run_id="strict512_coarse",
        label="Frozen mask head fpn_2_only d128 @ 512x512",
        short_label="strict512 coarse",
        source_dir="train_fpn_2_only_d128_512_test_e40",
        supervision="dense binary nucleus masks",
        frozen_backbone="yes",
        learned_prompt_generator="no",
        notes="Strict MoNuSeg anchor: official 30 -> 14 split semantics, explicit 512x512 resize, direct nucleus foreground metrics.",
    ),
    RunSpec(
        run_id="native_aug_coarse",
        label="Frozen mask head fpn_2_only d128 @ native + autosam aug",
        short_label="native coarse",
        source_dir="train_fpn_2_only_d128_native_autosamaug_e20",
        supervision="dense binary nucleus masks",
        frozen_backbone="yes",
        learned_prompt_generator="no",
        notes="Native-resolution coarse-only run with AutoSAM-style augmentation; roughly flat versus the strict 512x512 anchor.",
    ),
    RunSpec(
        run_id="native_aug_refine",
        label="Frozen mask head fpn_2+fpn_1 refine d128 @ native + autosam aug",
        short_label="native refine",
        source_dir="train_fpn_2_plus_fpn_1_refine_d128_native_autosamaug_e20",
        supervision="dense binary nucleus masks",
        frozen_backbone="yes",
        learned_prompt_generator="no",
        notes="Strongest current MoNuSeg run: coarse head plus tiny mid-scale residual refinement with only a small parameter increase.",
    ),
]

QUAL_STORIES = [
    {
        "story_id": "testa38_supervision_rescue",
        "crop_name": "testA_38",
        "tag": "supervision rescue",
        "title": "Supervision Repairs a Near-Collapse Case",
        "summary": (
            "The training-free CFC branch nearly misses the gland entirely here. Both supervised heads recover the structure, "
            "and the best coarse-only readout edges the all-scales head again."
        ),
    },
    {
        "story_id": "testa39_coarse_only_edge",
        "crop_name": "testA_39",
        "tag": "coarse-only edge",
        "title": "Coarse-Only Can Beat the Multiscale Head Cleanly",
        "summary": (
            "This crop is a useful reminder that the best current result is not just a width effect. "
            "The coarse-only d128 head is visibly cleaner than the all-scales d64 baseline."
        ),
    },
    {
        "story_id": "testa53_all_scales_holdout",
        "crop_name": "testA_53",
        "tag": "all-scales holdout",
        "title": "Coarse-Only Is Strongest Overall, But Not Universal",
        "summary": (
            "The strongest aggregate result is coarse-only, but this crop is a good counterexample. "
            "The all-scales d64 head is actually better here than the best coarse-only run."
        ),
    },
    {
        "story_id": "testa17_capacity_and_readout",
        "crop_name": "testA_17",
        "tag": "capacity helps",
        "title": "Better Readout First, Then Extra Width",
        "summary": (
            "CFC is weak, the all-scales head improves it, and the best coarse-only head improves further. "
            "This is representative of the current coarse-dominant readout story."
        ),
    },
    {
        "story_id": "testb14_limitation",
        "crop_name": "testB_14",
        "tag": "limitation",
        "title": "A Real Failure Case Still Exists",
        "summary": (
            "This crop stays difficult for the supervised frozen-feature heads and is worth keeping visible. "
            "The current page is not claiming that the readout fully dominates every CFC sample."
        ),
    },
]

MONUSEG_QUAL_STORIES = [
    {
        "story_id": "monuseg_44_2665_residual_jump",
        "crop_name": "TCGA-44-2665-01B-06-BS6",
        "tag": "residual jump",
        "title": "Residual Refinement Produces a Clear MoNuSeg Leap",
        "summary": (
            "This is the strongest single-sample gain in the current MoNuSeg set. The native coarse-only run barely changes "
            "the strict 512x512 baseline, while the coarse+mid residual branch sharpens the nucleus map immediately."
        ),
    },
    {
        "story_id": "monuseg_ac_a2fo_rescue",
        "crop_name": "TCGA-AC-A2FO-01A-01-TS1",
        "tag": "rescue case",
        "title": "When Coarse-Only Stalls, the Mid-Scale Branch Can Rescue It",
        "summary": (
            "This crop matters because it shows the structural story more clearly than the averages. The small residual "
            "branch adds a real correction, not just a cosmetic smoothing effect."
        ),
    },
    {
        "story_id": "monuseg_fg_a4mu_strong_anchor",
        "crop_name": "TCGA-FG-A4MU-01B-01-TS1",
        "tag": "already strong",
        "title": "The Best MoNuSeg Head Still Helps Even When the Baseline Is Decent",
        "summary": (
            "The strict 512x512 baseline is already respectable on this crop, but the residual branch still improves it. "
            "That is why the page treats the MoNuSeg gain as a readout effect, not just a rescue-on-failures effect."
        ),
    },
    {
        "story_id": "monuseg_ao_a0j2_limited_gain",
        "crop_name": "TCGA-AO-A0J2-01A-01-BSA",
        "tag": "limited gain",
        "title": "The MoNuSeg Refinement Branch Is Strong, Not Magic",
        "summary": (
            "This crop is one of the smallest positive deltas. Keeping it visible prevents the section from overstating the "
            "improvement and makes the average gain easier to trust."
        ),
    },
]


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return path


def load_json(path: Path) -> dict:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def fmt(value: float | None, digits: int = 6) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_params(value: int | str | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    return f"{value:,}"


def write_webp(image: Image.Image, dest_path: Path, *, width: int | None = None, quality: int = 82) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    image = image.convert("RGB")
    if width is not None and image.width > width:
        scale = width / image.width
        image = image.resize((width, max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    image.save(dest_path, format="WEBP", quality=quality, method=6)


def render_bar_chart(
    *,
    title: str,
    subtitle: str,
    groups: list[dict],
    highlight_reported: bool = False,
) -> str:
    width = 1160
    height = 520
    top = 100
    left = 72
    right = 28
    bottom = 130
    chart_height = height - top - bottom
    chart_width = width - left - right
    bar_width = 54
    intra_gap = 14
    group_gap = 42
    group_width = bar_width * 2 + intra_gap
    total_width = len(groups) * group_width + (len(groups) - 1) * group_gap
    start_x = left + max(0, (chart_width - total_width) / 2)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        '<rect width="100%" height="100%" fill="#ffffff" rx="24" />',
        f'<text x="28" y="40" font-family="Space Grotesk, sans-serif" font-size="24" font-weight="700" fill="#122033">{escape(title)}</text>',
        f'<text x="28" y="64" font-family="Space Grotesk, sans-serif" font-size="13" fill="#4c5d73">{escape(subtitle)}</text>',
        f'<rect x="{width - 230}" y="24" width="14" height="14" rx="4" fill="{METRIC_COLORS["fg_iou"]}" />',
        f'<text x="{width - 208}" y="36" font-family="Space Grotesk, sans-serif" font-size="12" fill="#122033">Foreground IoU</text>',
        f'<rect x="{width - 110}" y="24" width="14" height="14" rx="4" fill="{METRIC_COLORS["dice"]}" />',
        f'<text x="{width - 88}" y="36" font-family="Space Grotesk, sans-serif" font-size="12" fill="#122033">Dice</text>',
    ]

    for tick in [0.0, 0.25, 0.50, 0.75, 1.0]:
        y = top + chart_height * (1.0 - tick)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e3ebf2" stroke-width="1" />')
        lines.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Space Grotesk, sans-serif" font-size="11" fill="#4c5d73">{tick:.2f}</text>'
        )

    cursor_x = start_x
    for group in groups:
        group_center = cursor_x + group_width / 2
        if highlight_reported and group.get("reported_reference"):
            box_x = cursor_x - 16
            box_y = top - 18
            box_w = group_width + 32
            box_h = chart_height + 46
            lines.append(
                f'<rect x="{box_x:.1f}" y="{box_y:.1f}" width="{box_w:.1f}" height="{box_h:.1f}" rx="20" fill="#fff7ed" stroke="#ca6702" stroke-width="2" stroke-dasharray="8 8" />'
            )
            lines.append(
                f'<text x="{group_center:.1f}" y="{box_y + 16:.1f}" text-anchor="middle" font-family="Space Grotesk, sans-serif" font-size="11" font-weight="700" fill="#ca6702">reported reference</text>'
            )

        for metric_index, metric_key in enumerate(["fg_iou", "dice"]):
            value = float(group[metric_key])
            bar_height = chart_height * max(0.0, min(1.0, value))
            x = cursor_x + metric_index * (bar_width + intra_gap)
            y = top + chart_height - bar_height
            lines.extend(
                [
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" rx="10" fill="{METRIC_COLORS[metric_key]}" />',
                    f'<text x="{x + bar_width / 2:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-family="Space Grotesk, sans-serif" font-size="11" font-weight="700" fill="#122033">{value:.3f}</text>',
                ]
            )

        lines.append(
            f'<text x="{group_center:.1f}" y="{height - 54}" text-anchor="middle" font-family="Outfit, sans-serif" font-size="14" font-weight="700" fill="#005f73">{escape(group["label"])}</text>'
        )
        if group.get("sub_label"):
            lines.append(
                f'<text x="{group_center:.1f}" y="{height - 36}" text-anchor="middle" font-family="Space Grotesk, sans-serif" font-size="11" fill="#4c5d73">{escape(group["sub_label"])}</text>'
            )
        cursor_x += group_width + group_gap

    lines.append("</svg>")
    return "\n".join(lines)


def render_capacity_chart(*, title: str, subtitle: str, points: list[dict], all_scales_reference: dict) -> str:
    width = 1040
    height = 480
    top = 88
    left = 72
    right = 30
    bottom = 110
    chart_width = width - left - right
    chart_height = height - top - bottom
    x_positions = [left + chart_width * i / (len(points) - 1) for i in range(len(points))]

    def y_for(value: float) -> float:
        return top + chart_height * (1.0 - value)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        '<rect width="100%" height="100%" fill="#ffffff" rx="24" />',
        f'<text x="28" y="38" font-family="Space Grotesk, sans-serif" font-size="24" font-weight="700" fill="#122033">{escape(title)}</text>',
        f'<text x="28" y="62" font-family="Space Grotesk, sans-serif" font-size="13" fill="#4c5d73">{escape(subtitle)}</text>',
        f'<rect x="{width - 230}" y="24" width="14" height="14" rx="4" fill="{METRIC_COLORS["fg_iou"]}" />',
        f'<text x="{width - 208}" y="36" font-family="Space Grotesk, sans-serif" font-size="12" fill="#122033">Foreground IoU</text>',
        f'<rect x="{width - 118}" y="24" width="14" height="14" rx="4" fill="{METRIC_COLORS["dice"]}" />',
        f'<text x="{width - 96}" y="36" font-family="Space Grotesk, sans-serif" font-size="12" fill="#122033">Dice</text>',
    ]

    for tick in [0.6, 0.7, 0.8, 0.9, 1.0]:
        y = y_for(tick)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e3ebf2" stroke-width="1" />')
        lines.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Space Grotesk, sans-serif" font-size="11" fill="#4c5d73">{tick:.2f}</text>'
        )

    for metric_key, dash, label_y in [("fg_iou", "8 6", top + 18), ("dice", "5 5", top + 36)]:
        ref_y = y_for(float(all_scales_reference[metric_key]))
        lines.append(
            f'<line x1="{left}" y1="{ref_y:.1f}" x2="{width - right}" y2="{ref_y:.1f}" stroke="{METRIC_COLORS[metric_key]}" stroke-width="2" stroke-dasharray="{dash}" opacity="0.55" />'
        )
        lines.append(
            f'<text x="{width - right}" y="{label_y}" text-anchor="end" font-family="Space Grotesk, sans-serif" font-size="11" fill="{METRIC_COLORS[metric_key]}">all_scales d64 {metric_key} = {all_scales_reference[metric_key]:.3f}</text>'
        )

    for metric_key in ["fg_iou", "dice"]:
        color = METRIC_COLORS[metric_key]
        coords = " ".join(f"{x:.1f},{y_for(float(point[metric_key])):.1f}" for x, point in zip(x_positions, points))
        lines.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />')
        for x, point in zip(x_positions, points):
            y = y_for(float(point[metric_key]))
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" />')
            lines.append(
                f'<text x="{x:.1f}" y="{y - 12:.1f}" text-anchor="middle" font-family="Space Grotesk, sans-serif" font-size="11" font-weight="700" fill="#122033">{point[metric_key]:.3f}</text>'
            )

    for x, point in zip(x_positions, points):
        lines.append(
            f'<text x="{x:.1f}" y="{height - 52}" text-anchor="middle" font-family="Outfit, sans-serif" font-size="14" font-weight="700" fill="#005f73">d{point["decoder_dim"]}</text>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{height - 34}" text-anchor="middle" font-family="Space Grotesk, sans-serif" font-size="11" fill="#4c5d73">{format_params(point["trainable_parameter_count"])} params</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


def find_graphviz_dot() -> str:
    configured = os.environ.get("GRAPHVIZ_DOT")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.exists():
            return str(candidate)
        raise RuntimeError(f"GRAPHVIZ_DOT points to a missing file: {candidate}")
    for candidate in GRAPHVIZ_DOT_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    discovered = shutil.which("dot")
    if discovered:
        return discovered
    raise RuntimeError(
        "Graphviz 'dot' binary not found. Install Graphviz or place dot at "
        f"{GRAPHVIZ_DOT_CANDIDATES[0]} before rebuilding this report."
    )


def render_architecture_dot() -> str:
    return dedent(
        r'''
        digraph GlasFrozenSam3DenseHead {
          graph [
            rankdir=TB,
            splines=polyline,
            bgcolor="white",
            pad=0.25,
            nodesep=0.42,
            ranksep=0.62,
            fontname="DejaVu Sans",
            labelloc=t,
            labeljust=l,
            label="Frozen SAM3 Dense Mask Head on GlaS\nShared readout spine with explicit training and evaluation branches",
            fontsize=24
          ];

          node [
            fontname="DejaVu Sans",
            fontsize=12,
            color="#8a98ab",
            penwidth=1.4,
            margin="0.16,0.10",
            style="rounded,filled"
          ];

          edge [
            fontname="DejaVu Sans",
            fontsize=10,
            color="#4c5d73",
            penwidth=1.4,
            arrowsize=0.75
          ];

          subgraph cluster_data {
            label="Data and supervision contract";
            color="#88c0d0";
            fillcolor="#f4fbfd";
            style="rounded,filled";
            margin=20;

            input_image [shape=folder, fillcolor="#ffffff", label="RGB image\nGlaS crop"];
            gt_mask [shape=folder, fillcolor="#ffffff", label="Dense target mask\ngland foreground"];
            split_note [shape=note, fillcolor="#ffffff", label="train -> optimize head\ntest -> report metrics"];
            { rank=same; input_image; gt_mask; split_note; }
          }

          subgraph cluster_backbone {
            label="Frozen SAM3 backbone";
            color="#9db4ff";
            fillcolor="#f6f6ff";
            style="rounded,filled";
            margin=20;

            sam3 [shape=component, fillcolor="#ffffff", label="SAM3 feature extractor\nbackbone_fpn only"];
            freeze_tag [shape=note, fillcolor="#ffffff", label="shared at train + eval\nno prompt learning\nno backbone updates"];
          }

          subgraph cluster_pyramid {
            label="Observed multiscale feature pyramid";
            color="#f4a261";
            fillcolor="#fff8ef";
            style="rounded,filled";
            margin=20;

            p2 [shape=box, fillcolor="#ffffff", label="fpn_2\n256 x 72 x 72\ncoarsest, strongest"];
            p1 [shape=box, fillcolor="#ffffff", label="fpn_1\n256 x 144 x 144"];
            p0 [shape=box, fillcolor="#ffffff", label="fpn_0\n256 x 288 x 288\nfinest"];
            select_levels [shape=box, fillcolor="#fffdf8", label="Select subset\nall / coarse+mid /\ncoarse / fine"];
            { rank=same; p2; p1; p0; }
          }

          subgraph cluster_head {
            label="Shared dense readout";
            color="#58a37c";
            fillcolor="#eefaf4";
            style="rounded,filled";
            margin=20;

            project [shape=box3d, fillcolor="#ffffff", label="Per-level 1x1 projections\nto width d"];
            resize [shape=box, fillcolor="#ffffff", label="Resize to finest selected grid\nbilinear, align_corners=false"];
            fuse_decode [shape=box3d, fillcolor="#ffffff", label="Concat -> 2 x Conv3x3\nGroupNorm(8) + GELU"];
            logits [shape=tab, fillcolor="#ffffff", label="Upsample to image size\n1x1 classifier -> foreground logits"];
            head_params [shape=note, fillcolor="#ffffff", label="trainable only\nprojection + decoder + classifier"];
          }

          subgraph cluster_train {
            label="Training path";
            color="#d17aa8";
            fillcolor="#fef7fb";
            style="rounded,filled";
            margin=20;

            loss [shape=hexagon, fillcolor="#ffffff", label="BCEWithLogits + Dice"];
            optimizer [shape=box, fillcolor="#ffffff", label="AdamW\nlr 1e-3, wd 1e-4"];
          }

          subgraph cluster_eval {
            label="Inference / evaluation path";
            color="#a3b26d";
            fillcolor="#f8f9ee";
            style="rounded,filled";
            margin=20;

            sigmoid [shape=box, fillcolor="#ffffff", label="Sigmoid"];
            threshold [shape=diamond, fillcolor="#ffffff", label="threshold 0.5"];
            pred_fg [shape=tab, fillcolor="#ffffff", label="Predicted gland foreground"];
            pred_bg [shape=note, fillcolor="#ffffff", label="Background\ncomplement"];
            metrics [shape=note, fillcolor="#ffffff", label="Metrics vs GT reference\nHeadline: direct fg IoU, Dice\nAux: eval mIoU, eval ARI"];
          }

          input_image -> sam3;
          gt_mask -> loss [label="train target"];

          sam3 -> p2;
          sam3 -> p1;
          sam3 -> p0;
          freeze_tag -> sam3 [arrowhead=none, style=dashed, color="#7f8fb0"];

          p2 -> select_levels;
          p1 -> select_levels;
          p0 -> select_levels;

          select_levels -> project [label="chosen scales"];
          project -> resize;
          resize -> fuse_decode [label="aligned maps"];
          fuse_decode -> logits;
          head_params -> project [arrowhead=none, style=dashed, color="#58a37c"];
          head_params -> fuse_decode [arrowhead=none, style=dashed, color="#58a37c"];
          head_params -> logits [arrowhead=none, style=dashed, color="#58a37c"];

          logits -> loss [label="train only"];
          loss -> optimizer;
          optimizer -> head_params [style=dashed, color="#d17aa8", label="update head only"];

          logits -> sigmoid [label="eval only"];
          sigmoid -> threshold;
          threshold -> pred_fg [label=">= 0.5"];
          pred_fg -> pred_bg [style=dashed, color="#91a064", label="complement"];
          pred_fg -> metrics;

          split_note -> sam3 [style=invis, weight=0.2];
          optimizer -> threshold [style=invis, weight=0.2];
        }
        '''
    ).strip() + "\n"


def render_architecture_diagram() -> tuple[str, str]:
    dot_source = render_architecture_dot()
    dot_binary = find_graphviz_dot()
    rendered = subprocess.run(
        [dot_binary, "-Tsvg"],
        input=dot_source,
        text=True,
        capture_output=True,
        check=False,
    )
    if rendered.returncode != 0:
        raise RuntimeError(f"Graphviz render failed: {rendered.stderr.strip()}")
    return dot_source, rendered.stdout


def load_visual_rows(run_dir: Path) -> list[dict]:
    manifest_rows: dict[str, dict] = {}
    with require_file(run_dir / "visuals_manifest.jsonl").open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            crop_name = str(payload.get("crop_name"))
            visual_rel = Path(payload["visual_path"])
            manifest_rows[crop_name] = {
                "crop_name": crop_name,
                "sample_index": int(payload.get("sample_index", len(manifest_rows))),
                "caption": payload.get("caption", ""),
                "source_visual_path": require_file(run_dir / visual_rel),
            }

    metric_rows: dict[str, dict] = {}
    with require_file(run_dir / "per_sample_metrics.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metric_rows[row["crop_name"]] = row

    if set(manifest_rows) != set(metric_rows):
        raise RuntimeError(f"Visual/metric crop mismatch in {run_dir}")

    rows = []
    for crop_name, visual_row in manifest_rows.items():
        metric_row = metric_rows[crop_name]
        rows.append(
            {
                **visual_row,
                "direct_foreground_iou": float(metric_row["direct_foreground_iou"]),
                "direct_foreground_dice": float(metric_row["direct_foreground_dice"]),
                "eval_miou": float(metric_row["eval_miou"]),
                "eval_ari": float(metric_row["eval_ari"]),
            }
        )
    rows.sort(key=lambda row: row["sample_index"])
    return rows


def load_run(spec: RunSpec, output_root: Path) -> dict:
    run_dir = output_root / spec.source_dir
    for file_name in REQUIRED_RUN_FILES:
        require_file(run_dir / file_name)

    summary = load_json(run_dir / "summary.json")
    config = load_json(run_dir / "config.json")
    mean_metrics = summary["mean_metrics"]
    rows = load_visual_rows(run_dir)
    sample_lookup = {row["crop_name"]: row for row in rows}

    return {
        "run_id": spec.run_id,
        "label": spec.label,
        "short_label": spec.short_label,
        "source_dir": run_dir,
        "variant": summary.get("variant") or config.get("variant"),
        "variant_summary": summary.get("variant_summary") or config.get("variant_summary") or "",
        "num_samples": int(summary.get("num_evaluated_samples") or 0),
        "decoder_dim": config.get("decoder_dim"),
        "projection_dim": config.get("projection_dim"),
        "trainable_parameter_count": summary.get("trainable_parameter_count"),
        "supervision": spec.supervision,
        "frozen_backbone": spec.frozen_backbone,
        "learned_prompt_generator": spec.learned_prompt_generator,
        "notes": spec.notes,
        "fg_iou": float(mean_metrics["direct_foreground_iou"]),
        "dice": float(mean_metrics["direct_foreground_dice"]),
        "eval_miou": float(mean_metrics.get("eval_miou", mean_metrics.get("miou", 0.0))),
        "eval_ari": float(mean_metrics.get("eval_ari", mean_metrics.get("ari", 0.0))),
        "miou_agg": float(mean_metrics.get("miou_agg", 0.0)),
        "sample_lookup": sample_lookup,
    }


def enforce_alignment(runs: dict[str, dict], baseline_run_id: str) -> None:
    baseline = set(runs[baseline_run_id]["sample_lookup"])
    for run_id, run in runs.items():
        current = set(run["sample_lookup"])
        if current != baseline:
            diff = len(baseline.symmetric_difference(current))
            raise RuntimeError(f"Sample mismatch for {run_id}: {diff} mismatched crop ids relative to {baseline_run_id}")


def crop_triptych_panel(source_path: Path, panel_index: int) -> Image.Image:
    with Image.open(source_path) as image:
        image = image.convert("RGB")
        panel_width = image.width // 3
        if panel_width * 3 != image.width:
            raise RuntimeError(f"Unexpected triptych width for {source_path}: {image.width}")
        left = panel_width * panel_index
        return image.crop((left, 0, left + panel_width, image.height))


def compose_strip(images: list[Image.Image]) -> Image.Image:
    widths = [image.width for image in images]
    heights = [image.height for image in images]
    canvas = Image.new("RGB", (sum(widths), max(heights)), color=(255, 255, 255))
    cursor_x = 0
    for image in images:
        canvas.paste(image, (cursor_x, 0))
        cursor_x += image.width
    return canvas


def build_glas_story_blocks(runs: dict[str, dict]) -> list[dict]:
    blocks = []
    for story in QUAL_STORIES:
        crop_name = story["crop_name"]
        cfc_row = runs["cfc_baseline"]["sample_lookup"][crop_name]
        all_row = runs["all_scales_d64"]["sample_lookup"][crop_name]
        best_row = runs["fpn_2_only_d128"]["sample_lookup"][crop_name]

        asset_dir = DEST_DIR / "assets" / "gallery" / story["story_id"]
        panel_specs = [
            ("input", "Input", best_row["source_visual_path"], 0, None),
            ("gt", "GT", best_row["source_visual_path"], 1, None),
            ("cfc", "CFC baseline", cfc_row["source_visual_path"], 2, cfc_row),
            ("all_scales", "all_scales d64", all_row["source_visual_path"], 2, all_row),
            ("best", "fpn_2_only d128", best_row["source_visual_path"], 2, best_row),
        ]
        panels = []
        strip_images = []
        for key, label, source_visual, panel_index, row in panel_specs:
            image = crop_triptych_panel(source_visual, panel_index)
            strip_images.append(image.copy())
            rel = Path("assets") / "gallery" / story["story_id"] / f"{key}.webp"
            write_webp(image, DEST_DIR / rel, width=420 if key in {"input", "gt"} else 420, quality=82)
            panels.append(
                {
                    "key": key,
                    "label": label,
                    "asset_rel": rel.as_posix(),
                    "fg_iou": None if row is None else row["direct_foreground_iou"],
                    "dice": None if row is None else row["direct_foreground_dice"],
                    "eval_miou": None if row is None else row["eval_miou"],
                    "eval_ari": None if row is None else row["eval_ari"],
                }
            )

        strip_rel = Path("assets") / "gallery" / story["story_id"] / "strip.webp"
        write_webp(compose_strip(strip_images), DEST_DIR / strip_rel, width=1600, quality=82)
        blocks.append(
            {
                **story,
                "strip_rel": strip_rel.as_posix(),
                "panels": panels,
                "delta_label": "Δ best vs CFC",
                "delta_best_vs_cfc": best_row["direct_foreground_iou"] - cfc_row["direct_foreground_iou"],
                "delta_best_vs_all_scales": best_row["direct_foreground_iou"] - all_row["direct_foreground_iou"],
            }
        )
    return blocks


def build_monuseg_story_blocks(runs: dict[str, dict]) -> list[dict]:
    blocks = []
    for story in MONUSEG_QUAL_STORIES:
        crop_name = story["crop_name"]
        strict_row = runs["strict512_coarse"]["sample_lookup"][crop_name]
        native_coarse_row = runs["native_aug_coarse"]["sample_lookup"][crop_name]
        refine_row = runs["native_aug_refine"]["sample_lookup"][crop_name]

        panel_specs = [
            ("input", "Input", refine_row["source_visual_path"], 0, None),
            ("gt", "GT", refine_row["source_visual_path"], 1, None),
            ("strict512", "strict 512x512 coarse", strict_row["source_visual_path"], 2, strict_row),
            ("native_coarse", "native + aug coarse", native_coarse_row["source_visual_path"], 2, native_coarse_row),
            ("refine", "native + aug refine", refine_row["source_visual_path"], 2, refine_row),
        ]

        panels = []
        strip_images = []
        for key, label, source_visual, panel_index, row in panel_specs:
            image = crop_triptych_panel(source_visual, panel_index)
            strip_images.append(image.copy())
            rel = Path("assets") / "gallery" / story["story_id"] / f"{key}.webp"
            write_webp(image, DEST_DIR / rel, width=420, quality=82)
            panels.append(
                {
                    "key": key,
                    "label": label,
                    "asset_rel": rel.as_posix(),
                    "fg_iou": None if row is None else row["direct_foreground_iou"],
                    "dice": None if row is None else row["direct_foreground_dice"],
                    "eval_miou": None if row is None else row["eval_miou"],
                    "eval_ari": None if row is None else row["eval_ari"],
                }
            )

        strip_rel = Path("assets") / "gallery" / story["story_id"] / "strip.webp"
        write_webp(compose_strip(strip_images), DEST_DIR / strip_rel, width=1600, quality=82)
        blocks.append(
            {
                **story,
                "strip_rel": strip_rel.as_posix(),
                "panels": panels,
                "delta_label": "Δ refine vs strict512",
                "delta_best_vs_cfc": refine_row["direct_foreground_iou"] - strict_row["direct_foreground_iou"],
                "delta_best_vs_all_scales": refine_row["direct_foreground_iou"] - native_coarse_row["direct_foreground_iou"],
            }
        )
    return blocks


def build_gallery_payload(runs: dict[str, dict], story_blocks: list[dict]) -> dict:
    story_by_crop = {block["crop_name"]: block for block in story_blocks}
    records = []
    for crop_name, best_row in runs["fpn_2_only_d128"]["sample_lookup"].items():
        cfc_row = runs["cfc_baseline"]["sample_lookup"][crop_name]
        all_row = runs["all_scales_d64"]["sample_lookup"][crop_name]
        preview_rel = Path("assets") / "all_previews" / f"{crop_name}.webp"
        strip = compose_strip(
            [
                crop_triptych_panel(best_row["source_visual_path"], 0),
                crop_triptych_panel(best_row["source_visual_path"], 1),
                crop_triptych_panel(cfc_row["source_visual_path"], 2),
                crop_triptych_panel(all_row["source_visual_path"], 2),
                crop_triptych_panel(best_row["source_visual_path"], 2),
            ]
        )
        write_webp(strip, DEST_DIR / preview_rel, width=GALLERY_PREVIEW_WIDTH, quality=GALLERY_PREVIEW_QUALITY)

        records.append(
            {
                "sample_id": crop_name,
                "sample_index": best_row["sample_index"],
                "story_tag": story_by_crop.get(crop_name, {}).get("tag"),
                "best_fg_iou": best_row["direct_foreground_iou"],
                "best_dice": best_row["direct_foreground_dice"],
                "best_eval_miou": best_row["eval_miou"],
                "best_eval_ari": best_row["eval_ari"],
                "cfc_fg_iou": cfc_row["direct_foreground_iou"],
                "all_scales_fg_iou": all_row["direct_foreground_iou"],
                "delta_best_vs_cfc": best_row["direct_foreground_iou"] - cfc_row["direct_foreground_iou"],
                "delta_best_vs_all_scales": best_row["direct_foreground_iou"] - all_row["direct_foreground_iou"],
                "thumb_path": preview_rel.as_posix(),
            }
        )

    records.sort(key=lambda row: row["sample_index"])
    return {
        "sample_count": len(records),
        "story_count": len(story_blocks),
        "records": records,
    }


def build_results_rows(glas_runs: dict[str, dict], monuseg_runs: dict[str, dict]) -> list[dict]:
    rows = []
    for run_id in GLAS_RESULTS_ORDER:
        run = glas_runs[run_id]
        rows.append(
            {
                "dataset": "GlaS",
                "method": run["label"],
                "supervision": run["supervision"],
                "frozen_backbone": run["frozen_backbone"],
                "learned_prompt_generator": run["learned_prompt_generator"],
                "trainable_params": run["trainable_parameter_count"],
                "fg_iou": run["fg_iou"],
                "dice": run["dice"],
                "eval_miou": run["eval_miou"],
                "eval_ari": run["eval_ari"],
                "notes": run["notes"],
            }
        )
    rows.append(
        {
            "dataset": "GlaS",
            "method": "AutoSAM reported paper reference",
            "supervision": "dense gland masks (paper)",
            "frozen_backbone": "reported only",
            "learned_prompt_generator": "yes (reported)",
            "trainable_params": "learned prompt encoder (reported in paper)",
            "fg_iou": GLAS_AUTOSAM_REPORTED_FG_IOU,
            "dice": GLAS_AUTOSAM_REPORTED_DICE,
            "eval_miou": None,
            "eval_ari": None,
            "notes": "Reported paper reference only; not reproduced by us in this repo.",
        }
    )
    for run_id in ["strict512_coarse", "native_aug_coarse", "native_aug_refine"]:
        run = monuseg_runs[run_id]
        rows.append(
            {
                "dataset": "MoNuSeg",
                "method": run["label"],
                "supervision": run["supervision"],
                "frozen_backbone": run["frozen_backbone"],
                "learned_prompt_generator": run["learned_prompt_generator"],
                "trainable_params": run["trainable_parameter_count"],
                "fg_iou": run["fg_iou"],
                "dice": run["dice"],
                "eval_miou": run["eval_miou"],
                "eval_ari": run["eval_ari"],
                "notes": run["notes"],
            }
        )
    rows.append(
        {
            "dataset": "MoNuSeg",
            "method": "AutoSAM reported paper reference",
            "supervision": "dense nucleus masks (paper)",
            "frozen_backbone": "reported only",
            "learned_prompt_generator": "yes (reported)",
            "trainable_params": "learned prompt encoder (reported in paper)",
            "fg_iou": MONUSEG_AUTOSAM_REPORTED_FG_IOU,
            "dice": MONUSEG_AUTOSAM_REPORTED_DICE,
            "eval_miou": None,
            "eval_ari": None,
            "notes": "Reported paper reference only; current strongest local run is native-resolution and not resize-matched to the paper 512x512 setting.",
        }
    )
    return rows


def write_table_csv(results_rows: list[dict]) -> str:
    table_path = DEST_DIR / "assets" / "tables" / "main_results.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "method",
        "supervision",
        "frozen_backbone",
        "learned_prompt_generator",
        "trainable_params",
        "fg_iou",
        "dice",
        "eval_miou",
        "eval_ari",
        "notes",
    ]
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results_rows)
    return table_path.relative_to(DEST_DIR).as_posix()


def write_plot_assets(glas_runs: dict[str, dict], monuseg_runs: dict[str, dict]) -> dict[str, str]:
    glas_groups = [
        {
            "label": "CFC baseline",
            "sub_label": "training-free",
            "fg_iou": glas_runs["cfc_baseline"]["fg_iou"],
            "dice": glas_runs["cfc_baseline"]["dice"],
        },
        {
            "label": "all_scales d64",
            "sub_label": "frozen head",
            "fg_iou": glas_runs["all_scales_d64"]["fg_iou"],
            "dice": glas_runs["all_scales_d64"]["dice"],
        },
        {
            "label": "fpn_2_only d128",
            "sub_label": "readout study",
            "fg_iou": glas_runs["fpn_2_only_d128"]["fg_iou"],
            "dice": glas_runs["fpn_2_only_d128"]["dice"],
        },
        {
            "label": "fpn_2_only d128",
            "sub_label": "224 + autosam aug",
            "fg_iou": glas_runs["fpn_2_only_d128_224_autosamaug"]["fg_iou"],
            "dice": glas_runs["fpn_2_only_d128_224_autosamaug"]["dice"],
        },
        {
            "label": "AutoSAM",
            "sub_label": "paper reference",
            "fg_iou": GLAS_AUTOSAM_REPORTED_FG_IOU,
            "dice": GLAS_AUTOSAM_REPORTED_DICE,
            "reported_reference": True,
        },
    ]
    monuseg_groups = [
        {
            "label": "strict512 coarse",
            "sub_label": "clean anchor",
            "fg_iou": monuseg_runs["strict512_coarse"]["fg_iou"],
            "dice": monuseg_runs["strict512_coarse"]["dice"],
        },
        {
            "label": "native coarse",
            "sub_label": "autosam aug",
            "fg_iou": monuseg_runs["native_aug_coarse"]["fg_iou"],
            "dice": monuseg_runs["native_aug_coarse"]["dice"],
        },
        {
            "label": "native refine",
            "sub_label": "best current",
            "fg_iou": monuseg_runs["native_aug_refine"]["fg_iou"],
            "dice": monuseg_runs["native_aug_refine"]["dice"],
        },
        {
            "label": "AutoSAM",
            "sub_label": "paper reference",
            "fg_iou": MONUSEG_AUTOSAM_REPORTED_FG_IOU,
            "dice": MONUSEG_AUTOSAM_REPORTED_DICE,
            "reported_reference": True,
        },
    ]
    scale_groups = [
        {
            "label": "fpn_0_only",
            "sub_label": "d64",
            "fg_iou": glas_runs["fpn_0_only_d64"]["fg_iou"],
            "dice": glas_runs["fpn_0_only_d64"]["dice"],
        },
        {
            "label": "mid_plus_fine",
            "sub_label": "d64",
            "fg_iou": glas_runs["mid_plus_fine_d64"]["fg_iou"],
            "dice": glas_runs["mid_plus_fine_d64"]["dice"],
        },
        {
            "label": "all_scales",
            "sub_label": "d64",
            "fg_iou": glas_runs["all_scales_d64"]["fg_iou"],
            "dice": glas_runs["all_scales_d64"]["dice"],
        },
        {
            "label": "fpn_2_only",
            "sub_label": "d64",
            "fg_iou": glas_runs["fpn_2_only_d64"]["fg_iou"],
            "dice": glas_runs["fpn_2_only_d64"]["dice"],
        },
    ]
    capacity_points = [
        {
            "decoder_dim": glas_runs[run_id]["decoder_dim"],
            "trainable_parameter_count": glas_runs[run_id]["trainable_parameter_count"],
            "fg_iou": glas_runs[run_id]["fg_iou"],
            "dice": glas_runs[run_id]["dice"],
        }
        for run_id in GLAS_CAPACITY_SWEEP_IDS
    ]
    architecture_dot, architecture_svg = render_architecture_diagram()
    plots = {
        "method_architecture.svg": architecture_svg,
        "method_architecture.dot": architecture_dot,
        "supervision_ladder.svg": render_bar_chart(
            title="GlaS: CFC vs Frozen-Head Readout vs AutoSAM",
            subtitle="The GlaS story still starts with readout: weak training-free CFC, strong dense supervision, and a cleaner 224x224 AutoSAM-style endpoint.",
            groups=glas_groups,
            highlight_reported=True,
        ),
        "monuseg_readout_story.svg": render_bar_chart(
            title="MoNuSeg: Readout Architecture Matters More Than Augmentation Alone",
            subtitle="The native coarse-only run stays near the strict 512x512 anchor, but a tiny coarse+mid residual branch makes a substantial jump.",
            groups=monuseg_groups,
            highlight_reported=True,
        ),
        "scale_ablation.svg": render_bar_chart(
            title="GlaS Scale Ablation at d64",
            subtitle="On GlaS, the coarsest level alone is still the strongest strict scale choice.",
            groups=scale_groups,
        ),
        "capacity_sweep.svg": render_capacity_chart(
            title="GlaS Coarse-Only Capacity Sweep",
            subtitle="Width helps monotonically, but coarse-only d64 already beats all_scales d64 with fewer parameters.",
            points=capacity_points,
            all_scales_reference=glas_runs["all_scales_d64"],
        ),
    }
    for file_name, content in plots.items():
        write_text(DEST_DIR / "assets" / "plots" / file_name, content)
    return {name: f"assets/plots/{name}" for name in plots}


def build_metrics_payload(
    glas_runs: dict[str, dict],
    monuseg_runs: dict[str, dict],
    glas_story_blocks: list[dict],
    monuseg_story_blocks: list[dict],
    gallery_payload: dict,
) -> dict:
    glas_best = glas_runs["fpn_2_only_d128"]
    glas_protocol = glas_runs["fpn_2_only_d128_224_autosamaug"]
    coarse_d64 = glas_runs["fpn_2_only_d64"]
    all_scales = glas_runs["all_scales_d64"]
    cfc = glas_runs["cfc_baseline"]
    monu_strict = monuseg_runs["strict512_coarse"]
    monu_native_coarse = monuseg_runs["native_aug_coarse"]
    monu_best = monuseg_runs["native_aug_refine"]
    return {
        "status": "reviewed",
        "page_title": PAGE_TITLE,
        "headline": {
            "glas_best_readout_fg_iou": glas_best["fg_iou"],
            "glas_best_readout_dice": glas_best["dice"],
            "glas_protocol_fg_iou": glas_protocol["fg_iou"],
            "glas_protocol_dice": glas_protocol["dice"],
            "delta_best_vs_cfc_fg_iou": glas_best["fg_iou"] - cfc["fg_iou"],
            "delta_best_vs_cfc_dice": glas_best["dice"] - cfc["dice"],
            "delta_fpn2_d64_vs_all_scales_fg_iou": coarse_d64["fg_iou"] - all_scales["fg_iou"],
            "delta_fpn2_d64_vs_all_scales_dice": coarse_d64["dice"] - all_scales["dice"],
            "glas_gap_to_autosam_fg_iou": GLAS_AUTOSAM_REPORTED_FG_IOU - glas_protocol["fg_iou"],
            "glas_gap_to_autosam_dice": GLAS_AUTOSAM_REPORTED_DICE - glas_protocol["dice"],
            "monuseg_strict512_fg_iou": monu_strict["fg_iou"],
            "monuseg_best_fg_iou": monu_best["fg_iou"],
            "monuseg_delta_native_coarse_vs_strict_fg_iou": monu_native_coarse["fg_iou"] - monu_strict["fg_iou"],
            "monuseg_delta_refine_vs_strict_fg_iou": monu_best["fg_iou"] - monu_strict["fg_iou"],
            "monuseg_gap_to_autosam_fg_iou": MONUSEG_AUTOSAM_REPORTED_FG_IOU - monu_best["fg_iou"],
            "monuseg_gap_to_autosam_dice": MONUSEG_AUTOSAM_REPORTED_DICE - monu_best["dice"],
        },
        "glas_runs": {run_id: {key: value for key, value in run.items() if key not in {"sample_lookup", "source_dir"}} for run_id, run in glas_runs.items()},
        "monuseg_runs": {run_id: {key: value for key, value in run.items() if key not in {"sample_lookup", "source_dir"}} for run_id, run in monuseg_runs.items()},
        "autosam_reference": {
            "glas": {
                "fg_iou": GLAS_AUTOSAM_REPORTED_FG_IOU,
                "dice": GLAS_AUTOSAM_REPORTED_DICE,
                "source": f"{PHASE1_README}",
                "reported_only": True,
            },
            "monuseg": {
                "fg_iou": MONUSEG_AUTOSAM_REPORTED_FG_IOU,
                "dice": MONUSEG_AUTOSAM_REPORTED_DICE,
                "source": str(TEXTURE_REPO_ROOT / "outputs" / "autosam_cross_dataset_phase1" / "README.md"),
                "reported_only": True,
            },
        },
        "stories": {
            "glas": glas_story_blocks,
            "monuseg": monuseg_story_blocks,
        },
        "gallery": {
            "sample_count": gallery_payload["sample_count"],
            "story_count": gallery_payload["story_count"],
        },
        "notes": [
            "Training-free CFC on GlaS is materially weaker than the dense-supervised frozen-feature heads.",
            "GlaS still favors coarse-only, but MoNuSeg now favors a slightly richer coarse-plus-mid residual readout.",
            "On MoNuSeg, augmentation alone does not explain the jump; the architectural refinement does.",
            "AutoSAM values on this page are reported paper reference values, not reproduced runs in this repo.",
        ],
    }


def build_training_data(results_rows: list[dict], glas_story_blocks: list[dict], monuseg_story_blocks: list[dict], gallery_payload: dict) -> dict:
    return {
        "page": {
            "title": PAGE_TITLE,
            "date": PAGE_DATE,
            "status": "reviewed",
        },
        "results_rows": results_rows,
        "stories": {
            "glas": glas_story_blocks,
            "monuseg": monuseg_story_blocks,
        },
        "gallery": gallery_payload,
    }


def build_links_payload(table_csv_path: str, plot_paths: dict[str, str]) -> dict:
    return {
        "page": "index.html",
        "gallery": "gallery.html",
        "metrics": "metrics.json",
        "training_data": "training_data.json",
        "summary": "summary.md",
        "method_source": "method_source.md",
        "method_architecture_dot": plot_paths["method_architecture.dot"],
        "manifest": "manifest.yaml",
        "results_table_csv": table_csv_path,
        "plots": plot_paths,
    }


def build_manifest(total_story_count: int, gallery_payload: dict) -> str:
    return dedent(
        f"""\
        title: "{PAGE_TITLE}"
        model_ids:
          - "facebook/sam3"
          - "AutoSAM (reported reference)"
        dataset_ids:
          - "glas"
          - "monuseg"
        date: "{PAGE_DATE}"
        description: "Frozen SAM3 readout study across GlaS and MoNuSeg with current internal AutoSAM reference context."
        status: "reviewed"
        assets:
          story_block_count: {total_story_count}
          gallery_preview_count: {gallery_payload['sample_count']}
          plot_count: 5
        evaluation:
          primary_metrics:
            - "direct_foreground_iou"
            - "direct_foreground_dice"
          diagnostics:
            - "eval_miou"
            - "eval_ari"
        caveats:
          - "AutoSAM values are reported paper reference values."
          - "MoNuSeg native refine results are not resize-matched to the paper's explicit 512x512 route."
        """
    )


def render_findings_cards(glas_runs: dict[str, dict], monuseg_runs: dict[str, dict]) -> str:
    glas_best = glas_runs["fpn_2_only_d128"]
    glas_protocol = glas_runs["fpn_2_only_d128_224_autosamaug"]
    cfc = glas_runs["cfc_baseline"]
    monu_refine = monuseg_runs["native_aug_refine"]
    monu_strict = monuseg_runs["strict512_coarse"]
    return dedent(
        f"""\
        <div class="metric-grid compact-grid">
          <article class="metric">
            <div class="k">GlaS CFC vs Best Readout</div>
            <div class="v delta-positive">{signed(glas_best['fg_iou'] - cfc['fg_iou'])}</div>
            <div class="mini-note">Foreground IoU jump from training-free CFC to <code>fpn_2_only d128</code>.</div>
          </article>
          <article class="metric">
            <div class="k">Closest Clean GlaS Endpoint</div>
            <div class="v">{glas_protocol['fg_iou']:.6f}</div>
            <div class="mini-note">Foreground IoU {glas_protocol['fg_iou']:.6f} · Dice {glas_protocol['dice']:.6f} under <code>224x224 + autosam_dense_v1</code>.</div>
          </article>
          <article class="metric">
            <div class="k">MoNuSeg Residual Branch Payoff</div>
            <div class="v delta-positive">{signed(monu_refine['fg_iou'] - monu_strict['fg_iou'])}</div>
            <div class="mini-note">Foreground IoU gain over the strict <code>512x512</code> coarse-only anchor with only <code>{format_params(monu_refine['trainable_parameter_count'] - monu_strict['trainable_parameter_count'])}</code> extra params.</div>
          </article>
          <article class="metric">
            <div class="k">MoNuSeg Gap to AutoSAM</div>
            <div class="v">{MONUSEG_AUTOSAM_REPORTED_FG_IOU - monu_refine['fg_iou']:.4f}</div>
            <div class="mini-note">Remaining foreground IoU gap from the best current MoNuSeg run to the cited paper value.</div>
          </article>
        </div>
        """
    ).strip()


def render_cross_dataset_table(glas_runs: dict[str, dict], monuseg_runs: dict[str, dict]) -> str:
    glas_best = glas_runs["fpn_2_only_d128_224_autosamaug"]
    monu_best = monuseg_runs["native_aug_refine"]
    rows = [
        (
            "GlaS",
            "fpn_2_only d128 @ 224 + autosam aug",
            glas_best["fg_iou"],
            glas_best["dice"],
            GLAS_AUTOSAM_REPORTED_FG_IOU,
            GLAS_AUTOSAM_REPORTED_DICE,
            "Cleanest current endpoint; still a reported-reference comparison, not an AutoSAM reproduction.",
        ),
        (
            "MoNuSeg",
            "fpn_2+fpn_1 refine d128 @ native + autosam aug",
            monu_best["fg_iou"],
            monu_best["dice"],
            MONUSEG_AUTOSAM_REPORTED_FG_IOU,
            MONUSEG_AUTOSAM_REPORTED_DICE,
            "Strongest current row is close to the paper reference, but it is native-resolution rather than the paper's explicit 512x512 route.",
        ),
    ]
    rendered = []
    for dataset, method, fg_iou, dice, ref_iou, ref_dice, note in rows:
        rendered.append(
            dedent(
                f"""\
                <tr>
                  <td><strong>{dataset}</strong></td>
                  <td>{method}</td>
                  <td>{fg_iou:.6f}</td>
                  <td>{dice:.6f}</td>
                  <td>{ref_iou:.6f}</td>
                  <td>{ref_dice:.6f}</td>
                  <td>{ref_iou - fg_iou:.6f}</td>
                  <td>{ref_dice - dice:.6f}</td>
                  <td>{note}</td>
                </tr>
                """
            ).strip()
        )
    return "\n".join(rendered)


def render_results_table(rows: list[dict]) -> str:
    body_rows = []
    for row in rows:
        params = format_params(row["trainable_params"])
        fg_iou = fmt(row["fg_iou"])
        dice = fmt(row["dice"])
        eval_miou = fmt(row["eval_miou"])
        eval_ari = fmt(row["eval_ari"])
        body_rows.append(
            dedent(
                f"""\
                <tr>
                  <td><strong>{escape(str(row['dataset']))}</strong></td>
                  <td><strong>{escape(str(row['method']))}</strong></td>
                  <td>{escape(str(row['supervision']))}</td>
                  <td>{escape(str(row['frozen_backbone']))}</td>
                  <td>{escape(str(row['learned_prompt_generator']))}</td>
                  <td>{escape(params)}</td>
                  <td>{fg_iou}</td>
                  <td>{dice}</td>
                  <td>{eval_miou}</td>
                  <td>{eval_ari}</td>
                  <td>{escape(str(row['notes']))}</td>
                </tr>
                """
            ).strip()
        )
    return "\n".join(body_rows)


def render_story_blocks(blocks: list[dict]) -> str:
    rendered = []
    for block in blocks:
        figures = []
        for panel in block["panels"]:
            metric_line = (
                ""
                if panel["fg_iou"] is None
                else f'<div class="mini-note">Fg IoU {panel["fg_iou"]:.4f} · Dice {panel["dice"]:.4f}</div>'
            )
            figures.append(
                dedent(
                    f"""\
                    <figure>
                      <img src="{escape(panel['asset_rel'])}" alt="{escape(block['crop_name'])} — {escape(panel['label'])}" data-zoom-src="{escape(panel['asset_rel'])}" />
                      <figcaption>
                        <strong>{escape(panel['label'])}</strong>
                        {metric_line}
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
                    <span class="pill">{escape(block['crop_name'])}</span>
                    <span class="pill">{escape(block.get('delta_label', 'Δ'))} {signed(block['delta_best_vs_cfc'], 4)}</span>
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


def render_method_cards() -> str:
    return dedent(
        """\
        <div class="card-grid compact-grid">
          <article class="card">
            <h3>Problem Setup</h3>
            <p class="interpretation-copy">The method asks one narrow question: if SAM3 stays frozen, can a very small dense supervised readout already segment glands well on GlaS? The point is interpretability, not maximal engineering.</p>
            <p class="mini-note">No prompt generator, no prompt encoder changes, no transformer decoder, no attention blocks, no U-Net-scale decoder, and no SAM fine-tuning.</p>
          </article>
          <article class="card">
            <h3>Feature Path</h3>
            <p class="interpretation-copy">Frozen features come from <code>backbone_fpn</code>. The recorded GlaS runs expose <code>fpn_2</code>, <code>fpn_1</code>, and <code>fpn_0</code> with reference shapes <code>72x72</code>, <code>144x144</code>, and <code>288x288</code>.</p>
            <p class="mini-note">The dense mask-head family supports <code>all_scales</code>, <code>mid_plus_fine</code>, <code>coarse_plus_next_finer</code>, <code>fpn_2_only</code>, <code>fpn_1_only</code>, and <code>fpn_0_only</code>.</p>
          </article>
          <article class="card">
            <h3>Optimization Contract</h3>
            <p class="interpretation-copy">Only the head parameters are trainable. Optimization uses <code>AdamW</code> with <code>BCEWithLogits + Dice</code>. Dense-head defaults are <code>projection_dim=64</code>, <code>decoder_dim=64</code>, <code>GroupNorm(8)</code>, <code>lr=1e-3</code>, <code>wd=1e-4</code>, and threshold <code>0.5</code>.</p>
            <p class="mini-note">Headline metrics for the AutoSAM-style comparison are <code>direct_foreground_iou</code> and <code>direct_foreground_dice</code>. <code>eval_miou</code> and <code>eval_ari</code> stay auxiliary.</p>
          </article>
        </div>
        """
    ).strip()


def render_replication_cards() -> str:
    return dedent(
        """\
        <div class="card-grid compact-grid">
          <article class="card">
            <h3>Recorded Environment</h3>
            <p class="interpretation-copy">The saved runs used <code>./.venv/bin/python</code> with the shell prefix <code>PYTHONNOUSERSITE=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python</code>, seed <code>0</code>, device <code>cuda</code>, and model id <code>facebook/sam3</code>.</p>
            <p class="mini-note">Recorded package versions from the best current run: Pillow 12.1.1, datasets 4.7.0, numpy 1.26.4, torch 2.10.0, transformers 5.3.0.</p>
          </article>
          <article class="card">
            <h3>Audit First</h3>
            <p class="interpretation-copy">After any rerun, inspect <code>config.json</code>, <code>summary.json</code>, and <code>train_history.csv</code> before trusting the directory name or the page summary.</p>
            <p class="mini-note">The strongest example is <code>train_all_scales_test_e20_visuals</code>: the folder suffix says <code>e20</code>, but the saved metadata records <code>num_epochs=40</code>.</p>
          </article>
          <article class="card">
            <h3>Minimum Fields</h3>
            <p class="interpretation-copy">The replication-critical fields are <code>variant</code>, <code>selected_level_names</code>, <code>projection_dim</code>, <code>decoder_dim</code>, <code>num_epochs</code>, <code>trainable_parameter_count</code>, and the four headline metrics in <code>mean_metrics</code>.</p>
            <p class="mini-note">This is the smallest reliable audit set for catching accidental protocol drift.</p>
          </article>
        </div>
        """
    ).strip()


def render_monuseg_delta_cards(monuseg_runs: dict[str, dict]) -> str:
    strict_run = monuseg_runs["strict512_coarse"]
    native_coarse = monuseg_runs["native_aug_coarse"]
    refine = monuseg_runs["native_aug_refine"]
    return dedent(
        f"""\
        <div class="metric-grid compact-grid">
          <article class="metric">
            <div class="k">Native Coarse vs Strict 512</div>
            <div class="v">{signed(native_coarse['fg_iou'] - strict_run['fg_iou'], 4)}</div>
            <div class="mini-note">Foreground IoU delta from moving to native-resolution + AutoSAM-style augmentation without changing the head.</div>
          </article>
          <article class="metric">
            <div class="k">Residual Refine vs Strict 512</div>
            <div class="v delta-positive">{signed(refine['fg_iou'] - strict_run['fg_iou'], 4)}</div>
            <div class="mini-note">Foreground IoU gain after adding the tiny <code>fpn_1</code> residual branch.</div>
          </article>
          <article class="metric">
            <div class="k">Residual Refine vs Native Coarse</div>
            <div class="v delta-positive">{signed(refine['fg_iou'] - native_coarse['fg_iou'], 4)}</div>
            <div class="mini-note">Same training recipe, same backbone, different readout.</div>
          </article>
          <article class="metric">
            <div class="k">MoNuSeg Gap to AutoSAM</div>
            <div class="v">{MONUSEG_AUTOSAM_REPORTED_FG_IOU - refine['fg_iou']:.4f}</div>
            <div class="mini-note">Remaining foreground IoU gap from the strongest current MoNuSeg run to the cited paper value.</div>
          </article>
        </div>
        """
    ).strip()


def render_index_html(
    glas_runs: dict[str, dict],
    monuseg_runs: dict[str, dict],
    results_rows: list[dict],
    glas_story_blocks: list[dict],
    monuseg_story_blocks: list[dict],
    plot_paths: dict[str, str],
    table_csv_path: str,
) -> str:
    glas_best = glas_runs["fpn_2_only_d128"]
    glas_protocol = glas_runs["fpn_2_only_d128_224_autosamaug"]
    cfc = glas_runs["cfc_baseline"]
    coarse_d64 = glas_runs["fpn_2_only_d64"]
    all_scales = glas_runs["all_scales_d64"]
    monu_strict = monuseg_runs["strict512_coarse"]
    monu_refine = monuseg_runs["native_aug_refine"]
    return f"""<!DOCTYPE html>
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

    .callout-panel,
    .chart-card,
    .story-block,
    .note-card {{
      background: var(--surface-card);
      border: 1px solid var(--glass-border);
      border-radius: 20px;
      padding: 20px;
    }}

    .callout-panel {{
      display: grid;
      gap: 16px;
      grid-template-columns: 1.4fr 1fr;
      align-items: start;
    }}

    .callout-grid {{
      display: grid;
      gap: 10px;
    }}

    .callout-grid div {{
      display: grid;
      gap: 4px;
      padding: 14px;
      border: 1px solid var(--glass-border);
      border-radius: 16px;
      background: var(--surface-strong);
    }}

    .chart-grid {{
      display: grid;
      gap: 20px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-top: 24px;
    }}

    .chart-card img {{
      width: 100%;
      display: block;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: var(--surface-strong);
      cursor: zoom-in;
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
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
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

    .story-grid figcaption {{
      margin-top: 8px;
    }}

    .mini-note {{
      font-size: 0.78rem;
      color: var(--muted);
      margin-top: 4px;
    }}

    .artifact-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 20px;
    }}

    .code-panel {{
      background: var(--surface-card);
      border: 1px solid var(--glass-border);
      border-radius: 20px;
      padding: 20px;
    }}

    .code-panel pre {{
      margin: 14px 0 0;
      padding: 16px 18px;
      border-radius: 16px;
      background: var(--surface-strong);
      border: 1px solid var(--line);
      overflow-x: auto;
      color: var(--ink);
      font-size: 0.88rem;
      line-height: 1.5;
    }}

    .bullet-list {{
      margin: 16px 0 0;
      padding-left: 20px;
      color: var(--muted);
    }}

    .bullet-list li + li {{
      margin-top: 8px;
    }}

    .qual-note {{
      margin-top: 14px;
      color: var(--muted);
    }}

    @media (max-width: 960px) {{
      .callout-panel,
      .chart-grid {{
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
        <span class="tag">glas</span>
        <span class="tag">monuseg</span>
        <span class="tag">sam3</span>
        <span class="tag">autosam</span>
        <span class="tag">readout</span>
      </div>
      <div style="margin-top: 18px; display: flex; justify-content: center; gap: 12px; flex-wrap: wrap;">
        <a class="btn secondary" href="../../index.html">← Back to Dashboard</a>
        <a class="btn" href="gallery.html">Open Lightweight Gallery →</a>
      </div>
    </header>

    <section class="executive-summary">
      <h2>Executive Synopsis</h2>
      <p>GlaS still answers the original question cleanly: weak training-free CFC does <strong>not</strong> imply missing gland information in frozen SAM3. A tiny supervised head is already strong, and the GlaS scale story remains surprisingly coarse-dominant.</p>
      <p>MoNuSeg adds the more important AutoSAM-comparison update. The new <strong><code>fpn_2 + fpn_1</code> residual-refinement</strong> readout reaches <strong>fg IoU {monu_refine['fg_iou']:.6f}</strong> and <strong>Dice {monu_refine['dice']:.6f}</strong>, leaving only a small gap to the cited MoNuSeg AutoSAM paper reference. The combined picture is now more precise: the frozen features are strong, but the best readout is dataset-dependent.</p>
    </section>

    <section class="section">
      <h2>Headline Findings</h2>
      {render_findings_cards(glas_runs, monuseg_runs)}
    </section>

    <section class="section">
      <h2>Protocol / Fairness</h2>
      <div class="callout-panel">
        <div>
          <span class="tag warn">protocol caveat</span>
          <h3>Serious Internal Comparison, Still Not a Reproduced AutoSAM Benchmark</h3>
          <p class="interpretation-copy">AutoSAM values on this page are still reported paper references. The new evidence is valuable because it separates two different questions: whether frozen SAM3 features already carry the right signal, and whether the current readout family is expressive enough to use it well on each dataset.</p>
        </div>
        <div class="callout-grid">
          <div><strong>Datasets</strong><span>GlaS and MoNuSeg</span></div>
          <div><strong>Splits</strong><span>GlaS train → test, MoNuSeg official 30 → 14 or native test</span></div>
          <div><strong>SAM backbone frozen</strong><span>yes</span></div>
          <div><strong>Supervision</strong><span>dense binary gland or nucleus masks</span></div>
          <div><strong>Loss</strong><span>BCEWithLogits + Dice</span></div>
          <div><strong>AutoSAM values here</strong><span>reported paper references only</span></div>
          <div><strong>GlaS closest clean endpoint</strong><span><code>224x224 + autosam_dense_v1</code> final checkpoint</span></div>
          <div><strong>MoNuSeg strongest current run</strong><span>native-resolution <code>fpn_2 + fpn_1</code> residual refine</span></div>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>Method</h2>
      <p class="interpretation-copy">The page now includes the core method details from the internal phase-2 method note. This is the dense-supervised frozen-feature baseline actually used in the saved runs, not a paraphrased AutoSAM surrogate.</p>
      <div style="margin-top: 20px;">
        {render_method_cards()}
      </div>
      <div class="chart-grid">
        <article class="chart-card">
          <img src="{plot_paths['method_architecture.svg']}" alt="Frozen SAM GlaS dense-head architecture diagram" data-zoom-src="{plot_paths['method_architecture.svg']}" />
        </article>
        <article class="note-card">
          <h3>Architecture Readout</h3>
          <p class="interpretation-copy">This figure is now generated from a Graphviz DOT source, then rendered to SVG for the site. The main spine shows the shared frozen-feature path, while the lower branches make the training loss/update loop and the inference thresholding/metric path explicit.</p>
          <p class="interpretation-copy" style="margin-top: 14px;">That design choice matters for the page story: it keeps the head deliberately small enough that poor results still say something about the readout itself, not about a giant downstream decoder quietly doing most of the work.</p>
          <div class="artifact-row">
            <a class="btn secondary" href="method_source.md">Open Method Notes</a>
            <a class="btn secondary" href="assets/plots/method_architecture.dot">Open DOT Source</a>
          </div>
        </article>
      </div>
    </section>

    <section class="section">
      <h2>GlaS Readout Story</h2>
      <p class="interpretation-copy">The ladder below keeps the original GlaS question visible. Training-free CFC is weak, dense supervision changes the picture completely, and the cleanest current <code>224x224 + autosam_dense_v1</code> endpoint improves again over the earlier readout-study checkpoint.</p>
      <div class="chart-grid">
        <article class="chart-card">
          <img src="{plot_paths['supervision_ladder.svg']}" alt="Foreground metrics ladder from CFC to AutoSAM reference" data-zoom-src="{plot_paths['supervision_ladder.svg']}" />
        </article>
        <article class="note-card">
          <h3>What GlaS Still Says</h3>
          <p class="interpretation-copy">The jump from the training-free CFC baseline to the original best frozen head is <strong>{signed(glas_best['fg_iou'] - cfc['fg_iou'], 6)}</strong> in foreground IoU and <strong>{signed(glas_best['dice'] - cfc['dice'], 6)}</strong> in Dice. That is far too large to explain away as a minor evaluator mismatch.</p>
          <p class="interpretation-copy" style="margin-top: 14px;">Using the cleaner <code>224x224 + autosam_dense_v1</code> endpoint, the remaining GlaS gap to the cited AutoSAM reference is <strong>{GLAS_AUTOSAM_REPORTED_FG_IOU - glas_protocol['fg_iou']:.6f}</strong> in foreground IoU and <strong>{GLAS_AUTOSAM_REPORTED_DICE - glas_protocol['dice']:.6f}</strong> in Dice.</p>
        </article>
      </div>
    </section>

    <section class="section">
      <h2>MoNuSeg Extension</h2>
      <p class="interpretation-copy">MoNuSeg is the more important AutoSAM-comparison update. The strict <code>512x512</code> coarse-only anchor is clean and useful, the new native coarse-only run is basically flat, and the real movement comes when the readout gets a tiny <code>fpn_1</code> residual refinement branch.</p>
      <div class="chart-grid">
        <article class="chart-card">
          <img src="{plot_paths['monuseg_readout_story.svg']}" alt="MoNuSeg readout comparison against AutoSAM reference" data-zoom-src="{plot_paths['monuseg_readout_story.svg']}" />
        </article>
        <article class="note-card">
          <h3>What Changed on MoNuSeg</h3>
          <p class="interpretation-copy">The native coarse-only run changes little relative to the strict <code>512x512</code> anchor: it shifts foreground IoU by only <strong>{signed(monuseg_runs['native_aug_coarse']['fg_iou'] - monu_strict['fg_iou'], 6)}</strong> and Dice by <strong>{signed(monuseg_runs['native_aug_coarse']['dice'] - monu_strict['dice'], 6)}</strong>. The important move is architectural: the residual branch lifts foreground IoU by <strong>{signed(monu_refine['fg_iou'] - monu_strict['fg_iou'], 6)}</strong> and Dice by <strong>{signed(monu_refine['dice'] - monu_strict['dice'], 6)}</strong> over the strict anchor.</p>
          <p class="interpretation-copy" style="margin-top: 14px;">That leaves only <strong>{MONUSEG_AUTOSAM_REPORTED_FG_IOU - monu_refine['fg_iou']:.6f}</strong> IoU and <strong>{MONUSEG_AUTOSAM_REPORTED_DICE - monu_refine['dice']:.6f}</strong> Dice to the cited paper reference. It is still not protocol-identical, but it is no longer a “frozen features are far away” story.</p>
        </article>
      </div>
      <div style="margin-top: 20px;">
        {render_monuseg_delta_cards(monuseg_runs)}
      </div>
    </section>

    <section class="section">
      <h2>Cross-Dataset Snapshot</h2>
      <div class="table-wrap" style="margin-top: 16px;">
        <table>
          <thead>
            <tr>
              <th>Dataset</th>
              <th>Current strongest frozen run</th>
              <th>Fg IoU</th>
              <th>Dice</th>
              <th>AutoSAM IoU</th>
              <th>AutoSAM Dice</th>
              <th>IoU gap</th>
              <th>Dice gap</th>
              <th>Interpretation</th>
            </tr>
          </thead>
          <tbody>
            {render_cross_dataset_table(glas_runs, monuseg_runs)}
          </tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <h2>GlaS Scale Ablations</h2>
      <div class="chart-grid">
        <article class="chart-card">
          <img src="{plot_paths['scale_ablation.svg']}" alt="Scale ablation at d64" data-zoom-src="{plot_paths['scale_ablation.svg']}" />
        </article>
        <article class="chart-card">
          <img src="{plot_paths['capacity_sweep.svg']}" alt="Coarse-only capacity sweep" data-zoom-src="{plot_paths['capacity_sweep.svg']}" />
        </article>
      </div>
      <p class="qual-note">The GlaS readout is qualitative as much as numeric: dropping from <code>all_scales d64</code> to <code>fpn_2_only d64</code> already improves the result while using fewer parameters, and extra width on top of the coarse-only branch helps again. That is why the page still frames GlaS as <strong>coarse-dominant</strong>, not just “bigger head wins”.</p>
    </section>

    <section class="section">
      <h2>Main Results Table</h2>
      <div class="table-wrap" style="margin-top: 16px;">
        <table>
          <thead>
            <tr>
              <th>Dataset</th>
              <th>Method</th>
              <th>Supervision</th>
              <th>Frozen backbone?</th>
              <th>Learned prompt generator?</th>
              <th>Trainable params</th>
              <th>Fg IoU</th>
              <th>Dice</th>
              <th>eval mIoU</th>
              <th>eval ARI</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody>
            {render_results_table(results_rows)}
          </tbody>
        </table>
      </div>
      <div class="artifact-row">
        <a class="btn secondary" href="{table_csv_path}">Open Results CSV</a>
      </div>
    </section>

    <section class="section">
      <h2>Replication Details</h2>
      <p class="interpretation-copy">The commands below are the compact replication ladder for the main GlaS and MoNuSeg comparison points used on this page. They are enough to smoke-test the path, reproduce the clean GlaS endpoint, reproduce the strict MoNuSeg anchor, and rerun the current best MoNuSeg refine variant.</p>
      <div style="margin-top: 20px;">
        {render_replication_cards()}
      </div>
      <div class="chart-grid">
        <article class="code-panel">
          <h3>Shared Prefix + Smoke Test</h3>
          <pre><code>PYTHONNOUSERSITE=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ./.venv/bin/python main.py train-glas-frozen-sam-mask-head \\
  --dataset-root datasets/GlaS \\
  --device cuda \\
  --variant all_scales \\
  --train-limit 1 \\
  --eval-limit 1 \\
  --num-epochs 1 \\
  --output-dir outputs/glas_binary/frozen_sam_mask_head/smoke_all_scales_train_to_test_1x1 \\
  --no-save-visuals</code></pre>
        </article>
        <article class="code-panel">
          <h3>GlaS Clean Endpoint</h3>
          <pre><code># GlaS 224x224 + autosam_dense_v1 final checkpoint
PYTHONNOUSERSITE=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ./.venv/bin/python main.py train-glas-frozen-sam-mask-head \\
  --dataset-root datasets/GlaS \\
  --device cuda \\
  --variant fpn_2_only \\
  --projection-dim 128 \\
  --decoder-dim 128 \\
  --resize-height 224 \\
  --resize-width 224 \\
  --train-augmentation-policy autosam_dense_v1 \\
  --eval-every-epochs 1 \\
  --num-epochs 200 \\
  --output-dir outputs/glas_binary/frozen_sam_mask_head/train_fpn_2_only_d128_224_autosamaug_e200</code></pre>
        </article>
        <article class="code-panel">
          <h3>MoNuSeg Anchor + Best Current Run</h3>
          <pre><code># strict 512x512 anchor
PYTHONNOUSERSITE=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ./.venv/bin/python main.py train-monuseg-frozen-sam-mask-head \\
  --dataset-name RationAI/MoNuSeg \\
  --device cuda \\
  --variant fpn_2_only \\
  --projection-dim 128 \\
  --decoder-dim 128 \\
  --resize-height 512 \\
  --resize-width 512 \\
  --num-epochs 40 \\
  --output-dir outputs/monuseg_binary/frozen_sam_mask_head/train_fpn_2_only_d128_512_test_e40

# best current MoNuSeg refine run
PYTHONNOUSERSITE=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ./.venv/bin/python main.py train-monuseg-frozen-sam-mask-head \\
  --dataset-name RationAI/MoNuSeg \\
  --device cuda \\
  --variant fpn_2_plus_fpn_1_refine \\
  --projection-dim 128 \\
  --decoder-dim 128 \\
  --train-augmentation-policy autosam_dense_v1 \\
  --num-epochs 20 \\
  --output-dir outputs/monuseg_binary/frozen_sam_mask_head/train_fpn_2_plus_fpn_1_refine_d128_native_autosamaug_e20</code></pre>
        </article>
      </div>
      <div class="artifact-row">
        <a class="btn secondary" href="method_source.md">method_source.md</a>
        <a class="btn secondary" href="assets/plots/method_architecture.dot">method_architecture.dot</a>
      </div>
    </section>

    <section class="section">
      <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 20px; flex-wrap: wrap;">
        <div>
          <h2>GlaS Qualitative Comparison</h2>
          <p class="interpretation-copy">Each row below uses the repo-native GlaS triptych panels, cropped back into explicit <strong>Input</strong>, <strong>GT</strong>, and method-specific prediction columns. That keeps the page faithful to the saved run artifacts instead of fabricating a new qualitative format.</p>
        </div>
        <a class="btn" href="gallery.html">Open GlaS Gallery →</a>
      </div>
      <div style="margin-top: 24px;">
        {render_story_blocks(glas_story_blocks)}
      </div>
    </section>

    <section class="section">
      <h2>MoNuSeg Qualitative Comparison</h2>
      <p class="interpretation-copy">The MoNuSeg rows below use the same triptych-cropping approach, but the columns are now <strong>Input | GT | strict 512x512 coarse | native + aug coarse | native + aug refine</strong>. This section exists because the MoNuSeg average is not just a small training tweak: the residual branch visibly changes the prediction behavior.</p>
      <div style="margin-top: 24px;">
        {render_story_blocks(monuseg_story_blocks)}
      </div>
    </section>

    <section class="section">
      <h2>Interpretation</h2>
      <p class="interpretation-copy">The current evidence does <strong>not</strong> say that SAM3 is simply “better than AutoSAM” or that the problem is solved. It does say something narrower but important: weak training-free CFC on GlaS should not be read as evidence that frozen SAM3 lacks usable histology signal, and the new MoNuSeg runs make the same point from a different angle.</p>
      <p class="interpretation-copy" style="margin-top: 14px;">Across both datasets, the frozen representation looks much stronger than the weakest readout made it appear. The best readout is not universal, though. GlaS still favors a coarse-only head, while MoNuSeg benefits substantially from adding a tiny mid-scale residual correction. That makes the current lesson mostly about <strong>readout design</strong>, not feature absence.</p>
    </section>

    <section class="section">
      <h2>Caveats / Next Step</h2>
      <ul class="bullet-list">
        <li>AutoSAM on this page is a <strong>reported paper reference</strong>, not a reproduced run from this repo.</li>
        <li>The GlaS page view now includes the clean <code>224x224 + autosam_dense_v1</code> endpoint, but the cited AutoSAM numbers are still external reported references rather than a local reproduction.</li>
        <li>The strongest MoNuSeg run on this page is <strong>not</strong> resize-matched to the paper’s explicit <code>512x512</code> route; the strict <code>512x512</code> coarse-only anchor remains the clean apples-to-apples baseline.</li>
        <li><code>fpn_2_only</code> is labeled explicitly as <code>d64</code> or <code>d128</code> throughout this page so the default-width run is no longer ambiguous.</li>
        <li>The next high-value move is to run the <code>fpn_2 + fpn_1</code> residual readout under the strict MoNuSeg <code>512x512</code> route, rather than spending more effort on augmentation-only sweeps.</li>
      </ul>
      <div class="artifact-row">
        <a class="btn secondary" href="method_source.md">method_source.md</a>
        <a class="btn secondary" href="assets/plots/method_architecture.dot">method_architecture.dot</a>
        <a class="btn secondary" href="metrics.json">metrics.json</a>
        <a class="btn secondary" href="training_data.json">training_data.json</a>
        <a class="btn secondary" href="summary.md">summary.md</a>
      </div>
      <div style="margin-top: 28px; text-align: center;">
        <a class="btn secondary" href="../../index.html">← Back to Dashboard</a>
      </div>
    </section>

    <footer>
      <p>GlaS + MoNuSeg vs AutoSAM report | Rendered {PAGE_DATE}</p>
    </footer>
  </div>

  <dialog class="image-dialog" id="imageDialog">
    <img id="imageDialogImg" alt="Expanded preview" />
  </dialog>

  <script src="../../assets/site.js"></script>
</body>

</html>
"""


def render_gallery_html() -> str:
    return f"""<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{PAGE_TITLE} — GlaS Gallery</title>
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
    .controls {{
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      margin-bottom: 20px;
    }}

    .control {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}

    .control label {{
      font-size: 0.8rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-weight: 700;
      color: var(--muted);
    }}

    .control select,
    .control input {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 14px;
      background: var(--surface-strong);
      color: var(--ink);
      font: inherit;
    }}

    .gallery-card {{
      border-radius: 18px;
      overflow: hidden;
      background: var(--surface-strong);
      border: 1px solid var(--line);
      display: flex;
      flex-direction: column;
    }}

    .gallery-card img {{
      width: 100%;
      display: block;
      cursor: zoom-in;
      background: var(--surface-strong);
    }}

    .gallery-card .body {{
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}

    .gallery-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}

    .status-row {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
      margin-bottom: 18px;
      color: var(--muted);
    }}

    .load-more-wrap {{
      margin-top: 24px;
      text-align: center;
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

    <header style="text-align: center; margin-bottom: 42px;">
      <div class="subtitle">Full Gallery</div>
      <h1 class="title-gradient">GlaS vs AutoSAM — GlaS Gallery</h1>
      <p style="color: var(--muted); margin-top: 16px; font-weight: 300; max-width: 860px; margin-inline: auto;">
        Each strip is <strong>Input | GT | CFC baseline | all_scales d64 | fpn_2_only d128</strong>. The source run panels are cropped into this comparison view directly from the saved experiment visuals.
      </p>
      <div style="margin-top: 18px; display: flex; justify-content: center; gap: 12px; flex-wrap: wrap;">
        <a class="btn secondary" href="index.html">← Back to Report</a>
        <a class="btn secondary" href="../../index.html">← Back to Dashboard</a>
      </div>
    </header>

    <section class="section">
      <div class="controls">
        <div class="control">
          <label for="storyFilter">Story picks</label>
          <select id="storyFilter">
            <option value="all">All samples</option>
            <option value="story">Story picks only</option>
          </select>
        </div>
        <div class="control">
          <label for="sortBy">Sort</label>
          <select id="sortBy">
            <option value="sample">Sample order</option>
            <option value="best_fg_desc">Best fg IoU ↓</option>
            <option value="best_fg_asc">Best fg IoU ↑</option>
            <option value="delta_cfc_desc">Δ best vs CFC ↓</option>
            <option value="delta_cfc_asc">Δ best vs CFC ↑</option>
            <option value="delta_all_desc">Δ best vs all_scales ↓</option>
            <option value="delta_all_asc">Δ best vs all_scales ↑</option>
          </select>
        </div>
        <div class="control">
          <label for="pageSize">Per page</label>
          <select id="pageSize">
            <option value="12">12</option>
            <option value="24">24</option>
            <option value="48">48</option>
            <option value="80">80</option>
          </select>
        </div>
      </div>

      <div class="status-row">
        <div id="gallerySummary">Loading gallery…</div>
        <div id="galleryHint">Click any strip to expand. Arrow keys move between samples and auto-load more when needed.</div>
      </div>

      <div class="gallery" id="fullGallery"></div>

      <div class="load-more-wrap">
        <button id="loadMoreButton" type="button" class="btn">Load More</button>
      </div>

      <div style="margin-top: 28px; display: flex; justify-content: center; gap: 12px; flex-wrap: wrap;">
        <a class="btn secondary" href="index.html">← Back to Report</a>
        <a class="btn secondary" href="../../index.html">← Back to Dashboard</a>
      </div>
    </section>

    <footer>
      <p>GlaS gallery bundle | Lightweight previews only</p>
    </footer>
  </div>

  <dialog class="image-dialog" id="imageDialog">
    <img id="imageDialogImg" alt="Expanded preview" />
  </dialog>

  <script src="../../assets/site.js"></script>
  <script>
    (async () => {{
      const response = await fetch('training_data.json');
      if (!response.ok) {{
        throw new Error(`Failed to load training_data.json: ${{response.status}}`);
      }}
      const data = await response.json();
      const records = data.gallery.records.slice();
      const storyFilter = document.getElementById('storyFilter');
      const sortBy = document.getElementById('sortBy');
      const pageSize = document.getElementById('pageSize');
      const gallery = document.getElementById('fullGallery');
      const summary = document.getElementById('gallerySummary');
      const loadMoreButton = document.getElementById('loadMoreButton');
      const dialog = document.getElementById('imageDialog');
      const dialogImage = document.getElementById('imageDialogImg');
      const params = new URLSearchParams(window.location.search);
      let activeZoomIndex = -1;

      storyFilter.value = params.get('story') || 'all';
      sortBy.value = params.get('sort') || 'sample';
      pageSize.value = params.get('page_size') || '12';
      let shown = Number(params.get('shown')) || Number(pageSize.value);

      function syncUrl() {{
        const next = new URLSearchParams();
        if (storyFilter.value !== 'all') next.set('story', storyFilter.value);
        if (sortBy.value !== 'sample') next.set('sort', sortBy.value);
        if (pageSize.value !== '12') next.set('page_size', pageSize.value);
        if (shown > Number(pageSize.value)) next.set('shown', String(shown));
        const query = next.toString();
        history.replaceState(null, '', query ? `?${{query}}` : 'gallery.html');
      }}

      function filteredRecords() {{
        let items = records.slice();
        if (storyFilter.value === 'story') {{
          items = items.filter((item) => Boolean(item.story_tag));
        }}
        if (sortBy.value === 'best_fg_desc') {{
          items.sort((a, b) => b.best_fg_iou - a.best_fg_iou || a.sample_index - b.sample_index);
        }} else if (sortBy.value === 'best_fg_asc') {{
          items.sort((a, b) => a.best_fg_iou - b.best_fg_iou || a.sample_index - b.sample_index);
        }} else if (sortBy.value === 'delta_cfc_desc') {{
          items.sort((a, b) => b.delta_best_vs_cfc - a.delta_best_vs_cfc || a.sample_index - b.sample_index);
        }} else if (sortBy.value === 'delta_cfc_asc') {{
          items.sort((a, b) => a.delta_best_vs_cfc - b.delta_best_vs_cfc || a.sample_index - b.sample_index);
        }} else if (sortBy.value === 'delta_all_desc') {{
          items.sort((a, b) => b.delta_best_vs_all_scales - a.delta_best_vs_all_scales || a.sample_index - b.sample_index);
        }} else if (sortBy.value === 'delta_all_asc') {{
          items.sort((a, b) => a.delta_best_vs_all_scales - b.delta_best_vs_all_scales || a.sample_index - b.sample_index);
        }} else {{
          items.sort((a, b) => a.sample_index - b.sample_index);
        }}
        return items;
      }}

      function card(item) {{
        const storyTag = item.story_tag ? `<span class="tag good">${{item.story_tag}}</span>` : '';
        const deltaVsAll = item.delta_best_vs_all_scales > 0.0005
          ? `<span class="tag good">Δ vs all_scales +${{item.delta_best_vs_all_scales.toFixed(4)}}</span>`
          : item.delta_best_vs_all_scales < -0.0005
            ? `<span class="tag warn">Δ vs all_scales ${{item.delta_best_vs_all_scales.toFixed(4)}}</span>`
            : '<span class="tag">Δ vs all_scales 0.0000</span>';
        return `
          <article class="gallery-card">
            <img src="${{item.thumb_path}}" alt="${{item.sample_id}} comparison strip" data-zoom-src="${{item.thumb_path}}" />
            <div class="body">
              <div class="gallery-meta">
                <span class="pill">${{item.sample_id}}</span>
                <span class="tag">best fg IoU ${{item.best_fg_iou.toFixed(4)}}</span>
                <span class="tag">Δ vs CFC ${{item.delta_best_vs_cfc.toFixed(4)}}</span>
                ${{deltaVsAll}}
                ${{storyTag}}
              </div>
              <div><strong>Best Dice ${{item.best_dice.toFixed(4)}}</strong> · Best eval mIoU ${{item.best_eval_miou.toFixed(4)}} · Best eval ARI ${{item.best_eval_ari.toFixed(4)}}</div>
              <div class="mini-note">Left to right: Input, GT, CFC baseline, all_scales d64, fpn_2_only d128.</div>
            </div>
          </article>
        `;
      }}

      function loadedZoomNodes() {{
        return Array.from(gallery.querySelectorAll('[data-zoom-src]'));
      }}

      function openZoomAt(index) {{
        if (!dialog || !dialogImage) return false;
        const nodes = loadedZoomNodes();
        const node = nodes[index];
        if (!node) return false;
        activeZoomIndex = index;
        dialogImage.src = node.getAttribute('data-zoom-src');
        dialogImage.alt = node.alt || 'Expanded preview';
        if (!dialog.open) {{
          dialog.showModal();
        }}
        return true;
      }}

      function render(options = {{}}) {{
        const preserveZoom = Boolean(options.preserveZoom);
        const focusIndex = Number.isInteger(options.focusIndex) ? options.focusIndex : null;
        const items = filteredRecords();
        const limit = Math.min(shown, items.length);
        gallery.innerHTML = items.slice(0, limit).map(card).join('');
        summary.textContent = `Showing ${{limit}} of ${{items.length}} sample strips across ${{data.gallery.sample_count}} GlaS test crops.`;
        loadMoreButton.hidden = limit >= items.length;
        loadMoreButton.disabled = limit >= items.length;
        if (preserveZoom && focusIndex !== null) {{
          openZoomAt(focusIndex);
        }} else if (dialog?.open) {{
          dialog.close();
          activeZoomIndex = -1;
        }}
        syncUrl();
      }}

      function moveZoom(step) {{
        const items = filteredRecords();
        if (!items.length) return;
        if (activeZoomIndex < 0) {{
          openZoomAt(step > 0 ? 0 : Math.min(Math.min(shown, items.length), items.length) - 1);
          return;
        }}
        const nextIndex = activeZoomIndex + step;
        if (nextIndex < 0 || nextIndex >= items.length) {{
          return;
        }}
        if (nextIndex >= loadedZoomNodes().length) {{
          shown = Math.min(shown + Number(pageSize.value), items.length);
          render({{ preserveZoom: true, focusIndex: nextIndex }});
          return;
        }}
        openZoomAt(nextIndex);
      }}

      gallery.addEventListener('click', (event) => {{
        const target = event.target.closest('[data-zoom-src]');
        if (!target) return;
        activeZoomIndex = loadedZoomNodes().indexOf(target);
      }});

      dialog?.addEventListener('close', () => {{
        activeZoomIndex = -1;
      }});

      document.addEventListener('keydown', (event) => {{
        if (!dialog?.open) return;
        if (event.key === 'ArrowRight') {{
          event.preventDefault();
          moveZoom(1);
        }} else if (event.key === 'ArrowLeft') {{
          event.preventDefault();
          moveZoom(-1);
        }}
      }});

      storyFilter.addEventListener('change', () => {{
        shown = Number(pageSize.value);
        render();
      }});
      sortBy.addEventListener('change', () => {{
        shown = Number(pageSize.value);
        render();
      }});
      pageSize.addEventListener('change', () => {{
        shown = Number(pageSize.value);
        render();
      }});
      loadMoreButton.addEventListener('click', () => {{
        shown += Number(pageSize.value);
        render();
      }});

      render();
    }})().catch((error) => {{
      document.getElementById('gallerySummary').textContent = error.message;
      document.getElementById('loadMoreButton').hidden = true;
    }});
  </script>
</body>

</html>
"""


def render_summary_md(glas_runs: dict[str, dict], monuseg_runs: dict[str, dict]) -> str:
    glas_best = glas_runs["fpn_2_only_d128"]
    glas_protocol = glas_runs["fpn_2_only_d128_224_autosamaug"]
    cfc = glas_runs["cfc_baseline"]
    all_scales = glas_runs["all_scales_d64"]
    monu_strict = monuseg_runs["strict512_coarse"]
    monu_best = monuseg_runs["native_aug_refine"]
    return dedent(
        f"""\
        # {PAGE_TITLE}

        - Rendered: `{PAGE_DATE}`
        - Question: across GlaS and MoNuSeg, do frozen SAM3 features already carry the right histology signal for an AutoSAM-style comparison, or is the main bottleneck the readout?
        - Current answer: the frozen features look much stronger than the weakest readout suggested. GlaS still favors coarse-only, while MoNuSeg now improves sharply when a tiny mid-scale residual branch is added.

        ## Headline Numbers

        - Training-free CFC baseline: `fg_iou={cfc['fg_iou']:.6f}` `dice={cfc['dice']:.6f}` `eval_miou={cfc['eval_miou']:.6f}` `eval_ari={cfc['eval_ari']:.6f}`
        - Frozen mask head all_scales d64: `fg_iou={all_scales['fg_iou']:.6f}` `dice={all_scales['dice']:.6f}` `eval_miou={all_scales['eval_miou']:.6f}` `eval_ari={all_scales['eval_ari']:.6f}`
        - Best GlaS readout-study run (`fpn_2_only d128`): `fg_iou={glas_best['fg_iou']:.6f}` `dice={glas_best['dice']:.6f}` `eval_miou={glas_best['eval_miou']:.6f}` `eval_ari={glas_best['eval_ari']:.6f}`
        - Closest current clean GlaS endpoint (`fpn_2_only d128 @ 224 + autosam aug`): `fg_iou={glas_protocol['fg_iou']:.6f}` `dice={glas_protocol['dice']:.6f}`
        - Strict MoNuSeg anchor (`fpn_2_only d128 @ 512x512`): `fg_iou={monu_strict['fg_iou']:.6f}` `dice={monu_strict['dice']:.6f}`
        - Best current MoNuSeg run (`fpn_2 + fpn_1` residual refine): `fg_iou={monu_best['fg_iou']:.6f}` `dice={monu_best['dice']:.6f}` `eval_miou={monu_best['eval_miou']:.6f}` `eval_ari={monu_best['eval_ari']:.6f}`
        - AutoSAM reported paper references: `GlaS fg_iou={GLAS_AUTOSAM_REPORTED_FG_IOU:.6f}` `dice={GLAS_AUTOSAM_REPORTED_DICE:.6f}` | `MoNuSeg fg_iou={MONUSEG_AUTOSAM_REPORTED_FG_IOU:.6f}` `dice={MONUSEG_AUTOSAM_REPORTED_DICE:.6f}`

        ## Caveats

        - AutoSAM values are reported paper references, not reproduced runs from this repo.
        - The strongest MoNuSeg run on this page is native-resolution, so the strict `512x512` anchor remains the clean protocol reference.
        - The default-width coarse-only run is labeled explicitly as `fpn_2_only d64` on this page.
        """
    )


def main() -> None:
    require_file(AUTOSAM_COMPARISON_README)
    require_file(PHASE1_README)
    require_file(PHASE2_README)
    method_readme = require_file(METHOD_README)

    glas_runs = {spec.run_id: load_run(spec, GLAS_OUTPUT_ROOT) for spec in GLAS_RUN_SPECS}
    monuseg_runs = {spec.run_id: load_run(spec, MONUSEG_OUTPUT_ROOT) for spec in MONUSEG_RUN_SPECS}
    enforce_alignment(glas_runs, "cfc_baseline")
    enforce_alignment(monuseg_runs, "strict512_coarse")

    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    glas_story_blocks = build_glas_story_blocks(glas_runs)
    monuseg_story_blocks = build_monuseg_story_blocks(monuseg_runs)
    gallery_payload = build_gallery_payload(glas_runs, glas_story_blocks)
    results_rows = build_results_rows(glas_runs, monuseg_runs)
    plot_paths = write_plot_assets(glas_runs, monuseg_runs)
    table_csv_path = write_table_csv(results_rows)
    metrics_payload = build_metrics_payload(glas_runs, monuseg_runs, glas_story_blocks, monuseg_story_blocks, gallery_payload)
    training_data = build_training_data(results_rows, glas_story_blocks, monuseg_story_blocks, gallery_payload)
    links_payload = build_links_payload(table_csv_path, plot_paths)

    write_text(DEST_DIR / "index.html", render_index_html(glas_runs, monuseg_runs, results_rows, glas_story_blocks, monuseg_story_blocks, plot_paths, table_csv_path))
    write_text(DEST_DIR / "gallery.html", render_gallery_html())
    write_text(DEST_DIR / "metrics.json", json.dumps(metrics_payload, indent=2))
    write_text(DEST_DIR / "training_data.json", json.dumps(training_data, indent=2))
    write_text(DEST_DIR / "links.json", json.dumps(links_payload, indent=2))
    write_text(DEST_DIR / "summary.md", render_summary_md(glas_runs, monuseg_runs))
    write_text(DEST_DIR / "manifest.yaml", build_manifest(len(glas_story_blocks) + len(monuseg_story_blocks), gallery_payload))
    write_text(DEST_DIR / "method_source.md", method_readme.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
