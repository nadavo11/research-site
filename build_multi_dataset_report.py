#!/usr/bin/env python3
"""Build the SAM 3 cross-dataset overview from normalized CFC experiment roots."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import date
from html import escape
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
LEGACY_SOURCE_ROOT = ROOT / "experiments" / "Coarse Feature Clustering"
DEST_DIR = ROOT / "site" / "experiments" / "sam3-cross-dataset-overview"

DATASET_ORDER = ["rwtd", "caid", "stld", "cstd", "detexture"]
MULTI_TEXTURE_ORDER = ["detexture_multi"]
ALL_DATASET_ORDER = DATASET_ORDER + MULTI_TEXTURE_ORDER

FLAVOR_ORDER = ["coarse_only", "flip_averaged"]
MULTI_TEXTURE_FLAVOR_ORDER = ["oracle_k_full", "predicted_k_full"]
GALLERY_FLAVOR_ORDER = FLAVOR_ORDER + MULTI_TEXTURE_FLAVOR_ORDER

METRIC_CONTRACT = "architexture_binary_v1"
MULTI_TEXTURE_METRIC_CONTRACT = "detexture_multi_partition_v1"
METRIC_NOTE = (
    "The Coarse Feature Clustering flavor table is normalized onto the shared ArchiTexture binary evaluator "
    "(mIoU, ARI) so RWTD legacy aggregate-only summaries do not distort that comparison. "
    "The multi-texture table uses the DeTexture multi-partition contract instead."
)

DATASETS = {
    "rwtd": {
        "label": "RWTD",
        "title": "RWTD Dataset",
        "dataset_id": "aviadcohz/RWTD",
        "description": "Natural texture transitions and high-frequency boundary detection in organic scenes.",
    },
    "caid": {
        "label": "CAID",
        "title": "CAID Dataset",
        "dataset_id": "architexture:caid",
        "description": "Architectural plan partitions and region clustering in structured manifold domains.",
    },
    "stld": {
        "label": "STLD",
        "title": "STLD Dataset",
        "dataset_id": "architexture:stld",
        "description": "Structural and functional partitioning in diverse synthetic and real-world scenes.",
    },
    "cstd": {
        "label": "CSTD",
        "title": "CSTD Dataset",
        "dataset_id": "cstd",
        "description": "Binary region-versus-complement partitioning under the CSTD streaming benchmark.",
        "available_flavors": ["flip_averaged"],
        "run_dir_candidates": {
            "flip_averaged": ["cstd"],
        },
        "single_flavor_note": "Flip Averaged is published for CSTD, but there is no Coarse Only baseline yet.",
    },
    "detexture": {
        "label": "DeTexture",
        "title": "DeTexture ADE20K",
        "dataset_id": "detexture_ade20k",
        "description": "Binary texture-boundary crops from ADE20K evaluated as mask_a versus mask_b under the DeTexture benchmark.",
        "available_flavors": ["flip_averaged"],
        "single_flavor_note": "Flip Averaged is published for DeTexture ADE20K, but there is no Coarse Only baseline yet.",
    },
}

MULTI_TEXTURE_DATASETS = {
    "detexture_multi": {
        "label": "DeTexture Multi",
        "title": "DeTexture Multi-Texture",
        "dataset_id": "detexture_ade20k_multi",
        "description": "Multi-region DeTexture ADE20K crops evaluated under the order-invariant multi-partition contract.",
        "available_flavors": MULTI_TEXTURE_FLAVOR_ORDER,
        "comparison_flavors": MULTI_TEXTURE_FLAVOR_ORDER,
        "run_dir_candidates": {
            "oracle_k_full": ["detexture_multi/oracle_k_full"],
            "predicted_k_full": ["detexture_multi/predicted_k_full"],
        },
        "comparison_note": (
            "Oracle K supplies only the GT region count K, while Predicted K infers K automatically "
            "before the same pooled coarse clustering step."
        ),
    },
}

FLAVORS = {
    "coarse_only": {
        "label": "Coarse Only",
        "summary": "Average-pool the coarsest feature level once, then cluster and score the direct coarse partition.",
    },
    "flip_averaged": {
        "label": "Flip Averaged",
        "summary": "Average the coarsest features across identity, hflip, vflip, and hvflip before clustering.",
    },
    "oracle_k_full": {
        "label": "Oracle K",
        "summary": "Use the GT region count K while keeping clustering itself prompt-free and geometry-agnostic.",
    },
    "predicted_k_full": {
        "label": "Predicted K",
        "summary": "Predict K automatically, then run the same pooled coarse clustering without GT count access.",
    },
}

GALLERY_PAIR_QUOTAS = {
    "featured": 8,
    "delta_gain": 8,
    "delta_loss": 8,
    "leader_score": 8,
}

SINGLE_GALLERY_RECORD_QUOTAS = {
    "featured": 8,
    "score_desc": 8,
    "score_asc": 4,
    "coverage": 4,
}

CARD_PREVIEW_LIMIT = 2


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def to_public_record(record: dict) -> dict:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def parse_sample_numeric(sample_id: str) -> int:
    digits = "".join(ch for ch in sample_id if ch.isdigit())
    return int(digits) if digits else 0


def signed(value: float) -> str:
    return f"{value:+.3f}"


def human_join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def asset_relpath(dataset: str, flavor: str, file_name: str) -> Path:
    # Keep the historical coarse-only asset paths stable to avoid needless publish churn.
    if flavor == "coarse_only":
        return Path(dataset) / file_name
    return Path(flavor) / dataset / file_name


def dataset_catalog(dataset: str) -> dict:
    if dataset in DATASETS:
        return DATASETS[dataset]
    if dataset in MULTI_TEXTURE_DATASETS:
        return MULTI_TEXTURE_DATASETS[dataset]
    raise KeyError(f"Unknown dataset slug: {dataset}")


def pair_flavors_for_dataset(dataset: str) -> list[str]:
    return list(dataset_catalog(dataset).get("comparison_flavors", FLAVOR_ORDER))


def available_flavors_for_dataset(dataset: str) -> list[str]:
    info = dataset_catalog(dataset)
    return list(info.get("available_flavors", info.get("comparison_flavors", FLAVOR_ORDER)))


def resolve_run_dir(dataset: str, flavor: str) -> Path:
    info = dataset_catalog(dataset)
    override_candidates = info.get("run_dir_candidates", {}).get(flavor)
    if override_candidates:
        candidate_roots = [LEGACY_SOURCE_ROOT / relative_path for relative_path in override_candidates]
    else:
        dataset_root = LEGACY_SOURCE_ROOT / dataset
        if flavor == "coarse_only":
            candidate_roots = [dataset_root, dataset_root / "default"]
        else:
            candidate_roots = [LEGACY_SOURCE_ROOT / "flip_averaged" / dataset, dataset_root / "flip_averaged"]

    for candidate in candidate_roots:
        if (candidate / "summary.json").exists():
            return candidate

    for root in candidate_roots:
        if not root.exists() or not root.is_dir():
            continue
        for candidate in sorted(path for path in root.iterdir() if path.is_dir()):
            if (candidate / "summary.json").exists():
                return candidate

    raise FileNotFoundError(f"Could not resolve run dir for {dataset=} {flavor=}")



def leader_flavor_from_scores(
    left_flavor: str,
    left_miou: float,
    left_ari: float,
    right_flavor: str,
    right_miou: float,
    right_ari: float,
) -> str:
    if right_miou > left_miou:
        return right_flavor
    if right_miou < left_miou:
        return left_flavor
    if right_ari > left_ari:
        return right_flavor
    if right_ari < left_ari:
        return left_flavor
    return "tie"


def load_summary(run_dir: Path) -> dict:
    payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    metrics = payload["mean_metrics"]
    miou_value = metrics["eval_miou"] if "eval_miou" in metrics else metrics["miou"]
    ari_value = metrics["eval_ari"] if "eval_ari" in metrics else metrics["ari"]
    return {
        "mIoU": float(miou_value),
        "ARI": float(ari_value),
        "mIoU_agg": float(metrics.get("miou_agg", 0.0)),
        "variant": payload.get("variant", ""),
        "evaluation_contract": payload.get("evaluation_contract", "legacy_summary_metric"),
        "evaluation_view": payload.get("evaluation_view", ""),
        "sample_count": int(payload.get("num_evaluated_samples") or payload.get("num_total_samples") or payload.get("num_successes") or 0),
        "source_path": run_dir.relative_to(ROOT).as_posix(),
    }


def load_visual_records(dataset: str, flavor: str, run_dir: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    manifest_path = run_dir / "visuals_manifest.jsonl"
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            sample_id = str(raw.get("crop_name") or Path(raw["visual_path"]).stem)
            visual_name = Path(raw["visual_path"]).name
            source_file = run_dir / raw["visual_path"]
            asset_rel = asset_relpath(dataset, flavor, visual_name)
            record_id = f"{dataset}:{flavor}:{sample_id}"
            records[sample_id] = {
                "id": record_id,
                "pair_id": f"{dataset}:{sample_id}",
                "dataset": dataset,
                "flavor": flavor,
                "sample_id": sample_id,
                "sample_numeric": parse_sample_numeric(sample_id),
                "mIoU": round(float(raw.get("eval_miou", raw.get("miou", raw.get("sample_miou", 0.0)))), 4),
                "ARI": round(float(raw.get("eval_ari", raw.get("ari", raw.get("sample_ari", 0.0)))), 4),
                "file_path": asset_rel.as_posix(),
                "leader_flavor_for_sample": None,
                "pair_delta_mIoU": None,
                "pair_delta_ARI": None,
                "_source_path": source_file,
            }
    return records


def copy_asset(src: Path, dst: Path) -> None:
    ensure_parent(dst)
    shutil.copy2(src, dst)


def materialize_web_asset(record: dict) -> None:
    dst = DEST_DIR / "assets" / "all_previews" / record["file_path"]
    copy_asset(record["_source_path"], dst)


def extend_unique_pairs(selected: list[dict], seen: set[str], candidates: list[dict], limit: int) -> None:
    added = 0
    for pair in candidates:
        if pair["pair_id"] in seen:
            continue
        selected.append(pair)
        seen.add(pair["pair_id"])
        added += 1
        if added >= limit:
            break


def select_gallery_pairs(dataset_pairs: list[dict], featured_sample_ids: list[str]) -> list[dict]:
    pair_by_sample = {pair["sample_id"]: pair for pair in dataset_pairs}
    selected: list[dict] = []
    seen: set[str] = set()

    extend_unique_pairs(
        selected,
        seen,
        [pair_by_sample[sample_id] for sample_id in featured_sample_ids if sample_id in pair_by_sample],
        GALLERY_PAIR_QUOTAS["featured"],
    )
    extend_unique_pairs(
        selected,
        seen,
        sorted(
            dataset_pairs,
            key=lambda pair: (
                -pair["delta_mIoU"],
                -pair["leader_mIoU"],
                pair["sample_numeric"],
                pair["sample_id"],
            ),
        ),
        GALLERY_PAIR_QUOTAS["delta_gain"],
    )
    extend_unique_pairs(
        selected,
        seen,
        sorted(
            dataset_pairs,
            key=lambda pair: (
                pair["delta_mIoU"],
                pair["leader_mIoU"],
                pair["sample_numeric"],
                pair["sample_id"],
            ),
        ),
        GALLERY_PAIR_QUOTAS["delta_loss"],
    )
    extend_unique_pairs(
        selected,
        seen,
        sorted(
            dataset_pairs,
            key=lambda pair: (
                -pair["leader_mIoU"],
                -abs(pair["delta_mIoU"]),
                pair["sample_numeric"],
                pair["sample_id"],
            ),
        ),
        GALLERY_PAIR_QUOTAS["leader_score"],
    )
    return selected


def extend_unique_records(selected: list[dict], seen: set[str], candidates: list[dict], limit: int) -> None:
    added = 0
    for record in candidates:
        if record["id"] in seen:
            continue
        selected.append(record)
        seen.add(record["id"])
        added += 1
        if added >= limit:
            break


def evenly_spaced_records(records: list[dict], count: int) -> list[dict]:
    if count <= 0 or not records:
        return []
    if len(records) <= count:
        return list(records)
    if count == 1:
        return [records[len(records) // 2]]

    indices = {
        round(step * (len(records) - 1) / (count - 1))
        for step in range(count)
    }
    return [records[index] for index in sorted(indices)]


def select_single_flavor_records(records: list[dict], featured_sample_ids: list[str]) -> list[dict]:
    record_by_sample = {record["sample_id"]: record for record in records}
    ordered_by_sample = sorted(records, key=lambda record: (record["sample_numeric"], record["sample_id"]))
    ordered_by_score_desc = sorted(
        records,
        key=lambda record: (-record["mIoU"], -record["ARI"], record["sample_numeric"], record["sample_id"]),
    )
    ordered_by_score_asc = sorted(
        records,
        key=lambda record: (record["mIoU"], record["ARI"], record["sample_numeric"], record["sample_id"]),
    )

    selected: list[dict] = []
    seen: set[str] = set()
    extend_unique_records(
        selected,
        seen,
        [record_by_sample[sample_id] for sample_id in featured_sample_ids if sample_id in record_by_sample],
        SINGLE_GALLERY_RECORD_QUOTAS["featured"],
    )
    extend_unique_records(selected, seen, ordered_by_score_desc, SINGLE_GALLERY_RECORD_QUOTAS["score_desc"])
    extend_unique_records(selected, seen, ordered_by_score_asc, SINGLE_GALLERY_RECORD_QUOTAS["score_asc"])
    extend_unique_records(
        selected,
        seen,
        evenly_spaced_records(ordered_by_sample, SINGLE_GALLERY_RECORD_QUOTAS["coverage"]),
        SINGLE_GALLERY_RECORD_QUOTAS["coverage"],
    )
    return selected


def build_dataset_metrics(metrics_payload: dict, featured: dict[str, list[dict]]) -> str:
    summary = metrics_payload["summary"]
    single_flavor_datasets = summary["single_flavor_datasets"]
    all_dataset_labels = [metrics_payload["datasets"][dataset]["label"] for dataset in DATASET_ORDER]
    if single_flavor_datasets:
        single_flavor_blurbs = []
        for dataset in single_flavor_datasets:
            info = metrics_payload["datasets"][dataset]
            flavor_labels = ", ".join(FLAVORS[flavor]["label"] for flavor in info["available_flavors"])
            single_flavor_blurbs.append(f'{info["label"]} currently publishes {flavor_labels} only')
        coverage_copy = (
            f"Flavor-aware benchmarking for Coarse Feature Clustering across {human_join(all_dataset_labels)}. "
            + "; ".join(single_flavor_blurbs)
            + "."
        )
        single_label_copy = human_join(
            [metrics_payload["datasets"][dataset]["label"] for dataset in single_flavor_datasets]
        )
        singular = len(single_flavor_datasets) == 1
        executive_copy = (
            "Across the paired datasets, Flip Averaged leads RWTD and STLD, with the largest jump on STLD, "
            "while Coarse Only still leads CAID. "
            f"{single_label_copy} {'is' if singular else 'are'} included as "
            f"{'a ' if singular else ''}flip-only {'slice' if singular else 'slices'}, so "
            f"{'it expands' if singular else 'they expand'} coverage without affecting the paired deltas."
        )
    else:
        coverage_copy = (
            f"Flavor-aware benchmarking for Coarse Feature Clustering across {human_join(all_dataset_labels)}, now including the "
            "new <code>flip_averaged</code> variant."
        )
        executive_copy = (
            "The new <strong>Flip Averaged</strong> flavor changes the cross-dataset picture materially. It takes the "
            "lead on <strong>RWTD</strong> and <strong>STLD</strong>, with the largest jump on STLD, while the "
            "original <strong>Coarse Only</strong> flavor still leads CAID."
        )

    def preview_figures(dataset: str, info: dict) -> str:
        previews = []
        for record in featured[dataset][:CARD_PREVIEW_LIMIT]:
            src = f"assets/all_previews/{record['file_path']}"
            previews.append(
                f'''
                <figure>
                  <img src="{escape(src)}" alt="{escape(info["label"])} sample {escape(record["sample_id"])}"
                    data-zoom-src="{escape(src)}" />
                  <figcaption>
                    <span class="pill">id {escape(record["sample_id"])}</span>
                    <span class="pill">mIoU {record["mIoU"]:.3f}</span>
                  </figcaption>
                </figure>
                '''
            )
        return ''.join(previews)

    rows = []
    cards = []
    for dataset in DATASET_ORDER:
        info = metrics_payload["datasets"][dataset]
        coarse = info["flavors"].get("coarse_only")
        flip = info["flavors"].get("flip_averaged")
        leader = FLAVORS[info["leader_flavor"]]["label"]
        delta_miou = info["delta_mIoU"]
        delta_ari = info["delta_ARI"]
        delta_class = "good" if (delta_miou or 0) > 0 else "bad" if (delta_miou or 0) < 0 else ""
        coarse_cell = (
            f'''
                <strong>{coarse["mIoU"]:.3f}</strong>
                <div class="mini-note">ARI {coarse["ARI"]:.3f}</div>
            '''
            if coarse
            else '''
                <strong>-</strong>
                <div class="mini-note">not published</div>
            '''
        )
        flip_cell = (
            f'''
                <strong>{flip["mIoU"]:.3f}</strong>
                <div class="mini-note">ARI {flip["ARI"]:.3f}</div>
            '''
            if flip
            else '''
                <strong>-</strong>
                <div class="mini-note">not published</div>
            '''
        )
        if delta_miou is None or delta_ari is None:
            delta_miou_cell = '<span class="delta-pill">n/a</span>'
            delta_ari_cell = '<span class="delta-pill">n/a</span>'
        else:
            delta_miou_cell = f'<span class="delta-pill {delta_class}">{signed(delta_miou)}</span>'
            delta_ari_cell = f'<span class="delta-pill {delta_class}">{signed(delta_ari)}</span>'
        leader_note = '<div class="mini-note">single-flavor only</div>' if not info["comparable"] else ''
        rows.append(
            f'''
            <tr>
              <td>
                <strong>{escape(info["label"])}</strong>
                <div class="mini-note">{escape(info["dataset_id"])}</div>
              </td>
              <td>{coarse_cell}</td>
              <td>{flip_cell}</td>
              <td>{delta_miou_cell}</td>
              <td>{delta_ari_cell}</td>
              <td><span class="tag {'good' if info["leader_flavor"] == 'flip_averaged' else 'warn'}">{escape(leader)}</span>{leader_note}</td>
            </tr>
            '''
        )

        leader_tag = "good" if info["leader_flavor"] == "flip_averaged" else "warn"
        if info["comparable"] and delta_miou is not None and delta_ari is not None and delta_miou >= 0:
            summary_copy = f"Flip Averaged leads by {signed(delta_miou)} mIoU and {signed(delta_ari)} ARI."
        elif info["comparable"] and delta_miou is not None and delta_ari is not None:
            summary_copy = f"Coarse Only holds the lead by {abs(delta_miou):.3f} mIoU and {abs(delta_ari):.3f} ARI."
        else:
            summary_copy = info["single_flavor_note"]
        if info["comparable"]:
            win_counts = info["sample_wins"]
            stats_html = f'''
              <div class="leader-stats">
                <span class="pill">Leader mIoU {info["leader_mIoU"]:.3f}</span>
                <span class="pill">Leader ARI {info["leader_ARI"]:.3f}</span>
                <span class="pill">Wins F/C/T {win_counts["flip_averaged"]}/{win_counts["coarse_only"]}/{win_counts["ties"]}</span>
              </div>
            '''
            availability_tag = ''
            action_href = f"gallery.html?dataset={dataset}&view=paired"
            action_label = f"Open {escape(info['label'])} 1:1 Gallery"
        else:
            published_flavors = ", ".join(FLAVORS[flavor]["label"] for flavor in info["available_flavors"])
            stats_html = f'''
              <div class="leader-stats">
                <span class="pill">mIoU {info["leader_mIoU"]:.3f}</span>
                <span class="pill">ARI {info["leader_ARI"]:.3f}</span>
                <span class="pill">Samples {info["sample_count"]:,}</span>
              </div>
            '''
            availability_tag = f'<span class="tag">Published: {escape(published_flavors)} only</span>'
            action_href = f"gallery.html?dataset={dataset}&view={info['available_flavors'][0]}"
            action_label = f"Open {escape(info['label'])} {escape(FLAVORS[info['available_flavors'][0]]['label'])} Gallery"
        cards.append(
            f'''
            <article class="dataset-card cfc-card">
              <div class="card-topline">
                <span class="tag">{escape(info["label"])}</span>
                <span class="tag {leader_tag}">Leader: {escape(leader)}</span>
                {availability_tag}
              </div>
              <h3>{escape(info["title"])}</h3>
              <p class="kv card-summary">{escape(summary_copy)}</p>
              {stats_html}
              <div class="preview-grid">
                {preview_figures(dataset, info)}
              </div>
              <div class="card-actions">
                <a class="btn secondary" href="{action_href}">{action_label}</a>
              </div>
            </article>
            '''
        )

    multi_rows = []
    multi_cards = []
    multi_summary_bits = []
    for dataset in MULTI_TEXTURE_ORDER:
        info = metrics_payload["datasets"][dataset]
        if not info["comparable"]:
            continue
        left_flavor, right_flavor = info["pair_flavors"]
        left = info["flavors"][left_flavor]
        right = info["flavors"][right_flavor]
        leader_label = FLAVORS[info["leader_flavor"]]["label"]
        delta_miou = info["delta_mIoU"]
        delta_ari = info["delta_ARI"]
        delta_class = "good" if (delta_miou or 0) > 0 else "bad" if (delta_miou or 0) < 0 else ""
        multi_rows.append(
            f'''
            <tr>
              <td>
                <strong>{escape(info["label"])}</strong>
                <div class="mini-note">{escape(info["dataset_id"])}</div>
              </td>
              <td>
                <strong>{left["mIoU"]:.3f}</strong>
                <div class="mini-note">ARI {left["ARI"]:.3f}</div>
              </td>
              <td>
                <strong>{right["mIoU"]:.3f}</strong>
                <div class="mini-note">ARI {right["ARI"]:.3f}</div>
              </td>
              <td><span class="delta-pill {delta_class}">{signed(delta_miou)}</span></td>
              <td><span class="delta-pill {delta_class}">{signed(delta_ari)}</span></td>
              <td>
                <span class="tag good">{escape(leader_label)}</span>
                <div class="mini-note">{info["sample_count"]:,} paired samples</div>
              </td>
            </tr>
            '''
        )
        if delta_miou is not None and delta_ari is not None and delta_miou >= 0:
            summary_copy = f"{FLAVORS[right_flavor]['label']} leads by {signed(delta_miou)} mIoU and {signed(delta_ari)} ARI."
            multi_summary_bits.append(
                f"{info['label']} favors <strong>{escape(FLAVORS[right_flavor]['label'])}</strong> "
                f"({right['mIoU']:.3f} mIoU, {right['ARI']:.3f} ARI) over {escape(FLAVORS[left_flavor]['label'])}."
            )
        else:
            summary_copy = f"{FLAVORS[left_flavor]['label']} remains ahead by {abs(delta_miou):.3f} mIoU and {abs(delta_ari):.3f} ARI."
            multi_summary_bits.append(
                f"{info['label']} favors <strong>{escape(FLAVORS[left_flavor]['label'])}</strong> "
                f"({left['mIoU']:.3f} mIoU, {left['ARI']:.3f} ARI) over {escape(FLAVORS[right_flavor]['label'])} "
                f"({right['mIoU']:.3f} mIoU, {right['ARI']:.3f} ARI)."
            )
        win_counts = info["sample_wins"]
        stats_html = f'''
          <div class="leader-stats">
            <span class="pill">Leader mIoU {info["leader_mIoU"]:.3f}</span>
            <span class="pill">Leader ARI {info["leader_ARI"]:.3f}</span>
            <span class="pill">{escape(FLAVORS[left_flavor]['label'])} wins {win_counts[left_flavor]}</span>
            <span class="pill">{escape(FLAVORS[right_flavor]['label'])} wins {win_counts[right_flavor]}</span>
            <span class="pill">Ties {win_counts['ties']}</span>
          </div>
        '''
        multi_cards.append(
            f'''
            <article class="dataset-card cfc-card">
              <div class="card-topline">
                <span class="tag">{escape(info["label"])}</span>
                <span class="tag good">Leader: {escape(leader_label)}</span>
                <span class="tag">Paired: {escape(FLAVORS[left_flavor]['label'])} vs {escape(FLAVORS[right_flavor]['label'])}</span>
              </div>
              <h3>{escape(info["title"])}</h3>
              <p class="kv card-summary">{escape(summary_copy)}</p>
              {stats_html}
              <div class="preview-grid">
                {preview_figures(dataset, info)}
              </div>
              <div class="card-actions">
                <a class="btn secondary" href="gallery.html?dataset={dataset}&view=paired">Open {escape(info['label'])} 1:1 Gallery</a>
              </div>
            </article>
            '''
        )

    if multi_summary_bits:
        coverage_copy += " A separate multi-texture DeTexture slice compares Oracle K against Predicted K."
        executive_copy += " Separately, " + " ".join(multi_summary_bits)

    multi_table_section = ''
    if multi_rows:
        multi_table_section = dedent(
            f'''            <section class="section comparison-table">
              <h2>Multi-Texture Segmentation</h2>
              <p class="kv" style="margin-bottom: 18px;">
                Rows use the <code>{MULTI_TEXTURE_METRIC_CONTRACT}</code> evaluator (<code>mIoU</code>, <code>ARI</code>).
                Deltas are <code>{MULTI_TEXTURE_FLAVOR_ORDER[1]} - {MULTI_TEXTURE_FLAVOR_ORDER[0]}</code>.
              </p>
              <div class="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Dataset</th>
                      <th>Oracle K</th>
                      <th>Predicted K</th>
                      <th>Delta mIoU</th>
                      <th>Delta ARI</th>
                      <th>Leader</th>
                    </tr>
                  </thead>
                  <tbody>
                    {''.join(multi_rows)}
                  </tbody>
                </table>
              </div>
            </section>
            '''
        )

    multi_card_section = ''
    if multi_cards:
        multi_card_section = dedent(
            '''            <section class="section">
              <div style="display: flex; justify-content: space-between; gap: 20px; align-items: end; flex-wrap: wrap;">
                <div>
                  <h2>Multi-Texture Leader</h2>
                  <p class="kv">Same-sample Oracle-K vs Predicted-K comparisons on DeTexture multi-region crops.</p>
                </div>
                <a class="btn secondary" href="gallery.html?dataset=detexture_multi&view=paired">Open Multi-Texture Gallery</a>
              </div>
              <div class="dataset-grid" style="margin-top: 22px;">
                __MULTI_TEXTURE_CARDS__
              </div>
            </section>
            '''
        ).replace('__MULTI_TEXTURE_CARDS__', ''.join(multi_cards))

    page = dedent(
        '''        <!DOCTYPE html>
        <html lang="en">

        <head>
          <meta charset="UTF-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1.0" />
          <title>SAM 3 Cross-Dataset Benchmarking</title>
          <link rel="stylesheet" href="../../assets/site.css" />
          <style>
            .insight-note {
              margin-top: 18px;
              color: var(--muted);
              max-width: 82ch;
            }

            .compact-grid {
              grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            }

            .comparison-table table td,
            .comparison-table table th {
              vertical-align: top;
            }

            .mini-note {
              font-size: 0.78rem;
              color: var(--muted);
              margin-top: 6px;
            }

            .delta-pill {
              display: inline-flex;
              align-items: center;
              border-radius: 999px;
              padding: 6px 12px;
              font-size: 0.82rem;
              font-weight: 700;
              border: 1px solid var(--line);
              background: #fff;
            }

            .delta-pill.good,
            .delta-positive {
              color: var(--good);
            }

            .delta-pill.bad,
            .delta-negative {
              color: var(--bad);
            }

            .cfc-card {
              display: flex;
              flex-direction: column;
              gap: 14px;
            }

            .card-topline {
              display: flex;
              flex-wrap: wrap;
              gap: 10px;
            }

            .card-summary {
              margin: 0;
            }

            .leader-stats {
              display: flex;
              flex-wrap: wrap;
              gap: 10px;
            }

            .preview-grid {
              display: grid;
              grid-template-columns: repeat(2, minmax(0, 1fr));
              gap: 14px;
            }

            .preview-grid figure img {
              aspect-ratio: 1;
            }

            .card-actions {
              margin-top: auto;
            }

            @media (max-width: 720px) {
              .preview-grid {
                grid-template-columns: 1fr;
              }
            }
          </style>
        </head>

        <body>
          <div class="container animate-fade-in-down">
            <header style="text-align: center; margin-bottom: 60px;">
              <div class="subtitle">Unified Analysis</div>
              <h1 class="title-gradient">SAM 3 Cross-Dataset Benchmarking</h1>
              <p style="color: var(--muted); margin-top: 16px; font-weight: 300; max-width: 860px; margin-inline: auto;">
                __COVERAGE_COPY__
              </p>
            </header>

            <section class="executive-summary">
              <h2>Executive Synopsis</h2>
              <p>__EXECUTIVE_COPY__</p>
              <p class="insight-note">__METRIC_NOTE__</p>
            </section>

            <section class="section">
              <h2>Headline Readout</h2>
              <div class="metric-grid">
                __HEADLINE_CARDS__
              </div>
            </section>

            <section class="section comparison-table">
              <h2>Flavor Comparison</h2>
              <p class="kv" style="margin-bottom: 18px;">
                All rows use the shared <code>mIoU</code>/<code>ARI</code> contract. Deltas are
                <code>flip_averaged - coarse_only</code>; rows without a baseline are marked <code>n/a</code>.
              </p>
              <div class="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Dataset</th>
                      <th>Coarse Only</th>
                      <th>Flip Averaged</th>
                      <th>Delta mIoU</th>
                      <th>Delta ARI</th>
                      <th>Leader</th>
                    </tr>
                  </thead>
                  <tbody>
                    __TABLE_ROWS__
                  </tbody>
                </table>
              </div>
            </section>

            __MULTI_TEXTURE_TABLE_SECTION__

            <section class="section">
              <div style="display: flex; justify-content: space-between; gap: 20px; align-items: end; flex-wrap: wrap;">
                <div>
                  <h2>Dataset Leaders</h2>
                  <p class="kv">Fast read: who leads, by how much, and representative samples.</p>
                </div>
                <a class="btn" href="gallery.html">Open Gallery</a>
              </div>
              <div class="dataset-grid" style="margin-top: 22px;">
                __DATASET_CARDS__
              </div>
            </section>

            __MULTI_TEXTURE_CARD_SECTION__

            <footer style="margin-top: 80px; text-align: center;">
              <p><a href="../../index.html" style="text-decoration: none; color: var(--brand); font-weight: 600;">Back to Dashboard</a></p>
              <p style="margin-top: 12px;">Research Analytics by <strong>Antigravity</strong></p>
            </footer>
          </div>

          <dialog class="image-dialog" id="imageDialog">
            <img id="imageDialogImg" alt="Expanded preview" />
          </dialog>
          <script src="../../assets/site.js"></script>
        </body>

        </html>
        '''
    )
    return (
        page.replace("__METRIC_NOTE__", escape(METRIC_NOTE))
        .replace("__COVERAGE_COPY__", coverage_copy)
        .replace("__EXECUTIVE_COPY__", executive_copy)
        .replace("__HEADLINE_CARDS__", build_headline_cards(metrics_payload))
        .replace("__TABLE_ROWS__", ''.join(rows))
        .replace("__MULTI_TEXTURE_TABLE_SECTION__", multi_table_section)
        .replace("__DATASET_CARDS__", ''.join(cards))
        .replace("__MULTI_TEXTURE_CARD_SECTION__", multi_card_section)
    )


def build_headline_cards(metrics_payload: dict) -> str:
    summary = metrics_payload["summary"]
    paired_dataset_count = summary["comparable_dataset_count"]
    single_flavor_count = summary["single_flavor_count"]
    leader_label = FLAVORS[summary["datasets_led_by"]]["label"]
    best_gain_dataset_miou = metrics_payload["datasets"][summary["best_gain_dataset_mIoU"]]["label"]
    best_gain_dataset_ari = metrics_payload["datasets"][summary["best_gain_dataset_ARI"]]["label"]
    paired_note = f"{paired_dataset_count} paired dataset{'s' if paired_dataset_count != 1 else ''}"
    if single_flavor_count:
        paired_note += f" + {single_flavor_count} single-flavor"
    return (
        f"""
        <article class="metric">
          <div class="k">Paired Samples</div>
          <div class="v">{summary["paired_samples"]:,}</div>
          <div class="mini-note">{paired_note}</div>
        </article>
        <article class="metric">
          <div class="k">Flavor Lead Split</div>
          <div class="v">{summary["datasets_led"]['flip_averaged']}/{paired_dataset_count}</div>
          <div class="mini-note">{escape(leader_label)} leads more paired datasets.</div>
        </article>
        <article class="metric">
          <div class="k">Best mIoU Gain</div>
          <div class="v delta-positive">{signed(summary["best_gain_mIoU"])}</div>
          <div class="mini-note">{escape(best_gain_dataset_miou)} vs Coarse Only</div>
        </article>
        <article class="metric">
          <div class="k">Best ARI Gain</div>
          <div class="v delta-positive">{signed(summary["best_gain_ARI"])}</div>
          <div class="mini-note">{escape(best_gain_dataset_ari)} vs Coarse Only</div>
        </article>
        """
    )


def build_gallery_html() -> str:
    dataset_options = [
        '<option value="all">All Datasets</option>',
        *[
            f'<option value="{dataset}">{escape(dataset_catalog(dataset)["label"])} ({dataset_catalog(dataset)["dataset_id"]})</option>'
            for dataset in ALL_DATASET_ORDER
        ],
    ]
    flavor_options = [
        '<option value="paired">1:1 Comparison</option>',
        *[
            f'<option value="{flavor}">{escape(FLAVORS[flavor]["label"])}</option>'
            for flavor in GALLERY_FLAVOR_ORDER
        ],
    ]
    gallery = dedent(
        '''        <!DOCTYPE html>
        <html lang="en">

        <head>
          <meta charset="UTF-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1.0" />
          <title>Comparison Gallery - SAM 3</title>
          <link rel="stylesheet" href="../../assets/site.css" />
          <style>
            .gallery-toolbar {
              display: flex;
              flex-wrap: wrap;
              gap: 14px;
              align-items: center;
              background: var(--glass-bg);
              backdrop-filter: blur(12px);
              -webkit-backdrop-filter: blur(12px);
              border: 1px solid var(--glass-border);
              border-radius: 20px;
              padding: 18px 20px;
              box-shadow: var(--premium-shadow);
              margin-bottom: 26px;
            }

            .gallery-toolbar label {
              font-family: var(--font-accent);
              font-size: 0.9rem;
              font-weight: 600;
            }

            .gallery-toolbar select {
              font: inherit;
              border: 1px solid var(--line);
              border-radius: 999px;
              padding: 9px 14px;
              background: #fff;
            }

            .toolbar-count {
              margin-left: auto;
              color: var(--muted);
              font-size: 0.9rem;
            }

            .toolbar-note {
              width: 100%;
              color: var(--muted);
              font-size: 0.88rem;
              margin-top: 2px;
            }

            .gallery-controls {
              display: flex;
              justify-content: center;
              margin-top: 24px;
            }

            .gallery-controls[hidden] {
              display: none;
            }

            .gallery-controls .btn {
              min-width: 180px;
            }

            .gallery-stack {
              display: grid;
              gap: 20px;
            }

            .single-grid {
              display: grid;
              grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
              gap: 20px;
            }

            .single-card,
            .pair-card {
              border: 1px solid var(--line);
              border-radius: 20px;
              background: rgba(255, 255, 255, 0.82);
              overflow: hidden;
              box-shadow: 0 8px 18px rgba(18, 32, 51, 0.05);
            }

            .pair-card {
              padding: 18px;
            }

            .pair-head,
            .single-head {
              display: flex;
              flex-wrap: wrap;
              gap: 8px;
              align-items: center;
              padding: 16px 18px 0;
            }

            .pair-head {
              padding: 0 0 14px;
            }

            .pair-grid {
              display: grid;
              grid-template-columns: repeat(2, minmax(0, 1fr));
              gap: 16px;
            }

            .pair-grid figure,
            .single-card figure {
              margin: 0;
              border-radius: 16px;
              overflow: hidden;
              background: #fff;
              border: 1px solid var(--line);
            }

            .single-card figure img,
            .pair-grid figure img {
              width: 100%;
              display: block;
              aspect-ratio: 1;
              object-fit: cover;
              cursor: zoom-in;
            }

            .panel-label {
              padding: 10px 12px 0;
              font-size: 0.8rem;
              text-transform: uppercase;
              letter-spacing: 0.08em;
              color: var(--muted);
              font-weight: 700;
            }

            .metric-bar {
              padding: 12px;
              border-top: 1px solid var(--line);
              display: flex;
              flex-wrap: wrap;
              gap: 8px;
              background: #fbfdff;
            }

            .delta-positive {
              color: var(--good);
            }

            .delta-negative {
              color: var(--bad);
            }

            .pill.dataset-pill {
              font-weight: 700;
            }

            .empty-state {
              padding: 28px;
              border: 1px dashed var(--line);
              border-radius: 20px;
              background: rgba(255, 255, 255, 0.82);
              color: var(--muted);
              text-align: center;
            }

            @media (max-width: 900px) {
              .pair-grid {
                grid-template-columns: 1fr;
              }
            }
          </style>
        </head>

        <body>
          <div class="container animate-fade-in-down">
            <header style="text-align: center; margin-bottom: 42px;">
              <div class="subtitle">Interactive Gallery</div>
              <h1 class="title-gradient">Comparison Gallery</h1>
              <p style="color: var(--muted); margin-top: 16px; font-weight: 300; max-width: 860px; margin-inline: auto;">
                Browse paired same-sample comparisons for both Coarse Feature Clustering and the new multi-texture
                Oracle-K versus Predicted-K benchmark, or switch into single-flavor slices for datasets that
                currently publish only one flavor. The overview page metrics still reflect the full benchmark behind
                each section.
              </p>
            </header>

            <section class="section">
              <div class="gallery-toolbar">
                <label for="datasetSelect">Dataset</label>
                <select id="datasetSelect">__DATASET_OPTIONS__</select>

                <label for="flavorSelect">Flavor</label>
                <select id="flavorSelect">__FLAVOR_OPTIONS__</select>

                <label for="sortSelect">Sort</label>
                <select id="sortSelect">
                  <option value="delta_desc">Delta mIoU down</option>
                  <option value="score_desc">Score down</option>
                  <option value="score_asc">Score up</option>
                  <option value="sample_id_asc">Sample ID up</option>
                </select>

                <label for="pageSizeSelect">Per Page</label>
                <select id="pageSizeSelect">
                  <option value="8">8</option>
                  <option value="12">12</option>
                  <option value="16" selected>16</option>
                  <option value="24">24</option>
                  <option value="48">48</option>
                  <option value="all">All</option>
                </select>

                <div class="toolbar-count" id="countLabel">Loading...</div>
                <div class="toolbar-note" id="toolbarNote"></div>
              </div>

              <div id="galleryRoot" class="gallery-stack"></div>
              <div class="gallery-controls" id="galleryControls" hidden>
                <button class="btn secondary" id="loadMoreBtn" type="button">Load More</button>
              </div>
            </section>

            <footer style="margin-top: 60px; text-align: center;">
              <p><a href="index.html" style="text-decoration: none; color: var(--brand); font-weight: 600;">Back to report</a></p>
            </footer>
          </div>

          <dialog class="image-dialog" id="imageDialog">
            <img id="imageDialogImg" alt="Expanded preview" />
          </dialog>

          <script>
            (async () => {
              const data = await fetch('training_data.json').then((response) => response.json());
              const recordMap = new Map(data.per_image.map((record) => [record.id, record]));
              const root = document.getElementById('galleryRoot');
              const datasetSelect = document.getElementById('datasetSelect');
              const flavorSelect = document.getElementById('flavorSelect');
              const sortSelect = document.getElementById('sortSelect');
              const pageSizeSelect = document.getElementById('pageSizeSelect');
              const countLabel = document.getElementById('countLabel');
              const toolbarNote = document.getElementById('toolbarNote');
              const galleryControls = document.getElementById('galleryControls');
              const loadMoreBtn = document.getElementById('loadMoreBtn');
              const dialog = document.getElementById('imageDialog');
              const dialogImg = document.getElementById('imageDialogImg');
              const params = new URLSearchParams(window.location.search);
              let loadedPages = 1;

              const initialDataset = params.get('dataset');
              if (initialDataset && [...datasetSelect.options].some((option) => option.value === initialDataset)) {
                datasetSelect.value = initialDataset;
              } else {
                datasetSelect.value = 'all';
              }

              const initialView = params.get('view');
              if (initialView && [...flavorSelect.options].some((option) => option.value === initialView)) {
                flavorSelect.value = initialView;
              } else {
                flavorSelect.value = 'paired';
              }

              const initialPageSize = params.get('page_size');
              if (initialPageSize && [...pageSizeSelect.options].some((option) => option.value === initialPageSize)) {
                pageSizeSelect.value = initialPageSize;
              } else {
                pageSizeSelect.value = '16';
              }

              function currentPageSize() {
                return pageSizeSelect.value === 'all' ? Number.POSITIVE_INFINITY : Number(pageSizeSelect.value);
              }

              function resetPaging() {
                loadedPages = 1;
              }

              function pairFlavorLabels(datasetInfo) {
                if (!datasetInfo || !datasetInfo.pair_flavors || !datasetInfo.pair_flavors.length) return [];
                return datasetInfo.pair_flavors.map((flavor) => data.flavors[flavor]?.label || flavor);
              }

              function syncFlavorAvailability() {
                const dataset = datasetSelect.value;
                const datasetInfo = dataset === 'all' ? null : data.datasets[dataset];
                const availableFlavors = datasetInfo ? new Set(datasetInfo.available_flavors) : new Set(Object.keys(data.flavors));
                const pairedEnabled = datasetInfo ? datasetInfo.has_paired_comparison : data.counts.paired_samples > 0;

                [...flavorSelect.options].forEach((option) => {
                  if (option.value === 'paired') {
                    option.disabled = !pairedEnabled;
                    return;
                  }
                  option.disabled = dataset !== 'all' && !availableFlavors.has(option.value);
                });

                if (flavorSelect.selectedOptions[0]?.disabled) {
                  if (pairedEnabled) {
                    flavorSelect.value = 'paired';
                  } else {
                    const fallback = [...flavorSelect.options].find((option) => !option.disabled);
                    if (fallback) flavorSelect.value = fallback.value;
                  }
                }
              }

              const initialPages = Number(params.get('pages'));
              const initialShown = Number(params.get('shown'));
              if (Number.isFinite(initialPages) && initialPages > 0) {
                loadedPages = Math.max(1, Math.floor(initialPages));
              } else if (Number.isFinite(initialShown) && initialShown > 0 && Number.isFinite(currentPageSize())) {
                loadedPages = Math.max(1, Math.ceil(initialShown / currentPageSize()));
              } else {
                resetPaging();
              }

              function syncUrl() {
                const next = new URLSearchParams(window.location.search);
                next.set('dataset', datasetSelect.value);
                next.set('view', flavorSelect.value);
                next.set('sort', sortSelect.value);
                next.set('page_size', pageSizeSelect.value);
                if (Number.isFinite(currentPageSize()) && loadedPages > 1) {
                  next.set('pages', String(loadedPages));
                } else {
                  next.delete('pages');
                }
                next.delete('shown');
                history.replaceState({}, '', `${window.location.pathname}?${next.toString()}`);
              }

              function deltaClass(value) {
                if (value > 0) return 'delta-positive';
                if (value < 0) return 'delta-negative';
                return '';
              }

              function sortItems(items, mode) {
                const sorted = items.slice();
                sorted.sort((left, right) => {
                  if (mode === 'sample_id_asc') {
                    return (left.sample_numeric || 0) - (right.sample_numeric || 0) || String(left.sample_id).localeCompare(String(right.sample_id));
                  }
                  if (mode === 'score_asc') {
                    const leftScore = left.leader_mIoU ?? left.mIoU ?? 0;
                    const rightScore = right.leader_mIoU ?? right.mIoU ?? 0;
                    return leftScore - rightScore;
                  }
                  if (mode === 'score_desc') {
                    const leftScore = left.leader_mIoU ?? left.mIoU ?? 0;
                    const rightScore = right.leader_mIoU ?? right.mIoU ?? 0;
                    return rightScore - leftScore;
                  }
                  const leftDelta = left.delta_mIoU ?? left.pair_delta_mIoU ?? 0;
                  const rightDelta = right.delta_mIoU ?? right.pair_delta_mIoU ?? 0;
                  return rightDelta - leftDelta;
                });
                return sorted;
              }

              const initialSort = params.get('sort');
              if (initialSort && [...sortSelect.options].some((option) => option.value === initialSort)) {
                sortSelect.value = initialSort;
              }

              function visibleCountFor(itemsLength) {
                if (!Number.isFinite(currentPageSize())) return itemsLength;
                return Math.min(itemsLength, currentPageSize() * loadedPages);
              }

              function visibleSlice(items) {
                return items.slice(0, visibleCountFor(items.length));
              }

              function updateLoadMore(items, noun) {
                const pageSize = currentPageSize();
                const shown = visibleCountFor(items.length);
                if (!Number.isFinite(pageSize) || shown >= items.length) {
                  galleryControls.hidden = true;
                  return;
                }

                galleryControls.hidden = false;
                const remaining = items.length - shown;
                const nextCount = Math.min(pageSize, remaining);
                loadMoreBtn.textContent = `Load ${nextCount} More ${noun}`;
              }

              function renderEmptyState(message) {
                root.className = 'gallery-stack';
                root.innerHTML = `<article class="empty-state">${message}</article>`;
                galleryControls.hidden = true;
              }

              function renderPairCard(pair) {
                const left = recordMap.get(pair.record_ids[pair.left_flavor]);
                const right = recordMap.get(pair.record_ids[pair.right_flavor]);
                const comparisonLabel = `${data.flavors[pair.left_flavor].label} vs ${data.flavors[pair.right_flavor].label}`;
                return `
                  <article class="pair-card">
                    <div class="pair-head">
                      <span class="pill dataset-pill">${data.datasets[pair.dataset].label}</span>
                      <span class="pill">${comparisonLabel}</span>
                      <span class="pill">id ${pair.sample_id}</span>
                      <span class="pill ${pair.leader_flavor === 'tie' ? '' : 'good'}">
                        Leader: ${pair.leader_flavor === 'tie' ? 'Tie' : data.flavors[pair.leader_flavor].label}
                      </span>
                      <span class="pill ${deltaClass(pair.delta_mIoU)}">Delta mIoU ${pair.delta_mIoU.toFixed(3)}</span>
                      <span class="pill ${deltaClass(pair.delta_ARI)}">Delta ARI ${pair.delta_ARI.toFixed(3)}</span>
                    </div>
                    <div class="pair-grid">
                      ${renderPanel(left)}
                      ${renderPanel(right)}
                    </div>
                  </article>
                `;
              }

              function renderPanel(record) {
                return `
                  <figure>
                    <div class="panel-label">${data.flavors[record.flavor].label}</div>
                    <img src="assets/all_previews/${record.file_path}" alt="${data.datasets[record.dataset].label} sample ${record.sample_id}"
                      data-zoom-src="assets/all_previews/${record.file_path}" loading="lazy" />
                    <div class="metric-bar">
                      <span class="pill">mIoU ${record.mIoU.toFixed(3)}</span>
                      <span class="pill">ARI ${record.ARI.toFixed(3)}</span>
                    </div>
                  </figure>
                `;
              }

              function renderSingleCard(record) {
                const leaderLabel = record.leader_flavor_for_sample
                  ? (record.leader_flavor_for_sample === 'tie' ? 'Tie' : data.flavors[record.leader_flavor_for_sample].label)
                  : null;
                const deltaValue = typeof record.pair_delta_mIoU === 'number'
                  ? `<span class="pill ${deltaClass(record.pair_delta_mIoU)}">Delta mIoU ${record.pair_delta_mIoU.toFixed(3)}</span>`
                  : '';
                return `
                  <article class="single-card">
                    <div class="single-head">
                      <span class="pill dataset-pill">${data.datasets[record.dataset].label}</span>
                      <span class="pill">${data.flavors[record.flavor].label}</span>
                      <span class="pill">id ${record.sample_id}</span>
                      ${deltaValue}
                    </div>
                    <figure>
                      <img src="assets/all_previews/${record.file_path}" alt="${data.datasets[record.dataset].label} sample ${record.sample_id}"
                        data-zoom-src="assets/all_previews/${record.file_path}" loading="lazy" />
                      <div class="metric-bar">
                        <span class="pill">mIoU ${record.mIoU.toFixed(3)}</span>
                        <span class="pill">ARI ${record.ARI.toFixed(3)}</span>
                        ${leaderLabel ? `<span class="pill">Leader ${leaderLabel}</span>` : ''}
                      </div>
                    </figure>
                  </article>
                `;
              }

              function render() {
                const dataset = datasetSelect.value;
                const datasetInfo = dataset === 'all' ? null : data.datasets[dataset];
                syncFlavorAvailability();
                const view = flavorSelect.value;
                const sort = sortSelect.value;

                if (view === 'paired') {
                  const filteredPairs = data.pairs.filter((pair) => dataset === 'all' || pair.dataset === dataset);
                  const items = sortItems(filteredPairs, sort);
                  const visibleItems = visibleSlice(items);
                  if (visibleItems.length) {
                    root.className = 'gallery-stack';
                    root.innerHTML = visibleItems.map(renderPairCard).join('');
                  } else if (datasetInfo && !datasetInfo.has_paired_comparison) {
                    const fallbackFlavor = datasetInfo.available_flavors[0];
                    const fallbackLabel = fallbackFlavor ? data.flavors[fallbackFlavor].label : 'the published flavor';
                    renderEmptyState(`${datasetInfo.label} currently publishes single-flavor records only. Switch the Flavor control to ${fallbackLabel} to browse the published slice.`);
                  } else {
                    renderEmptyState('No paired samples match the current filters.');
                  }
                  countLabel.textContent = `Showing ${visibleItems.length} of ${items.length} paired samples`;
                  if (dataset === 'all') {
                    toolbarNote.textContent = `This gallery publishes ${data.counts.paired_samples} paired samples drawn from ${data.counts.available_paired_samples} available comparisons across both CFC flavors and the multi-texture oracle-vs-predicted slice, plus ${data.counts.single_flavor_records} curated single-flavor records for datasets without a paired baseline.`;
                  } else if (datasetInfo && datasetInfo.has_paired_comparison) {
                    const labels = pairFlavorLabels(datasetInfo);
                    toolbarNote.textContent = `Paired mode keeps ${labels[0]} on the left and ${labels[1]} on the right. Deltas are ${labels[1]} - ${labels[0]}. Use Load More to continue through the published ${datasetInfo.label} slice.`;
                  } else {
                    toolbarNote.textContent = datasetInfo.single_flavor_note || `${datasetInfo.label} currently publishes a single-flavor slice only.`;
                  }
                  updateLoadMore(items, 'Pairs');
                  syncUrl();
                  return;
                }

                const filteredRecords = data.per_image.filter((record) => {
                  if (dataset !== 'all' && record.dataset !== dataset) return false;
                  return record.flavor === view;
                });
                const items = sortItems(filteredRecords, sort);
                const visibleItems = visibleSlice(items);
                if (visibleItems.length) {
                  root.className = 'single-grid';
                  root.innerHTML = visibleItems.map(renderSingleCard).join('');
                } else {
                  renderEmptyState(`No published ${data.flavors[view].label.toLowerCase()} records match this filter.`);
                }
                countLabel.textContent = `Showing ${visibleItems.length} of ${items.length} ${data.flavors[view].label.toLowerCase()} samples`;
                if (datasetInfo && !datasetInfo.has_paired_comparison) {
                  toolbarNote.textContent = `${datasetInfo.single_flavor_note} This gallery publishes ${datasetInfo.published_record_count} curated records from ${datasetInfo.available_record_count} available ${datasetInfo.label} samples.`;
                } else if (datasetInfo) {
                  toolbarNote.textContent = `Single-flavor mode uses the published ${datasetInfo.label} slice: ${datasetInfo.published_record_count} records from ${datasetInfo.available_record_count} available sample renders.`;
                } else {
                  toolbarNote.textContent = `Single-flavor mode uses the same published gallery slice: ${data.counts.records} total records across all published datasets.`;
                }
                updateLoadMore(items, 'Samples');
                syncUrl();
              }

              root.addEventListener('click', (event) => {
                const target = event.target.closest('[data-zoom-src]');
                if (!target) return;
                dialogImg.src = target.getAttribute('data-zoom-src');
                dialog.showModal();
              });

              dialog.addEventListener('click', (event) => {
                if (event.target === dialog) dialog.close();
              });

              function rerenderFromFirstPage() {
                resetPaging();
                render();
              }

              datasetSelect.addEventListener('change', rerenderFromFirstPage);
              flavorSelect.addEventListener('change', rerenderFromFirstPage);
              sortSelect.addEventListener('change', rerenderFromFirstPage);
              pageSizeSelect.addEventListener('change', rerenderFromFirstPage);
              loadMoreBtn.addEventListener('click', () => {
                if (Number.isFinite(currentPageSize())) {
                  loadedPages += 1;
                }
                render();
              });
              render();
            })();
          </script>
        </body>

        </html>
        '''
    )
    return gallery.replace('__DATASET_OPTIONS__', ''.join(dataset_options)).replace('__FLAVOR_OPTIONS__', ''.join(flavor_options))


def build_manifest(metrics_payload: dict, training_payload: dict) -> str:
    dataset_lines = []
    for dataset in ALL_DATASET_ORDER:
        info = metrics_payload["datasets"][dataset]
        available_flavors = ", ".join(f'"{flavor}"' for flavor in info["available_flavors"])
        pair_flavors = ", ".join(f'"{flavor}"' for flavor in info["pair_flavors"])
        dataset_lines.extend(
            [
                f'  - slug: "{dataset}"',
                f'    id: "{info["dataset_id"]}"',
                f'    samples: {info["sample_count"]}',
                f'    leader_flavor: "{info["leader_flavor"]}"',
                f'    available_flavors: [{available_flavors}]',
                f'    pair_flavors: [{pair_flavors}]',
                f'    comparison_kind: "{info["comparison_kind"]}"',
            ]
        )
    cfc_labels = human_join([metrics_payload["datasets"][dataset]["label"] for dataset in DATASET_ORDER])
    multi_labels = human_join([metrics_payload["datasets"][dataset]["label"] for dataset in MULTI_TEXTURE_ORDER])
    single_flavor_labels = [
        metrics_payload["datasets"][dataset]["label"] for dataset in metrics_payload["summary"]["single_flavor_datasets"]
    ]
    single_flavor_suffix = ""
    if single_flavor_labels:
        verb = "publishes" if len(single_flavor_labels) == 1 else "publish"
        single_flavor_suffix = f" {human_join(single_flavor_labels)} currently {verb} flip_averaged only."
    multi_suffix = ""
    if multi_labels:
        multi_suffix = f" A separate multi-texture section compares Oracle K versus Predicted K on {multi_labels}."
    counts = training_payload["counts"]
    lines = [
        'title: "SAM 3 Cross-Dataset Benchmarking"',
        'model_id: "facebook/sam3"',
        'ui_standard: "premium"',
        f'date: "{date.today().isoformat()}"',
        f'comparison_contract: "{METRIC_CONTRACT}"',
        f'multi_texture_comparison_contract: "{MULTI_TEXTURE_METRIC_CONTRACT}"',
        f'description: "Flavor-aware benchmarking for SAM 3 Coarse Feature Clustering across {cfc_labels}.{single_flavor_suffix}{multi_suffix}"',
        'datasets:',
        *dataset_lines,
        'assets:',
        f'  published_preview_records: {counts["records"]}',
        f'  published_paired_samples: {counts["paired_samples"]}',
        f'  available_paired_samples: {counts["available_paired_samples"]}',
        f'  gallery_preview_count: {sum(metrics_payload["datasets"][dataset]["gallery_preview_count"] for dataset in ALL_DATASET_ORDER)}',
        'evaluation:',
        '  primary_metrics: ["mIoU", "ARI"]',
        '  metrics_file: "metrics.json"',
        '  training_data_file: "training_data.json"',
    ]
    return "\n".join(lines) + "\n"


def build_summary_md(metrics_payload: dict) -> str:
    info = metrics_payload["datasets"]
    summary = metrics_payload["summary"]
    single_flavor_note = ""
    if summary["single_flavor_datasets"]:
        single_dataset_bits = []
        for dataset in summary["single_flavor_datasets"]:
            single_dataset_bits.append(
                f"{info[dataset]['label']} is currently published as `flip_averaged` only "
                f"(`{info[dataset]['leader_mIoU']:.3f}` mIoU, `{info[dataset]['leader_ARI']:.3f}` ARI), "
                "so it expands coverage without a coarse-only delta yet."
            )
        single_flavor_note = "\n\n" + " ".join(single_dataset_bits)

    multi_texture_note = ""
    if MULTI_TEXTURE_ORDER:
        multi = info[MULTI_TEXTURE_ORDER[0]]
        if multi["comparable"]:
            left_flavor, right_flavor = multi["pair_flavors"]
            multi_texture_note = (
                "\n\n"
                f"Separately, `{multi['label']}` compares `{left_flavor}` against `{right_flavor}` under "
                f"`{MULTI_TEXTURE_METRIC_CONTRACT}`. `{FLAVORS[multi['leader_flavor']]['label']}` leads "
                f"(`{multi['leader_mIoU']:.3f}` mIoU, `{multi['leader_ARI']:.3f}` ARI), and the published delta "
                f"for `{right_flavor}` relative to `{left_flavor}` is `{signed(multi['delta_mIoU'])}` mIoU and "
                f"`{signed(multi['delta_ARI'])}` ARI."
            )
    return dedent(
        f'''\
        This page compares the published Coarse Feature Clustering flavors under the shared
        `{METRIC_CONTRACT}` evaluator (`mIoU`, `ARI`).

        Across the paired datasets, `flip_averaged` now leads RWTD (`{info['rwtd']['leader_mIoU']:.3f}` mIoU, `{info['rwtd']['leader_ARI']:.3f}` ARI)
        and STLD (`{info['stld']['leader_mIoU']:.3f}` mIoU, `{info['stld']['leader_ARI']:.3f}` ARI), with the largest
        gain on STLD (`{signed(info['stld']['delta_mIoU'])}` mIoU, `{signed(info['stld']['delta_ARI'])}` ARI).
        CAID still favors `coarse_only` (`{info['caid']['leader_mIoU']:.3f}` mIoU, `{info['caid']['leader_ARI']:.3f}` ARI).{single_flavor_note}{multi_texture_note}
        '''
    ).strip()


def build_report() -> tuple[dict, dict, str, str, str, dict]:
    # GitHub Pages artifact uploads reject symlinks, so the published overview
    # must be rebuilt as a self-contained asset tree each time.
    shutil.rmtree(DEST_DIR / "assets", ignore_errors=True)
    (DEST_DIR / "assets" / "all_previews").mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    featured_by_dataset: dict[str, list[dict]] = {}
    all_pairs_payload: list[dict] = []
    pairs_by_dataset: dict[str, list[dict]] = {}
    single_records_by_dataset: dict[str, list[dict]] = {}
    available_record_counts: dict[str, int] = {}
    available_pair_counts: dict[str, int] = {}

    dataset_metrics: dict[str, dict] = {}
    links_payload = {
        "source_runs": {},
        "metric_contract": METRIC_CONTRACT,
        "metric_note": METRIC_NOTE,
        "multi_texture_metric_contract": MULTI_TEXTURE_METRIC_CONTRACT,
    }

    for dataset in ALL_DATASET_ORDER:
        info = dataset_catalog(dataset)
        links_payload["source_runs"][dataset] = {}
        summary_by_flavor: dict[str, dict] = {}
        records_by_flavor: dict[str, dict[str, dict]] = {}
        dataset_pairs: list[dict] = []
        available_flavors = available_flavors_for_dataset(dataset)
        pair_flavors = pair_flavors_for_dataset(dataset)

        for flavor in available_flavors:
            run_dir = resolve_run_dir(dataset, flavor)
            summary = load_summary(run_dir)
            summary_by_flavor[flavor] = summary
            records_by_flavor[flavor] = load_visual_records(dataset, flavor, run_dir)
            links_payload["source_runs"][dataset][flavor] = summary["source_path"]

        sample_wins = Counter()
        comparable = all(flavor in summary_by_flavor for flavor in pair_flavors)

        if comparable:
            left_flavor, right_flavor = pair_flavors
            common_sample_ids = sorted(
                set(records_by_flavor[left_flavor]) & set(records_by_flavor[right_flavor]),
                key=lambda sample_id: (parse_sample_numeric(sample_id), sample_id),
            )

            for sample_id in common_sample_ids:
                left = records_by_flavor[left_flavor][sample_id]
                right = records_by_flavor[right_flavor][sample_id]
                delta_miou = round(right["mIoU"] - left["mIoU"], 4)
                delta_ari = round(right["ARI"] - left["ARI"], 4)
                winner = leader_flavor_from_scores(
                    left_flavor,
                    left["mIoU"],
                    left["ARI"],
                    right_flavor,
                    right["mIoU"],
                    right["ARI"],
                )
                sample_wins[winner] += 1
                left["pair_delta_mIoU"] = delta_miou
                left["pair_delta_ARI"] = delta_ari
                left["leader_flavor_for_sample"] = winner
                right["pair_delta_mIoU"] = delta_miou
                right["pair_delta_ARI"] = delta_ari
                right["leader_flavor_for_sample"] = winner
                pair_record = {
                    "pair_id": left["pair_id"],
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "sample_numeric": left["sample_numeric"],
                    "left_flavor": left_flavor,
                    "right_flavor": right_flavor,
                    "leader_flavor": winner,
                    "leader_mIoU": max(left["mIoU"], right["mIoU"]),
                    "delta_mIoU": delta_miou,
                    "delta_ARI": delta_ari,
                    "record_ids": {
                        left_flavor: left["id"],
                        right_flavor: right["id"],
                    },
                }
                if pair_flavors == FLAVOR_ORDER:
                    pair_record["delta_mIoU_flip_vs_coarse"] = delta_miou
                    pair_record["delta_ARI_flip_vs_coarse"] = delta_ari
                dataset_pairs.append(pair_record)
                all_pairs_payload.append(pair_record)

            leader = leader_flavor_from_scores(
                left_flavor,
                summary_by_flavor[left_flavor]["mIoU"],
                summary_by_flavor[left_flavor]["ARI"],
                right_flavor,
                summary_by_flavor[right_flavor]["mIoU"],
                summary_by_flavor[right_flavor]["ARI"],
            )
            if leader == "tie":
                leader = right_flavor

            delta_miou = round(summary_by_flavor[right_flavor]["mIoU"] - summary_by_flavor[left_flavor]["mIoU"], 6)
            delta_ari = round(summary_by_flavor[right_flavor]["ARI"] - summary_by_flavor[left_flavor]["ARI"], 6)
        else:
            leader = max(
                available_flavors,
                key=lambda flavor: (
                    summary_by_flavor[flavor]["mIoU"],
                    summary_by_flavor[flavor]["ARI"],
                    -GALLERY_FLAVOR_ORDER.index(flavor),
                ),
            )
            delta_miou = None
            delta_ari = None
            for flavor in available_flavors:
                for record in records_by_flavor[flavor].values():
                    record["leader_flavor_for_sample"] = flavor

        sample_wins_payload = {flavor: sample_wins.get(flavor, 0) for flavor in pair_flavors}
        sample_wins_payload["ties"] = sample_wins.get("tie", 0)
        dataset_metrics[dataset] = {
            "label": info["label"],
            "title": info["title"],
            "description": info["description"],
            "dataset_id": info["dataset_id"],
            "comparison_kind": "multi_texture" if dataset in MULTI_TEXTURE_DATASETS else "cfc",
            "comparison_note": info.get("comparison_note"),
            "pair_flavors": pair_flavors,
            "sample_count": max(summary_by_flavor[flavor]["sample_count"] for flavor in available_flavors),
            "available_flavors": available_flavors,
            "comparable": comparable,
            "leader_flavor": leader,
            "leader_mIoU": round(summary_by_flavor[leader]["mIoU"], 6),
            "leader_ARI": round(summary_by_flavor[leader]["ARI"], 6),
            "delta_mIoU": delta_miou,
            "delta_ARI": delta_ari,
            "delta_mIoU_flip_vs_coarse": delta_miou if pair_flavors == FLAVOR_ORDER and comparable else None,
            "delta_ARI_flip_vs_coarse": delta_ari if pair_flavors == FLAVOR_ORDER and comparable else None,
            "single_flavor_note": info.get("single_flavor_note"),
            "sample_wins": sample_wins_payload,
            "flavors": {
                flavor: {
                    "mIoU": round(summary_by_flavor[flavor]["mIoU"], 6),
                    "ARI": round(summary_by_flavor[flavor]["ARI"], 6),
                    "mIoU_agg": round(summary_by_flavor[flavor]["mIoU_agg"], 6),
                    "variant": summary_by_flavor[flavor]["variant"],
                    "evaluation_contract": summary_by_flavor[flavor]["evaluation_contract"],
                    "evaluation_view": summary_by_flavor[flavor]["evaluation_view"],
                    "source_path": summary_by_flavor[flavor]["source_path"],
                }
                if flavor in summary_by_flavor
                else None
                for flavor in GALLERY_FLAVOR_ORDER
            },
        }

        leader_records = sorted(
            records_by_flavor[leader].values(),
            key=lambda record: (-record["mIoU"], record["sample_numeric"], record["sample_id"]),
        )[:4]
        featured_by_dataset[dataset] = list(leader_records)

        pairs_by_dataset[dataset] = dataset_pairs
        for flavor in available_flavors:
            all_records.extend(records_by_flavor[flavor].values())
        if not comparable:
            single_records_by_dataset[dataset] = select_single_flavor_records(
                list(records_by_flavor[leader].values()),
                [record["sample_id"] for record in leader_records],
            )
        available_record_counts[dataset] = sum(len(records_by_flavor[flavor]) for flavor in available_flavors)
        available_pair_counts[dataset] = len(dataset_pairs)
        dataset_metrics[dataset]["gallery_preview_count"] = len(featured_by_dataset[dataset])

    record_lookup = {record["id"]: record for record in all_records}
    published_pairs: list[dict] = []
    published_records: list[dict] = []
    published_record_ids: set[str] = set()
    published_record_counts = Counter()
    published_pair_counts = Counter()
    published_single_records = 0

    for dataset in ALL_DATASET_ORDER:
        if dataset_metrics[dataset]["comparable"]:
            featured_sample_ids = [record["sample_id"] for record in featured_by_dataset[dataset]]
            dataset_publish_pairs = select_gallery_pairs(pairs_by_dataset[dataset], featured_sample_ids)
            published_pairs.extend(dataset_publish_pairs)
            published_pair_counts[dataset] = len(dataset_publish_pairs)

            for pair in dataset_publish_pairs:
                for flavor in pair["record_ids"]:
                    record = record_lookup[pair["record_ids"][flavor]]
                    if record["id"] in published_record_ids:
                        continue
                    materialize_web_asset(record)
                    published_records.append(record)
                    published_record_ids.add(record["id"])
                    published_record_counts[dataset] += 1
        else:
            for record in single_records_by_dataset.get(dataset, []):
                if record["id"] in published_record_ids:
                    continue
                materialize_web_asset(record)
                published_records.append(record)
                published_record_ids.add(record["id"])
                published_record_counts[dataset] += 1
                published_single_records += 1

        for record in featured_by_dataset[dataset]:
            if record["id"] in published_record_ids:
                continue
            materialize_web_asset(record)
            published_records.append(record)
            published_record_ids.add(record["id"])
            published_record_counts[dataset] += 1

    public_records = [
        to_public_record(record)
        for record in sorted(
            published_records,
            key=lambda record: (
                ALL_DATASET_ORDER.index(record["dataset"]),
                GALLERY_FLAVOR_ORDER.index(record["flavor"]),
                record["sample_numeric"],
                record["sample_id"],
            ),
        )
    ]
    published_pairs = sorted(
        published_pairs,
        key=lambda pair: (ALL_DATASET_ORDER.index(pair["dataset"]), pair["sample_numeric"], pair["sample_id"]),
    )

    comparable_datasets = [dataset for dataset in DATASET_ORDER if dataset_metrics[dataset]["comparable"]]
    single_flavor_datasets = [dataset for dataset in DATASET_ORDER if not dataset_metrics[dataset]["comparable"]]
    datasets_led = Counter(dataset_metrics[dataset]["leader_flavor"] for dataset in comparable_datasets)
    cfc_available_pairs = sum(available_pair_counts[dataset] for dataset in DATASET_ORDER)
    multi_texture_comparable = [dataset for dataset in MULTI_TEXTURE_ORDER if dataset_metrics[dataset]["comparable"]]

    summary_payload = {
        "paired_samples": cfc_available_pairs,
        "datasets_total": len(DATASET_ORDER),
        "comparable_dataset_count": len(comparable_datasets),
        "comparable_datasets": comparable_datasets,
        "single_flavor_count": len(single_flavor_datasets),
        "single_flavor_datasets": single_flavor_datasets,
        "datasets_led": {flavor: datasets_led.get(flavor, 0) for flavor in FLAVOR_ORDER},
        "datasets_led_by": max(
            FLAVOR_ORDER,
            key=lambda flavor: datasets_led.get(flavor, 0),
        ),
        "best_gain_dataset_mIoU": max(
            comparable_datasets, key=lambda dataset: dataset_metrics[dataset]["delta_mIoU"]
        ),
        "best_gain_dataset_ARI": max(
            comparable_datasets, key=lambda dataset: dataset_metrics[dataset]["delta_ARI"]
        ),
        "best_gain_mIoU": max(dataset_metrics[dataset]["delta_mIoU"] for dataset in comparable_datasets),
        "best_gain_ARI": max(dataset_metrics[dataset]["delta_ARI"] for dataset in comparable_datasets),
    }
    multi_texture_summary = {
        "datasets_total": len(MULTI_TEXTURE_ORDER),
        "comparable_dataset_count": len(multi_texture_comparable),
        "paired_samples": sum(available_pair_counts[dataset] for dataset in MULTI_TEXTURE_ORDER),
        "datasets": multi_texture_comparable,
    }
    gallery_summary = {
        "records": len(public_records),
        "paired_samples": len(published_pairs),
        "available_paired_samples": len(all_pairs_payload),
        "single_flavor_records": published_single_records,
        "available_single_flavor_records": sum(
            available_record_counts[dataset]
            for dataset in ALL_DATASET_ORDER
            if not dataset_metrics[dataset]["comparable"]
        ),
    }

    metrics_payload = {
        "metric_contract": METRIC_CONTRACT,
        "multi_texture_metric_contract": MULTI_TEXTURE_METRIC_CONTRACT,
        "metric_note": METRIC_NOTE,
        "dataset_order": ALL_DATASET_ORDER,
        "cfc_dataset_order": DATASET_ORDER,
        "multi_texture_order": MULTI_TEXTURE_ORDER,
        "flavor_order": GALLERY_FLAVOR_ORDER,
        "cfc_flavor_order": FLAVOR_ORDER,
        "multi_texture_flavor_order": MULTI_TEXTURE_FLAVOR_ORDER,
        "datasets": dataset_metrics,
        "summary": summary_payload,
        "multi_texture_summary": multi_texture_summary,
        "gallery_summary": gallery_summary,
    }

    training_payload = {
        "metric_contract": METRIC_CONTRACT,
        "multi_texture_metric_contract": MULTI_TEXTURE_METRIC_CONTRACT,
        "metric_note": METRIC_NOTE,
        "dataset_order": ALL_DATASET_ORDER,
        "cfc_dataset_order": DATASET_ORDER,
        "multi_texture_order": MULTI_TEXTURE_ORDER,
        "flavor_order": GALLERY_FLAVOR_ORDER,
        "cfc_flavor_order": FLAVOR_ORDER,
        "multi_texture_flavor_order": MULTI_TEXTURE_FLAVOR_ORDER,
        "datasets": {
            dataset: {
                "label": dataset_catalog(dataset)["label"],
                "dataset_id": dataset_catalog(dataset)["dataset_id"],
                "description": dataset_catalog(dataset)["description"],
                "comparison_kind": dataset_metrics[dataset]["comparison_kind"],
                "comparison_note": dataset_metrics[dataset]["comparison_note"],
                "pair_flavors": dataset_metrics[dataset]["pair_flavors"],
                "available_flavors": dataset_metrics[dataset]["available_flavors"],
                "has_paired_comparison": dataset_metrics[dataset]["comparable"],
                "single_flavor_note": dataset_metrics[dataset]["single_flavor_note"],
                "published_record_count": published_record_counts[dataset],
                "available_record_count": available_record_counts[dataset],
                "published_pair_count": published_pair_counts[dataset],
                "available_pair_count": available_pair_counts[dataset],
            }
            for dataset in ALL_DATASET_ORDER
        },
        "flavors": {flavor: FLAVORS[flavor] for flavor in GALLERY_FLAVOR_ORDER},
        "counts": gallery_summary,
        "per_image": public_records,
        "pairs": published_pairs,
    }

    return (
        metrics_payload,
        training_payload,
        build_index_html(metrics_payload, featured_by_dataset),
        build_gallery_html(),
        build_summary_md(metrics_payload),
        links_payload,
    )


def build_index_html(metrics_payload: dict, featured_by_dataset: dict[str, list[dict]]) -> str:
    return build_dataset_metrics(metrics_payload, featured_by_dataset)


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    metrics_payload, training_payload, index_html, gallery_html, summary_md, links_payload = build_report()
    public_records = training_payload["per_image"]
    pairs_payload = training_payload["pairs"]

    (DEST_DIR / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    (DEST_DIR / "training_data.json").write_text(json.dumps(training_payload, indent=2), encoding="utf-8")
    (DEST_DIR / "index.html").write_text(index_html, encoding="utf-8")
    (DEST_DIR / "gallery.html").write_text(gallery_html, encoding="utf-8")
    (DEST_DIR / "summary.md").write_text(summary_md + "\n", encoding="utf-8")
    (DEST_DIR / "manifest.yaml").write_text(
        build_manifest(metrics_payload, training_payload),
        encoding="utf-8",
    )
    (DEST_DIR / "links.json").write_text(json.dumps(links_payload, indent=2), encoding="utf-8")

    print(
        f"Built cross-dataset overview with {len(public_records)} flavor-specific previews and "
        f"{len(pairs_payload)} paired comparisons."
    )


if __name__ == "__main__":
    main()
