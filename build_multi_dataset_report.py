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

DATASET_ORDER = ["rwtd", "caid", "stld", "cstd"]
FLAVOR_ORDER = ["coarse_only", "flip_averaged"]

METRIC_CONTRACT = "architexture_binary_v1"
METRIC_NOTE = (
    "Cross-flavor comparison is normalized onto the shared ArchiTexture binary evaluator "
    "(mIoU, ARI) so RWTD legacy aggregate-only summaries do not distort the page."
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


def asset_relpath(dataset: str, flavor: str, file_name: str) -> Path:
    # Keep the historical coarse-only asset paths stable to avoid needless publish churn.
    if flavor == "coarse_only":
        return Path(dataset) / file_name
    return Path(flavor) / dataset / file_name


def available_flavors_for_dataset(dataset: str) -> list[str]:
    return list(DATASETS[dataset].get("available_flavors", FLAVOR_ORDER))


def resolve_run_dir(dataset: str, flavor: str) -> Path:
    override_candidates = DATASETS[dataset].get("run_dir_candidates", {}).get(flavor)
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


def leader_flavor_from_scores(left_miou: float, left_ari: float, right_miou: float, right_ari: float) -> str:
    if right_miou > left_miou:
        return "flip_averaged"
    if right_miou < left_miou:
        return "coarse_only"
    if right_ari > left_ari:
        return "flip_averaged"
    if right_ari < left_ari:
        return "coarse_only"
    return "tie"


def load_summary(run_dir: Path) -> dict:
    payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    metrics = payload["mean_metrics"]
    return {
        "mIoU": float(metrics.get("eval_miou", metrics["miou"])),
        "ARI": float(metrics.get("eval_ari", metrics["ari"])),
        "mIoU_agg": float(metrics.get("miou_agg", 0.0)),
        "variant": payload.get("variant", ""),
        "evaluation_contract": payload.get("evaluation_contract", "legacy_summary_metric"),
        "evaluation_view": payload.get("evaluation_view", ""),
        "sample_count": int(payload.get("num_evaluated_samples", 0)),
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
                "delta_mIoU_flip_vs_coarse": None,
                "delta_ARI_flip_vs_coarse": None,
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
                -pair["delta_mIoU_flip_vs_coarse"],
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
                pair["delta_mIoU_flip_vs_coarse"],
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
                -abs(pair["delta_mIoU_flip_vs_coarse"]),
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
    if single_flavor_datasets:
        single_flavor_blurbs = []
        for dataset in single_flavor_datasets:
            info = metrics_payload["datasets"][dataset]
            flavor_labels = ", ".join(FLAVORS[flavor]["label"] for flavor in info["available_flavors"])
            single_flavor_blurbs.append(f'{info["label"]} currently publishes {flavor_labels} only')
        coverage_copy = (
            "Flavor-aware benchmarking for Coarse Feature Clustering across RWTD, CAID, STLD, and CSTD. "
            + "; ".join(single_flavor_blurbs)
            + "."
        )
        single_label_copy = ", ".join(metrics_payload["datasets"][dataset]["label"] for dataset in single_flavor_datasets)
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
            "Flavor-aware benchmarking for Coarse Feature Clustering across RWTD, CAID, and STLD, now including the "
            "new <code>flip_averaged</code> variant."
        )
        executive_copy = (
            "The new <strong>Flip Averaged</strong> flavor changes the cross-dataset picture materially. It takes the "
            "lead on <strong>RWTD</strong> and <strong>STLD</strong>, with the largest jump on STLD, while the "
            "original <strong>Coarse Only</strong> flavor still leads CAID."
        )

    rows = []
    for dataset in DATASET_ORDER:
        info = metrics_payload["datasets"][dataset]
        coarse = info["flavors"].get("coarse_only")
        flip = info["flavors"].get("flip_averaged")
        leader = FLAVORS[info["leader_flavor"]]["label"]
        delta_miou = info["delta_mIoU_flip_vs_coarse"]
        delta_ari = info["delta_ARI_flip_vs_coarse"]
        delta_class = "good" if (delta_miou or 0) > 0 else "bad" if (delta_miou or 0) < 0 else ""
        coarse_cell = (
            f"""
                <strong>{coarse["mIoU"]:.3f}</strong>
                <div class="mini-note">ARI {coarse["ARI"]:.3f}</div>
            """
            if coarse
            else """
                <strong>—</strong>
                <div class="mini-note">not published</div>
            """
        )
        flip_cell = (
            f"""
                <strong>{flip["mIoU"]:.3f}</strong>
                <div class="mini-note">ARI {flip["ARI"]:.3f}</div>
            """
            if flip
            else """
                <strong>—</strong>
                <div class="mini-note">not published</div>
            """
        )
        if delta_miou is None or delta_ari is None:
            delta_miou_cell = '<span class="delta-pill">n/a</span>'
            delta_ari_cell = '<span class="delta-pill">n/a</span>'
        else:
            delta_miou_cell = f'<span class="delta-pill {delta_class}">{signed(delta_miou)}</span>'
            delta_ari_cell = f'<span class="delta-pill {delta_class}">{signed(delta_ari)}</span>'
        leader_note = '<div class="mini-note">single-flavor only</div>' if not info["comparable"] else ""
        rows.append(
            f"""
            <tr>
              <td>
                <strong>{escape(info["label"])}</strong>
                <div class="mini-note">{escape(DATASETS[dataset]["dataset_id"])}</div>
              </td>
              <td>{coarse_cell}</td>
              <td>{flip_cell}</td>
              <td>{delta_miou_cell}</td>
              <td>{delta_ari_cell}</td>
              <td><span class="tag {'good' if info["leader_flavor"] == 'flip_averaged' else 'warn'}">{escape(leader)}</span>{leader_note}</td>
            </tr>
            """
        )

    cards = []
    for dataset in DATASET_ORDER:
        info = metrics_payload["datasets"][dataset]
        leader_flavor = info["leader_flavor"]
        leader_label = FLAVORS[leader_flavor]["label"]
        leader_tag = "good" if leader_flavor == "flip_averaged" else "warn"
        delta_miou = info["delta_mIoU_flip_vs_coarse"]
        delta_ari = info["delta_ARI_flip_vs_coarse"]
        if info["comparable"] and delta_miou is not None and delta_ari is not None and delta_miou >= 0:
            summary_copy = f"Flip Averaged leads by {signed(delta_miou)} mIoU and {signed(delta_ari)} ARI."
        elif info["comparable"] and delta_miou is not None and delta_ari is not None:
            summary_copy = f"Coarse Only holds the lead by {abs(delta_miou):.3f} mIoU and {abs(delta_ari):.3f} ARI."
        else:
            summary_copy = info["single_flavor_note"]
        previews = []
        for record in featured[dataset][:CARD_PREVIEW_LIMIT]:
            src = f"assets/all_previews/{record['file_path']}"
            previews.append(
                f"""
                <figure>
                  <img src="{escape(src)}" alt="{escape(info["label"])} sample {escape(record["sample_id"])}"
                    data-zoom-src="{escape(src)}" />
                  <figcaption>
                    <span class="pill">id {escape(record["sample_id"])}</span>
                    <span class="pill">mIoU {record["mIoU"]:.3f}</span>
                  </figcaption>
                </figure>
                """
            )
        if info["comparable"]:
            win_counts = info["sample_wins"]
            stats_html = (
                f"""
              <div class="leader-stats">
                <span class="pill">Leader mIoU {info["leader_mIoU"]:.3f}</span>
                <span class="pill">Leader ARI {info["leader_ARI"]:.3f}</span>
                <span class="pill">Wins F/C/T {win_counts["flip_averaged"]}/{win_counts["coarse_only"]}/{win_counts["ties"]}</span>
              </div>
                """
            )
            availability_tag = ""
            action_href = f"gallery.html?dataset={dataset}&view=paired"
            action_label = f"Open {escape(info['label'])} 1:1 Gallery"
        else:
            published_flavors = ", ".join(FLAVORS[flavor]["label"] for flavor in info["available_flavors"])
            stats_html = (
                f"""
              <div class="leader-stats">
                <span class="pill">mIoU {info["leader_mIoU"]:.3f}</span>
                <span class="pill">ARI {info["leader_ARI"]:.3f}</span>
                <span class="pill">Samples {info["sample_count"]:,}</span>
              </div>
                """
            )
            availability_tag = f'<span class="tag">Published: {escape(published_flavors)} only</span>'
            action_href = f"gallery.html?dataset={dataset}&view={info['available_flavors'][0]}"
            action_label = f"Open {escape(info['label'])} {escape(FLAVORS[info['available_flavors'][0]]['label'])} Gallery"
        cards.append(
            f"""
            <article class="dataset-card cfc-card">
              <div class="card-topline">
                <span class="tag">{escape(info["label"])}</span>
                <span class="tag {leader_tag}">Leader: {escape(leader_label)}</span>
                {availability_tag}
              </div>
              <h3>{escape(info["title"])}</h3>
              <p class="kv card-summary">{escape(summary_copy)}</p>
              {stats_html}
              <div class="preview-grid">
                {"".join(previews)}
              </div>
              <div class="card-actions">
                <a class="btn secondary" href="{action_href}">{action_label}</a>
              </div>
            </article>
            """
        )

    page = dedent(
        """\
        <!DOCTYPE html>
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
                      <th>ΔmIoU</th>
                      <th>ΔARI</th>
                      <th>Leader</th>
                    </tr>
                  </thead>
                  <tbody>
                    __TABLE_ROWS__
                  </tbody>
                </table>
              </div>
            </section>

            <section class="section">
              <div style="display: flex; justify-content: space-between; gap: 20px; align-items: end; flex-wrap: wrap;">
                <div>
                  <h2>Dataset Leaders</h2>
                  <p class="kv">Fast read: who leads, by how much, and two representative samples.</p>
                </div>
                <a class="btn" href="gallery.html">Open Gallery</a>
              </div>
              <div class="dataset-grid" style="margin-top: 22px;">
                __DATASET_CARDS__
              </div>
            </section>

            <footer style="margin-top: 80px; text-align: center;">
              <p><a href="../../index.html" style="text-decoration: none; color: var(--brand); font-weight: 600;">← Back to Dashboard</a></p>
              <p style="margin-top: 12px;">Research Analytics by <strong>Antigravity</strong></p>
            </footer>
          </div>

          <dialog class="image-dialog" id="imageDialog">
            <img id="imageDialogImg" alt="Expanded preview" />
          </dialog>
          <script src="../../assets/site.js"></script>
        </body>

        </html>
        """
    )
    return (
        page.replace("__METRIC_NOTE__", escape(METRIC_NOTE))
        .replace("__COVERAGE_COPY__", coverage_copy)
        .replace("__EXECUTIVE_COPY__", executive_copy)
        .replace("__HEADLINE_CARDS__", build_headline_cards(metrics_payload))
        .replace("__TABLE_ROWS__", "".join(rows))
        .replace("__DATASET_CARDS__", "".join(cards))
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
            f'<option value="{dataset}">{escape(DATASETS[dataset]["label"])} ({DATASETS[dataset]["dataset_id"]})</option>'
            for dataset in DATASET_ORDER
        ],
    ]
    gallery = dedent(
        """\
        <!DOCTYPE html>
        <html lang="en">

        <head>
          <meta charset="UTF-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1.0" />
          <title>Cross-Flavor Gallery — SAM 3</title>
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
              <h1 class="title-gradient">Cross-Flavor / Single-Flavor Gallery</h1>
              <p style="color: var(--muted); margin-top: 16px; font-weight: 300; max-width: 860px; margin-inline: auto;">
                Browse paired same-sample comparisons where both flavors exist, or switch into a single-flavor slice
                for datasets like CSTD that currently publish only <code>flip_averaged</code>. The headline metrics on
                the overview page still reflect the full benchmark.
              </p>
            </header>

            <section class="section">
              <div class="gallery-toolbar">
                <label for="datasetSelect">Dataset</label>
                <select id="datasetSelect">__DATASET_OPTIONS__</select>

                <label for="flavorSelect">Flavor</label>
                <select id="flavorSelect">
                  <option value="paired">1:1 Comparison</option>
                  <option value="coarse_only">Coarse Only</option>
                  <option value="flip_averaged">Flip Averaged</option>
                </select>

                <label for="sortSelect">Sort</label>
                <select id="sortSelect">
                  <option value="delta_desc">ΔmIoU ↓ biggest first</option>
                  <option value="score_desc">Score ↓ best first</option>
                  <option value="score_asc">Score ↑ worst first</option>
                  <option value="sample_id_asc">Sample ID ↑</option>
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

                <div class="toolbar-count" id="countLabel">Loading…</div>
                <div class="toolbar-note" id="toolbarNote"></div>
              </div>

              <div id="galleryRoot" class="gallery-stack"></div>
              <div class="gallery-controls" id="galleryControls" hidden>
                <button class="btn secondary" id="loadMoreBtn" type="button">Load More</button>
              </div>
            </section>

            <footer style="margin-top: 60px; text-align: center;">
              <p><a href="index.html" style="text-decoration: none; color: var(--brand); font-weight: 600;">← Back to report</a></p>
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
                  } else if (availableFlavors.has('flip_averaged')) {
                    flavorSelect.value = 'flip_averaged';
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
                  const leftDelta = left.delta_mIoU_flip_vs_coarse ?? 0;
                  const rightDelta = right.delta_mIoU_flip_vs_coarse ?? 0;
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
                const coarse = recordMap.get(pair.record_ids.coarse_only);
                const flip = recordMap.get(pair.record_ids.flip_averaged);
                return `
                  <article class="pair-card">
                    <div class="pair-head">
                      <span class="pill dataset-pill">${data.datasets[pair.dataset].label}</span>
                      <span class="pill">id ${pair.sample_id}</span>
                      <span class="pill ${pair.leader_flavor === 'flip_averaged' ? 'good' : pair.leader_flavor === 'coarse_only' ? 'warn' : ''}">
                        Leader: ${pair.leader_flavor === 'tie' ? 'Tie' : data.flavors[pair.leader_flavor].label}
                      </span>
                      <span class="pill ${deltaClass(pair.delta_mIoU_flip_vs_coarse)}">ΔmIoU ${pair.delta_mIoU_flip_vs_coarse.toFixed(3)}</span>
                      <span class="pill ${deltaClass(pair.delta_ARI_flip_vs_coarse)}">ΔARI ${pair.delta_ARI_flip_vs_coarse.toFixed(3)}</span>
                    </div>
                    <div class="pair-grid">
                      ${renderPanel(coarse)}
                      ${renderPanel(flip)}
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
                const deltaValue = typeof record.delta_mIoU_flip_vs_coarse === 'number'
                  ? `<span class="pill ${deltaClass(record.delta_mIoU_flip_vs_coarse)}">ΔmIoU ${record.delta_mIoU_flip_vs_coarse.toFixed(3)}</span>`
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
                    renderEmptyState(`${datasetInfo.label} currently publishes only Flip Averaged. Switch the Flavor control to browse its curated single-flavor slice.`);
                  } else {
                    renderEmptyState('No paired samples match the current filters.');
                  }
                  countLabel.textContent = `Showing ${visibleItems.length} of ${items.length} paired samples`;
                  if (dataset === 'all') {
                    toolbarNote.textContent = `This gallery publishes ${data.counts.paired_samples} paired samples drawn from ${data.counts.available_paired_samples} available benchmark pairs, plus ${data.counts.single_flavor_records} curated single-flavor records for datasets without a vanilla baseline.`;
                  } else if (datasetInfo && datasetInfo.has_paired_comparison) {
                    toolbarNote.textContent = `Paired mode keeps Coarse Only on the left and Flip Averaged on the right. Use Load More to continue through the published ${datasetInfo.label} slice.`;
                  } else {
                    toolbarNote.textContent = `${datasetInfo.single_flavor_note} Switch Flavor to Flip Averaged to browse the published slice.`;
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
        """
    )
    return gallery.replace("__DATASET_OPTIONS__", "".join(dataset_options))


def build_manifest(metrics_payload: dict, public_records: list[dict], pairs_payload: list[dict]) -> str:
    dataset_lines = []
    for dataset in DATASET_ORDER:
        info = metrics_payload["datasets"][dataset]
        available_flavors = ", ".join(f'"{flavor}"' for flavor in info["available_flavors"])
        dataset_lines.append(
            f"""  - slug: "{dataset}"
    id: "{info['dataset_id']}"
    samples: {info['sample_count']}
    leader_flavor: "{info['leader_flavor']}"
    available_flavors: [{available_flavors}]
"""
        )
    return (
        "title: \"SAM 3 Cross-Dataset Benchmarking\"\n"
        "model_id: \"facebook/sam3\"\n"
        "ui_standard: \"premium\"\n"
        f"date: \"{date.today().isoformat()}\"\n"
        f"comparison_contract: \"{METRIC_CONTRACT}\"\n"
        "description: \"Flavor-aware benchmarking for SAM 3 Coarse Feature Clustering across RWTD, CAID, STLD, and CSTD. CSTD currently publishes flip_averaged only.\"\n"
        "datasets:\n"
        f"{''.join(dataset_lines)}"
        "assets:\n"
        f"  published_preview_records: {len(public_records)}\n"
        f"  published_paired_samples: {len(pairs_payload)}\n"
        f"  available_paired_samples: {metrics_payload['summary']['paired_samples']}\n"
        f"  gallery_preview_count: {sum(metrics_payload['datasets'][dataset]['gallery_preview_count'] for dataset in DATASET_ORDER)}\n"
        "evaluation:\n"
        "  primary_metrics: [\"mIoU\", \"ARI\"]\n"
        "  metrics_file: \"metrics.json\"\n"
        "  training_data_file: \"training_data.json\"\n"
    )


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
    return dedent(
        f"""\
        This page compares the published Coarse Feature Clustering flavors under the shared
        `{METRIC_CONTRACT}` evaluator (`mIoU`, `ARI`).

        Across the paired datasets, `flip_averaged` now leads RWTD (`{info['rwtd']['leader_mIoU']:.3f}` mIoU, `{info['rwtd']['leader_ARI']:.3f}` ARI)
        and STLD (`{info['stld']['leader_mIoU']:.3f}` mIoU, `{info['stld']['leader_ARI']:.3f}` ARI), with the largest
        gain on STLD (`{signed(info['stld']['delta_mIoU_flip_vs_coarse'])}` mIoU, `{signed(info['stld']['delta_ARI_flip_vs_coarse'])}` ARI).
        CAID still favors `coarse_only` (`{info['caid']['leader_mIoU']:.3f}` mIoU, `{info['caid']['leader_ARI']:.3f}` ARI).{single_flavor_note}
        """
    ).strip()


def build_report() -> tuple[dict, dict, str, str, str, dict]:
    # GitHub Pages artifact uploads reject symlinks, so the published overview
    # must be rebuilt as a self-contained asset tree each time.
    shutil.rmtree(DEST_DIR / "assets", ignore_errors=True)
    (DEST_DIR / "assets" / "all_previews").mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict] = {}
    all_records: list[dict] = []
    featured_by_dataset: dict[str, list[dict]] = {}
    all_pairs_payload: list[dict] = []
    pairs_by_dataset: dict[str, list[dict]] = {}
    single_records_by_dataset: dict[str, list[dict]] = {}
    available_record_counts: dict[str, int] = {}
    available_pair_counts: dict[str, int] = {}

    dataset_metrics: dict[str, dict] = {}
    links_payload = {"source_runs": {}, "metric_contract": METRIC_CONTRACT, "metric_note": METRIC_NOTE}

    for dataset in DATASET_ORDER:
        links_payload["source_runs"][dataset] = {}
        summary_by_flavor: dict[str, dict] = {}
        records_by_flavor: dict[str, dict[str, dict]] = {}
        dataset_pairs: list[dict] = []
        available_flavors = available_flavors_for_dataset(dataset)

        for flavor in available_flavors:
            run_dir = resolve_run_dir(dataset, flavor)
            summary = load_summary(run_dir)
            summary_by_flavor[flavor] = summary
            records_by_flavor[flavor] = load_visual_records(dataset, flavor, run_dir)
            links_payload["source_runs"][dataset][flavor] = summary["source_path"]

        sample_wins = Counter()
        comparable = all(flavor in summary_by_flavor for flavor in FLAVOR_ORDER)

        if comparable:
            common_sample_ids = sorted(
                set(records_by_flavor["coarse_only"]) & set(records_by_flavor["flip_averaged"]),
                key=lambda sample_id: (parse_sample_numeric(sample_id), sample_id),
            )

            for sample_id in common_sample_ids:
                coarse = records_by_flavor["coarse_only"][sample_id]
                flip = records_by_flavor["flip_averaged"][sample_id]
                delta_miou = round(flip["mIoU"] - coarse["mIoU"], 4)
                delta_ari = round(flip["ARI"] - coarse["ARI"], 4)
                winner = leader_flavor_from_scores(coarse["mIoU"], coarse["ARI"], flip["mIoU"], flip["ARI"])
                sample_wins[winner] += 1
                coarse["delta_mIoU_flip_vs_coarse"] = delta_miou
                coarse["delta_ARI_flip_vs_coarse"] = delta_ari
                coarse["leader_flavor_for_sample"] = winner
                flip["delta_mIoU_flip_vs_coarse"] = delta_miou
                flip["delta_ARI_flip_vs_coarse"] = delta_ari
                flip["leader_flavor_for_sample"] = winner
                pair_record = {
                    "pair_id": coarse["pair_id"],
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "sample_numeric": coarse["sample_numeric"],
                    "leader_flavor": winner,
                    "leader_mIoU": max(coarse["mIoU"], flip["mIoU"]),
                    "delta_mIoU_flip_vs_coarse": delta_miou,
                    "delta_ARI_flip_vs_coarse": delta_ari,
                    "record_ids": {
                        "coarse_only": coarse["id"],
                        "flip_averaged": flip["id"],
                    },
                }
                dataset_pairs.append(pair_record)
                all_pairs_payload.append(pair_record)

            leader = leader_flavor_from_scores(
                summary_by_flavor["coarse_only"]["mIoU"],
                summary_by_flavor["coarse_only"]["ARI"],
                summary_by_flavor["flip_averaged"]["mIoU"],
                summary_by_flavor["flip_averaged"]["ARI"],
            )
            if leader == "tie":
                leader = "flip_averaged"

            delta_miou = round(
                summary_by_flavor["flip_averaged"]["mIoU"] - summary_by_flavor["coarse_only"]["mIoU"], 6
            )
            delta_ari = round(
                summary_by_flavor["flip_averaged"]["ARI"] - summary_by_flavor["coarse_only"]["ARI"], 6
            )
        else:
            leader = max(
                available_flavors,
                key=lambda flavor: (
                    summary_by_flavor[flavor]["mIoU"],
                    summary_by_flavor[flavor]["ARI"],
                    -FLAVOR_ORDER.index(flavor),
                ),
            )
            delta_miou = None
            delta_ari = None
            for flavor in available_flavors:
                for record in records_by_flavor[flavor].values():
                    record["leader_flavor_for_sample"] = flavor

        dataset_metrics[dataset] = {
            "label": DATASETS[dataset]["label"],
            "title": DATASETS[dataset]["title"],
            "description": DATASETS[dataset]["description"],
            "dataset_id": DATASETS[dataset]["dataset_id"],
            "sample_count": max(summary_by_flavor[flavor]["sample_count"] for flavor in available_flavors),
            "available_flavors": available_flavors,
            "comparable": comparable,
            "leader_flavor": leader,
            "leader_mIoU": round(summary_by_flavor[leader]["mIoU"], 6),
            "leader_ARI": round(summary_by_flavor[leader]["ARI"], 6),
            "delta_mIoU_flip_vs_coarse": delta_miou,
            "delta_ARI_flip_vs_coarse": delta_ari,
            "single_flavor_note": DATASETS[dataset].get("single_flavor_note"),
            "sample_wins": {
                "coarse_only": sample_wins.get("coarse_only", 0),
                "flip_averaged": sample_wins.get("flip_averaged", 0),
                "ties": sample_wins.get("tie", 0),
            },
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
                for flavor in FLAVOR_ORDER
            },
        }

        leader_records = sorted(
            records_by_flavor[leader].values(),
            key=lambda record: (-record["mIoU"], record["sample_numeric"], record["sample_id"]),
        )[:4]
        featured_by_dataset[dataset] = list(leader_records)

        summaries[dataset] = summary_by_flavor
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

    for dataset in DATASET_ORDER:
        if dataset_metrics[dataset]["comparable"]:
            featured_sample_ids = [record["sample_id"] for record in featured_by_dataset[dataset]]
            dataset_publish_pairs = select_gallery_pairs(pairs_by_dataset[dataset], featured_sample_ids)
            published_pairs.extend(dataset_publish_pairs)
            published_pair_counts[dataset] = len(dataset_publish_pairs)

            for pair in dataset_publish_pairs:
                for flavor in FLAVOR_ORDER:
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
                DATASET_ORDER.index(record["dataset"]),
                FLAVOR_ORDER.index(record["flavor"]),
                record["sample_numeric"],
                record["sample_id"],
            ),
        )
    ]
    published_pairs = sorted(
        published_pairs,
        key=lambda pair: (DATASET_ORDER.index(pair["dataset"]), pair["sample_numeric"], pair["sample_id"]),
    )
    comparable_datasets = [dataset for dataset in DATASET_ORDER if dataset_metrics[dataset]["comparable"]]
    single_flavor_datasets = [dataset for dataset in DATASET_ORDER if not dataset_metrics[dataset]["comparable"]]
    datasets_led = Counter(dataset_metrics[dataset]["leader_flavor"] for dataset in comparable_datasets)

    summary_payload = {
        "paired_samples": len(all_pairs_payload),
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
            comparable_datasets, key=lambda dataset: dataset_metrics[dataset]["delta_mIoU_flip_vs_coarse"]
        ),
        "best_gain_dataset_ARI": max(
            comparable_datasets, key=lambda dataset: dataset_metrics[dataset]["delta_ARI_flip_vs_coarse"]
        ),
        "best_gain_mIoU": max(
            dataset_metrics[dataset]["delta_mIoU_flip_vs_coarse"] for dataset in comparable_datasets
        ),
        "best_gain_ARI": max(dataset_metrics[dataset]["delta_ARI_flip_vs_coarse"] for dataset in comparable_datasets),
    }

    metrics_payload = {
        "metric_contract": METRIC_CONTRACT,
        "metric_note": METRIC_NOTE,
        "dataset_order": DATASET_ORDER,
        "flavor_order": FLAVOR_ORDER,
        "datasets": dataset_metrics,
        "summary": summary_payload,
    }

    training_payload = {
        "metric_contract": METRIC_CONTRACT,
        "metric_note": METRIC_NOTE,
        "dataset_order": DATASET_ORDER,
        "flavor_order": FLAVOR_ORDER,
        "datasets": {
            dataset: {
                "label": DATASETS[dataset]["label"],
                "dataset_id": DATASETS[dataset]["dataset_id"],
                "description": DATASETS[dataset]["description"],
                "available_flavors": dataset_metrics[dataset]["available_flavors"],
                "has_paired_comparison": dataset_metrics[dataset]["comparable"],
                "single_flavor_note": dataset_metrics[dataset]["single_flavor_note"],
                "published_record_count": published_record_counts[dataset],
                "available_record_count": available_record_counts[dataset],
                "published_pair_count": published_pair_counts[dataset],
                "available_pair_count": available_pair_counts[dataset],
            }
            for dataset in DATASET_ORDER
        },
        "flavors": {flavor: FLAVORS[flavor] for flavor in FLAVOR_ORDER},
        "counts": {
            "records": len(public_records),
            "paired_samples": len(published_pairs),
            "available_paired_samples": len(all_pairs_payload),
            "single_flavor_records": published_single_records,
            "available_single_flavor_records": sum(
                available_record_counts[dataset] for dataset in single_flavor_datasets
            ),
        },
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
        build_manifest(metrics_payload, public_records, pairs_payload),
        encoding="utf-8",
    )
    (DEST_DIR / "links.json").write_text(json.dumps(links_payload, indent=2), encoding="utf-8")

    print(
        f"Built cross-dataset overview with {len(public_records)} flavor-specific previews and "
        f"{len(pairs_payload)} paired comparisons."
    )


if __name__ == "__main__":
    main()
