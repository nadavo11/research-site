#!/usr/bin/env python3
"""Build the SAM 3 cross-dataset overview from normalized CFC experiment roots."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from datetime import date
from html import escape
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
LEGACY_SOURCE_ROOT = ROOT / "experiments" / "Coarse Feature Clustering"
DEST_DIR = ROOT / "site" / "experiments" / "sam3-cross-dataset-overview"

DATASET_ORDER = ["rwtd", "caid", "stld"]
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
    "featured": 4,
    "delta_gain": 4,
    "delta_loss": 4,
    "leader_score": 4,
}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def relative_symlink(src: Path, dst: Path, *, preserve_existing: bool = False) -> None:
    ensure_parent(dst)
    if preserve_existing and (dst.exists() or dst.is_symlink()):
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src, dst.parent))


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


def resolve_run_dir(dataset: str, flavor: str) -> Path:
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
            if flavor == "coarse_only":
                relative_symlink(
                    source_file,
                    DEST_DIR / "assets" / "all_previews" / asset_rel,
                    preserve_existing=True,
                )
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
    if record["flavor"] == "coarse_only":
        relative_symlink(record["_source_path"], dst, preserve_existing=True)
        return
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


def build_dataset_metrics(metrics_payload: dict, featured: dict[str, list[dict]]) -> str:
    rows = []
    for dataset in DATASET_ORDER:
        info = metrics_payload["datasets"][dataset]
        coarse = info["flavors"]["coarse_only"]
        flip = info["flavors"]["flip_averaged"]
        leader = FLAVORS[info["leader_flavor"]]["label"]
        delta_class = "good" if info["delta_mIoU_flip_vs_coarse"] > 0 else "bad" if info["delta_mIoU_flip_vs_coarse"] < 0 else ""
        rows.append(
            f"""
            <tr>
              <td>
                <strong>{escape(info["label"])}</strong>
                <div class="mini-note">{escape(DATASETS[dataset]["dataset_id"])}</div>
              </td>
              <td>
                <strong>{coarse["mIoU"]:.3f}</strong>
                <div class="mini-note">ARI {coarse["ARI"]:.3f}</div>
              </td>
              <td>
                <strong>{flip["mIoU"]:.3f}</strong>
                <div class="mini-note">ARI {flip["ARI"]:.3f}</div>
              </td>
              <td><span class="delta-pill {delta_class}">{signed(info["delta_mIoU_flip_vs_coarse"])}</span></td>
              <td><span class="delta-pill {delta_class}">{signed(info["delta_ARI_flip_vs_coarse"])}</span></td>
              <td><span class="tag {'good' if info["leader_flavor"] == 'flip_averaged' else 'warn'}">{escape(leader)}</span></td>
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
        if delta_miou >= 0:
            delta_copy = (
                f"Flip Averaged improves over Coarse Only by {signed(delta_miou)} mIoU and "
                f"{signed(delta_ari)} ARI on the shared evaluator."
            )
        else:
            delta_copy = (
                f"Flip Averaged trails Coarse Only by {abs(delta_miou):.3f} mIoU and "
                f"{abs(delta_ari):.3f} ARI here, so Coarse Only remains the leader."
            )
        previews = []
        for record in featured[dataset]:
            badge = "good" if record["mIoU"] >= 0.75 else "warn" if record["mIoU"] >= 0.5 else "bad"
            src = f"assets/all_previews/{record['file_path']}"
            previews.append(
                f"""
                <figure>
                  <img src="{escape(src)}" alt="{escape(info["label"])} sample {escape(record["sample_id"])}"
                    data-zoom-src="{escape(src)}" />
                  <figcaption>
                    <span class="pill">id {escape(record["sample_id"])}</span>
                    <span class="pill">{escape(FLAVORS[record["flavor"]]["label"])}</span>
                    <span class="tag {badge}">mIoU {record["mIoU"]:.3f}</span>
                    <span class="pill">ARI {record["ARI"]:.3f}</span>
                  </figcaption>
                </figure>
                """
            )
        win_counts = info["sample_wins"]
        cards.append(
            f"""
            <article class="dataset-card cfc-card">
              <div class="card-topline">
                <span class="tag">{escape(info["label"])}</span>
                <span class="tag {leader_tag}">Leader: {escape(leader_label)}</span>
              </div>
              <h3>{escape(info["title"])}</h3>
              <p>{escape(info["description"])}</p>
              <div class="metric-grid compact-grid">
                <article class="metric">
                  <div class="k">Leader mIoU</div>
                  <div class="v">{info["leader_mIoU"]:.3f}</div>
                </article>
                <article class="metric">
                  <div class="k">Leader ARI</div>
                  <div class="v">{info["leader_ARI"]:.3f}</div>
                </article>
                <article class="metric">
                  <div class="k">Flip Avg ΔmIoU</div>
                  <div class="v delta-copy {'delta-positive' if delta_miou > 0 else 'delta-negative' if delta_miou < 0 else ''}">{signed(delta_miou)}</div>
                </article>
                <article class="metric">
                  <div class="k">Flip Avg ΔARI</div>
                  <div class="v delta-copy {'delta-positive' if delta_ari > 0 else 'delta-negative' if delta_ari < 0 else ''}">{signed(delta_ari)}</div>
                </article>
              </div>
              <p class="kv card-note">{escape(delta_copy)}</p>
              <p class="kv card-note">
                Sample wins: Flip Averaged {win_counts["flip_averaged"]} | Coarse Only {win_counts["coarse_only"]} | Ties {win_counts["ties"]}.
              </p>
              <div class="preview-grid">
                {"".join(previews)}
              </div>
              <div class="card-actions">
                <a class="btn secondary" href="gallery.html?dataset={dataset}&view=paired">Open {escape(info["label"])} 1:1 Gallery</a>
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
              gap: 18px;
            }

            .card-topline {
              display: flex;
              flex-wrap: wrap;
              gap: 10px;
            }

            .card-note {
              margin: 0;
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
                Flavor-aware benchmarking for Coarse Feature Clustering across RWTD, CAID, and STLD, now including the
                new <code>flip_averaged</code> variant.
              </p>
            </header>

            <section class="executive-summary">
              <h2>Executive Synopsis</h2>
              <p>
                The new <strong>Flip Averaged</strong> flavor changes the cross-dataset picture materially. It takes the
                lead on <strong>RWTD</strong> and <strong>STLD</strong>, with the largest jump on STLD, while the
                original <strong>Coarse Only</strong> flavor still leads CAID.
              </p>
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
                <code>flip_averaged - coarse_only</code>.
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
                  <p class="kv">Each card now calls out the current leader flavor and its headline metrics.</p>
                </div>
                <a class="btn" href="gallery.html">Open Cross-Flavor Gallery</a>
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
        .replace("__HEADLINE_CARDS__", build_headline_cards(metrics_payload))
        .replace("__TABLE_ROWS__", "".join(rows))
        .replace("__DATASET_CARDS__", "".join(cards))
    )


def build_headline_cards(metrics_payload: dict) -> str:
    summary = metrics_payload["summary"]
    leader_label = FLAVORS[summary["datasets_led_by"]]["label"]
    best_gain_dataset_miou = metrics_payload["datasets"][summary["best_gain_dataset_mIoU"]]["label"]
    best_gain_dataset_ari = metrics_payload["datasets"][summary["best_gain_dataset_ARI"]]["label"]
    return (
        f"""
        <article class="metric">
          <div class="k">Paired Samples</div>
          <div class="v">{summary["paired_samples"]:,}</div>
        </article>
        <article class="metric">
          <div class="k">Flavor Lead Split</div>
          <div class="v">{summary["datasets_led"]['flip_averaged']}/{len(DATASET_ORDER)}</div>
          <div class="mini-note">{escape(leader_label)} leads more datasets.</div>
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
              <h1 class="title-gradient">Cross-Flavor 1:1 Comparison</h1>
              <p style="color: var(--muted); margin-top: 16px; font-weight: 300; max-width: 860px; margin-inline: auto;">
                Browse a curated publication slice of the same sample across flavors, or isolate a single flavor per
                dataset. The headline metrics on the overview page still reflect the full benchmark.
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
              let visibleCount = 16;

              const initialDataset = params.get('dataset');
              if (initialDataset && [...datasetSelect.options].some((option) => option.value === initialDataset)) {
                datasetSelect.value = initialDataset;
              } else {
                datasetSelect.value = 'rwtd';
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

              function resetVisibleCount() {
                visibleCount = currentPageSize();
              }

              const initialShown = Number(params.get('shown'));
              if (Number.isFinite(initialShown) && initialShown > 0) {
                visibleCount = initialShown;
              } else {
                resetVisibleCount();
              }

              function syncUrl() {
                const next = new URLSearchParams(window.location.search);
                next.set('dataset', datasetSelect.value);
                next.set('view', flavorSelect.value);
                next.set('sort', sortSelect.value);
                next.set('page_size', pageSizeSelect.value);
                if (Number.isFinite(visibleCount)) {
                  next.set('shown', String(visibleCount));
                } else {
                  next.delete('shown');
                }
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

              function visibleSlice(items) {
                if (!Number.isFinite(visibleCount)) return items;
                return items.slice(0, Math.min(visibleCount, items.length));
              }

              function updateLoadMore(items, noun) {
                if (!Number.isFinite(visibleCount) || visibleCount >= items.length) {
                  galleryControls.hidden = true;
                  return;
                }

                galleryControls.hidden = false;
                const remaining = items.length - visibleCount;
                const increment = currentPageSize();
                const nextCount = Number.isFinite(increment) ? Math.min(increment, remaining) : remaining;
                loadMoreBtn.textContent = `Load ${nextCount} More ${noun}`;
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
                        <span class="pill">Leader ${record.leader_flavor_for_sample === 'tie' ? 'Tie' : data.flavors[record.leader_flavor_for_sample].label}</span>
                      </div>
                    </figure>
                  </article>
                `;
              }

              function render() {
                const dataset = datasetSelect.value;
                const view = flavorSelect.value;
                const sort = sortSelect.value;

                if (view === 'paired') {
                  const filteredPairs = data.pairs.filter((pair) => dataset === 'all' || pair.dataset === dataset);
                  const items = sortItems(filteredPairs, sort);
                  const visibleItems = visibleSlice(items);
                  root.className = 'gallery-stack';
                  root.innerHTML = visibleItems.map(renderPairCard).join('');
                  countLabel.textContent = `Showing ${visibleItems.length} of ${items.length} paired samples`;
                  toolbarNote.textContent = dataset === 'all'
                    ? 'This is the curated publication subset across all datasets. Narrow to one dataset for faster inspection.'
                    : 'Paired mode keeps Coarse Only on the left and Flip Averaged on the right for the same curated sample.';
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
                root.className = 'single-grid';
                root.innerHTML = visibleItems.map(renderSingleCard).join('');
                countLabel.textContent = `Showing ${visibleItems.length} of ${items.length} ${data.flavors[view].label.toLowerCase()} samples`;
                toolbarNote.textContent = 'Single-flavor mode uses the same curated publication subset as the paired gallery.';
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
                resetVisibleCount();
                render();
              }

              datasetSelect.addEventListener('change', rerenderFromFirstPage);
              flavorSelect.addEventListener('change', rerenderFromFirstPage);
              sortSelect.addEventListener('change', rerenderFromFirstPage);
              pageSizeSelect.addEventListener('change', rerenderFromFirstPage);
              loadMoreBtn.addEventListener('click', () => {
                if (Number.isFinite(visibleCount)) {
                  visibleCount += currentPageSize();
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
        dataset_lines.append(
            f"""  - slug: "{dataset}"
    id: "{info['dataset_id']}"
    samples: {info['sample_count']}
    leader_flavor: "{info['leader_flavor']}"
"""
        )
    return (
        "title: \"SAM 3 Cross-Dataset Benchmarking\"\n"
        "model_id: \"facebook/sam3\"\n"
        "ui_standard: \"premium\"\n"
        f"date: \"{date.today().isoformat()}\"\n"
        f"comparison_contract: \"{METRIC_CONTRACT}\"\n"
        "description: \"Flavor-aware benchmarking for SAM 3 Coarse Feature Clustering across RWTD, CAID, and STLD.\"\n"
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
    return dedent(
        f"""\
        This page compares the `coarse_only` and `flip_averaged` Coarse Feature Clustering flavors under the shared
        `{METRIC_CONTRACT}` evaluator (`mIoU`, `ARI`).

        `flip_averaged` now leads RWTD (`{info['rwtd']['leader_mIoU']:.3f}` mIoU, `{info['rwtd']['leader_ARI']:.3f}` ARI)
        and STLD (`{info['stld']['leader_mIoU']:.3f}` mIoU, `{info['stld']['leader_ARI']:.3f}` ARI), with the largest
        gain on STLD (`{signed(info['stld']['delta_mIoU_flip_vs_coarse'])}` mIoU, `{signed(info['stld']['delta_ARI_flip_vs_coarse'])}` ARI).
        CAID still favors `coarse_only` (`{info['caid']['leader_mIoU']:.3f}` mIoU, `{info['caid']['leader_ARI']:.3f}` ARI).
        """
    ).strip()


def build_report() -> tuple[dict, dict, str, str, str, dict]:
    (DEST_DIR / "assets" / "all_previews").mkdir(parents=True, exist_ok=True)
    shutil.rmtree(DEST_DIR / "assets" / "all_previews" / "flip_averaged", ignore_errors=True)

    summaries: dict[str, dict] = {}
    all_records: list[dict] = []
    featured_by_dataset: dict[str, list[dict]] = {}
    all_pairs_payload: list[dict] = []
    pairs_by_dataset: dict[str, list[dict]] = {}

    dataset_metrics: dict[str, dict] = {}
    links_payload = {"source_runs": {}, "metric_contract": METRIC_CONTRACT, "metric_note": METRIC_NOTE}

    for dataset in DATASET_ORDER:
        links_payload["source_runs"][dataset] = {}
        summary_by_flavor: dict[str, dict] = {}
        records_by_flavor: dict[str, dict[str, dict]] = {}
        dataset_pairs: list[dict] = []

        for flavor in FLAVOR_ORDER:
            run_dir = resolve_run_dir(dataset, flavor)
            summary = load_summary(run_dir)
            summary_by_flavor[flavor] = summary
            records_by_flavor[flavor] = load_visual_records(dataset, flavor, run_dir)
            links_payload["source_runs"][dataset][flavor] = summary["source_path"]

        common_sample_ids = sorted(
            set(records_by_flavor["coarse_only"]) & set(records_by_flavor["flip_averaged"]),
            key=lambda sample_id: (parse_sample_numeric(sample_id), sample_id),
        )
        sample_wins = Counter()

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
        dataset_metrics[dataset] = {
            "label": DATASETS[dataset]["label"],
            "title": DATASETS[dataset]["title"],
            "description": DATASETS[dataset]["description"],
            "dataset_id": DATASETS[dataset]["dataset_id"],
            "sample_count": summary_by_flavor["coarse_only"]["sample_count"],
            "leader_flavor": leader,
            "leader_mIoU": round(summary_by_flavor[leader]["mIoU"], 6),
            "leader_ARI": round(summary_by_flavor[leader]["ARI"], 6),
            "delta_mIoU_flip_vs_coarse": delta_miou,
            "delta_ARI_flip_vs_coarse": delta_ari,
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
        all_records.extend(records_by_flavor["coarse_only"].values())
        all_records.extend(records_by_flavor["flip_averaged"].values())
        dataset_metrics[dataset]["gallery_preview_count"] = len(featured_by_dataset[dataset])

    record_lookup = {record["id"]: record for record in all_records}
    published_pairs: list[dict] = []
    published_records: list[dict] = []
    published_record_ids: set[str] = set()

    for dataset in DATASET_ORDER:
        featured_sample_ids = [record["sample_id"] for record in featured_by_dataset[dataset]]
        dataset_publish_pairs = select_gallery_pairs(pairs_by_dataset[dataset], featured_sample_ids)
        published_pairs.extend(dataset_publish_pairs)

        for pair in dataset_publish_pairs:
            for flavor in FLAVOR_ORDER:
                record = record_lookup[pair["record_ids"][flavor]]
                if record["id"] in published_record_ids:
                    continue
                materialize_web_asset(record)
                published_records.append(record)
                published_record_ids.add(record["id"])

        for record in featured_by_dataset[dataset]:
            if record["id"] in published_record_ids:
                continue
            materialize_web_asset(record)
            published_records.append(record)
            published_record_ids.add(record["id"])

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

    summary_payload = {
        "paired_samples": len(all_pairs_payload),
        "datasets_led": dict(Counter(dataset_metrics[dataset]["leader_flavor"] for dataset in DATASET_ORDER)),
        "datasets_led_by": max(
            FLAVOR_ORDER,
            key=lambda flavor: sum(dataset_metrics[dataset]["leader_flavor"] == flavor for dataset in DATASET_ORDER),
        ),
        "best_gain_dataset_mIoU": max(
            DATASET_ORDER, key=lambda dataset: dataset_metrics[dataset]["delta_mIoU_flip_vs_coarse"]
        ),
        "best_gain_dataset_ARI": max(
            DATASET_ORDER, key=lambda dataset: dataset_metrics[dataset]["delta_ARI_flip_vs_coarse"]
        ),
        "best_gain_mIoU": max(dataset_metrics[dataset]["delta_mIoU_flip_vs_coarse"] for dataset in DATASET_ORDER),
        "best_gain_ARI": max(dataset_metrics[dataset]["delta_ARI_flip_vs_coarse"] for dataset in DATASET_ORDER),
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
            }
            for dataset in DATASET_ORDER
        },
        "flavors": {flavor: FLAVORS[flavor] for flavor in FLAVOR_ORDER},
        "counts": {
            "records": len(public_records),
            "paired_samples": len(published_pairs),
            "available_paired_samples": len(all_pairs_payload),
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
