import os
import json
import shutil
from pathlib import Path

source_dir = Path('/home/nada/PycharmProjects/research-site/experiments/sam3-rwtd-both-test-visuals')
dest_dir = Path('/home/nada/PycharmProjects/research-site/site/experiments/sam3-rwtd-both-test-visuals')

# Create directories
os.makedirs(dest_dir, exist_ok=True)
os.makedirs(dest_dir / 'repro', exist_ok=True)
for sub in ['oracle_points', 'text', 'baseline']:
    os.makedirs(dest_dir / 'assets/gallery' / sub, exist_ok=True)
    os.makedirs(dest_dir / 'assets/qc' / sub, exist_ok=True)
    os.makedirs(dest_dir / 'assets/all_previews' / sub, exist_ok=True)

# 1. Coping config and terms
shutil.copy(source_dir / 'config.json', dest_dir / 'repro/config.json')
shutil.copy(source_dir / 'experiment_terms.md', dest_dir / 'repro/experiment_terms.md')

# 2. Extract metrics to metrics.json
with open(source_dir / 'summary.json', 'r') as f:
    summ = json.load(f)

op = summ['protocols']['oracle_points']['mean_metrics']
metrics = {
    "mIoU": round(op['sample_miou'], 3),
    "ARI": round(op['sample_ari'], 3),
    "Boundary_Dice": round(op['boundary_dice'], 3),
    "Macro_Dice": round(op['sample_macro_dice'], 3)
}

# Also extract unprompted baseline metrics
with open('/home/nada/PycharmProjects/research-site/experiments/full_all/summary.json', 'r') as f:
    summ_baseline = json.load(f)
dense_baseline = summ_baseline['variants']['dense']['mean_metrics']
metrics["Baseline_mIoU"] = round(dense_baseline['miou_agg'], 3)
metrics["Baseline_ARI"] = round(dense_baseline['ari'], 3)

with open(dest_dir / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# 3. Create manifest.yaml
manifest = f"""title: "SAM 3 RWTD Both Protocols"
model_id: "facebook/sam3"
dataset_id: "aviadcohz/RWTD"
split: "test"
date: "2026-03-12"
description: "Evaluation of SAM 3 on RWTD test split using both text and oracle points protocols."
assets:
  gallery_count: 8
  qc_count: 4
  all_previews_count: 454
evaluation:
  primary_metric: "mIoU"
  metrics_file: "metrics.json"
"""
with open(dest_dir / 'manifest.yaml', 'w') as f:
    f.write(manifest)

# 4. Create summary.md
summary = f"This experiment evaluates `facebook/sam3` on the RWTD test set using two prompting protocols: `text` (natural language descriptions) and `oracle_points` (ground-truth sampled points). The page focuses on the `oracle_points` protocol for the main visual gallery. The fully unprompted baseline (`full_all/dense`) achieves a baseline mIoU of {metrics['Baseline_mIoU']}."
with open(dest_dir / 'summary.md', 'w') as f:
    f.write(summary)

with open(dest_dir / 'links.json', 'w') as f:
    json.dump({}, f)

# 5. Process visuals and build training_data.json
import pandas as pd
training_data = {
    "learning_curves": {
        "steps": [],
        "train_loss": [],
        "val_ap": [],
        "rwtd_ap": [],
        "ods_f1": []
    },
    "per_image": []
}

all_samples = []

# Load baseline CSV for baseline metrics
baseline_csv = Path('/home/nada/PycharmProjects/research-site/experiments/full_all/dense/per_sample_metrics.csv')
baseline_df = pd.read_csv(baseline_csv)
baseline_metrics_map = {}
for _, row in baseline_df.iterrows():
    c_name = str(row['crop_name']).replace('.jpg', '').replace('.png', '')
    baseline_metrics_map[c_name + '.png'] = row['miou_agg']

protocols_config = [
    ('oracle_points', source_dir / 'oracle_points'),
    ('text', source_dir / 'text'),
    ('baseline', Path('/home/nada/PycharmProjects/research-site/experiments/test_dense'))
]

for protocol, p_dir in protocols_config:
    manifest_path = p_dir / 'visuals_manifest.jsonl'
    vis_dir = p_dir / 'visuals'
    
    if protocol != 'baseline':
        if not manifest_path.exists(): continue
        with open(manifest_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                file_name_orig = data['visual_path'].split('/')[-1]
                img_id = f"{protocol}_{file_name_orig.replace('.png', '')}"
                
                f1 = data['sample_macro_dice']
                precision = data['sample_macro_precision']
                recall = data['sample_macro_recall']
                threshold = data['boundary_dice']
                
                rel_path = f"{protocol}/{file_name_orig}"
                training_data["per_image"].append({
                    "id": img_id,
                    "f1": round(f1, 4),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "threshold": round(threshold, 4),
                    "file_path": rel_path
                })
                all_samples.append({
                    "name": file_name_orig,
                    "protocol": protocol,
                    "f1": f1,
                    "src_path": vis_dir / file_name_orig,
                    "dest_rel": rel_path
                })
                # Copy to all_previews/{protocol}
                shutil.copy(vis_dir / file_name_orig, dest_dir / 'assets/all_previews' / protocol / file_name_orig)
    else:
        # Baseline uses the loose visuals from test_dense and CSV for metrics
        if not vis_dir.exists(): continue
        for img_path in vis_dir.glob('*.png'):
            file_name_orig = img_path.name
            img_id = f"{protocol}_{file_name_orig.replace('.png', '')}"
            
            f1 = baseline_metrics_map.get(file_name_orig, 0)
            rel_path = f"{protocol}/{file_name_orig}"
            training_data["per_image"].append({
                "id": img_id,
                "f1": round(f1, 4),
                "precision": 0,
                "recall": 0,
                "threshold": 0,
                "file_path": rel_path
            })
            all_samples.append({
                "name": file_name_orig,
                "protocol": protocol,
                "f1": f1,
                "src_path": img_path,
                "dest_rel": rel_path
            })
            shutil.copy(img_path, dest_dir / 'assets/all_previews' / protocol / file_name_orig)

# Sort by F1 to pick top 8 for gallery and bottom 4 for QC
all_samples.sort(key=lambda x: x['f1'], reverse=True)

# Pick top 8 representing oracle_points if possible
gallery_samples = [s for s in all_samples if s['protocol'] == 'oracle_points'][:8]
qc_samples = [s for s in all_samples if s['protocol'] == 'oracle_points' and s['f1'] < 0.2][:4]
if len(qc_samples) < 4:
    qc_samples = [s for s in all_samples if s['protocol'] == 'oracle_points'][-4:]

for s in gallery_samples:
    shutil.copy(s['src_path'], dest_dir / 'assets/gallery' / s['protocol'] / s['name'])
    
for s in qc_samples:
    shutil.copy(s['src_path'], dest_dir / 'assets/qc' / s['protocol'] / s['name'])

dest_json = dest_dir / 'training_data.json'
with open(dest_json, 'w') as f:
    json.dump(training_data, f, indent=2)

print(f"Built SAM 3 page structure with {len(all_samples)} preview images.")
print(f"Gallery selection: {[s['name'] for s in gallery_samples]}")
print(f"QC selection: {[s['name'] for s in qc_samples]}")
