import os
import json
import shutil
from pathlib import Path
import pandas as pd

base_dir = Path('/home/nada/PycharmProjects/research-site')
dest_dir = base_dir / 'site/experiments/sam3-cross-dataset-overview'

# Source directories - updated to unified Coarse Feature Clustering structure
rwtd_source = base_dir / 'experiments/Coarse Feature Clustering/rwtd'
caid_source = base_dir / 'experiments/Coarse Feature Clustering/caid'
stld_source = base_dir / 'experiments/Coarse Feature Clustering/stld'

# 1. Gather Metrics
with open(rwtd_source / 'summary.json', 'r') as f:
    rwtd_summ = json.load(f)
with open(caid_source / 'summary.json', 'r') as f:
    caid_summ = json.load(f)
with open(stld_source / 'summary.json', 'r') as f:
    stld_summ = json.load(f)

# Extract metrics for all three datasets
rwtd_metrics = rwtd_summ['mean_metrics']
caid_metrics = caid_summ['mean_metrics']
stld_metrics = stld_summ['mean_metrics']

metrics = {
    "rwtd": {
        "mIoU": round(rwtd_metrics['miou'], 3),
        "ARI": round(rwtd_metrics['ari'], 3)
    },
    "caid": {
        "mIoU": round(caid_metrics['miou'], 3),
        "ARI": round(caid_metrics['ari'], 3)
    },
    "stld": {
        "mIoU": round(stld_metrics['miou'], 3),
        "ARI": round(stld_metrics['ari'], 3)
    }
}

with open(dest_dir / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# 2. Process Visuals & training_data.json
training_data = {
    "learning_curves": {},
    "per_image": []
}

def process_dataset(name, source_path):
    manifest_path = source_path / 'visuals_manifest.jsonl'
    vis_dir = source_path / 'visuals'
    
    if not manifest_path.exists():
        return
        
    # Ensure asset directory exists
    (dest_dir / 'assets/all_previews' / name).mkdir(parents=True, exist_ok=True)
    (dest_dir / 'assets/gallery' / name).mkdir(parents=True, exist_ok=True)
        
    with open(manifest_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            file_name_orig = data['visual_path'].split('/')[-1]
            # Unique ID prevents collisions between datasets
            img_id = f"{name}_{file_name_orig.replace('.png', '')}"
            numeric_id = "".join(filter(str.isdigit, file_name_orig.replace('.png', '')))
            
            # Metric mapping
            f1 = data.get('miou', data.get('sample_miou', 0))
            ari = data.get('ari', data.get('sample_ari', 0))
            
            rel_path = f"{name}/{file_name_orig}"
            training_data["per_image"].append({
                "id": img_id,
                "sample_id": int(numeric_id) if numeric_id else 0,
                "dataset": name,
                "mIoU": round(f1, 4),
                "ARI": round(ari, 4),
                "file_path": rel_path
            })
            
            # Create relative symlink to source visual
            src_file = vis_dir / file_name_orig
            dst_file = dest_dir / 'assets/all_previews' / rel_path
            if dst_file.exists() or dst_file.is_symlink():
                dst_file.unlink()
            
            # Use relative path for symlink
            rel_src = os.path.relpath(src_file, dst_file.parent)
            os.symlink(rel_src, dst_file)

# Process all three datasets
process_dataset('rwtd', rwtd_source)
process_dataset('caid', caid_source)
process_dataset('stld', stld_source)

# Sort and pick top samples for summary galleries
all_per_image = training_data["per_image"]
for ds in ['rwtd', 'caid', 'stld']:
    ds_samples = [s for s in all_per_image if s['dataset'] == ds]
    ds_samples.sort(key=lambda x: x['mIoU'], reverse=True)
    # Pick top 8 for main view
    for s in ds_samples[:8]:
        dst_gallery = dest_dir / 'assets/gallery' / s['file_path']
        if dst_gallery.exists() or dst_gallery.is_symlink():
            dst_gallery.unlink()
        
        rel_src = os.path.relpath(dest_dir / 'assets/all_previews' / s['file_path'], dst_gallery.parent)
        os.symlink(rel_src, dst_gallery)

with open(dest_dir / 'training_data.json', 'w') as f:
    json.dump(training_data, f, indent=2)

# 3. Manifest and Summary
manifest = f"""title: "SAM 3 Cross-Dataset Benchmarking"
model_id: "facebook/sam3"
datasets: ["aviadcohz/RWTD", "architexture:caid", "architexture:stld"]
date: "{pd.Timestamp.now().strftime('%Y-%m-%d')}"
description: "Unified evaluation of SAM 3 Coarse Feature Clustering performance across RWTD, CAID, and STLD datasets."
asset_summary:
  rwtd_samples: {len([s for s in all_per_image if s['dataset'] == 'rwtd'])}
  caid_samples: {len([s for s in all_per_image if s['dataset'] == 'caid'])}
  stld_samples: {len([s for s in all_per_image if s['dataset'] == 'stld'])}
evaluation:
  metrics: ["mIoU", "ARI"]
  metrics_file: "metrics.json"
"""
with open(dest_dir / 'manifest.yaml', 'w') as f:
    f.write(manifest)

summary = "Comprehensive cross-dataset benchmark comparing SAM 3 Coarse Feature Clustering on RWTD (texture transitions), CAID (architectural plans), and STLD (structural segmentation). Unified metrics show consistent performance patterns across domains."
with open(dest_dir / 'summary.md', 'w') as f:
    f.write(summary)

with open(dest_dir / 'links.json', 'w') as f:
    json.dump({}, f)

print(f"Built Cross-Dataset report with {len(all_per_image)} samples.")
