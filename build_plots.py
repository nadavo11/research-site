import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

base_dir = Path('/home/nada/PycharmProjects/research-site/site/experiments/sam3-rwtd-both-test-visuals')
plots_dir = base_dir / 'assets/plots'
os.makedirs(plots_dir, exist_ok=True)

with open(base_dir / 'training_data.json') as f:
    data = json.load(f)

# Group by sample ID
samples = {}
for img in data['per_image']:
    parts = img['id'].split('_') # e.g., oracle_points_123 or text_123
    # Check if 'oracle_points' is in the id
    if 'oracle' in img['id']:
        mode = 'oracle_points'
        sample_id = int(parts[-1])
    elif 'text' in img['id']:
        mode = 'text'
        sample_id = int(parts[-1])
    else:
        continue
    
    if sample_id not in samples:
        samples[sample_id] = {}
    samples[sample_id][mode] = img['f1']

# Filter only those that have both
valid_samples = {k: v for k, v in samples.items() if 'oracle_points' in v and 'text' in v}

oracle_f1 = [v['oracle_points'] for v in valid_samples.values()]
text_f1 = [v['text'] for v in valid_samples.values()]

sns.set_theme(style="whitegrid")

# 1. Histogram / KDE Plot
plt.figure(figsize=(8, 5))
sns.histplot(oracle_f1, color='blue', label='Oracle Points', kde=True, stat="density", linewidth=0, alpha=0.5)
sns.histplot(text_f1, color='orange', label='Text Descriptions', kde=True, stat="density", linewidth=0, alpha=0.5)
plt.title('Distribution of mIoU (F1) Scores by Protocol')
plt.xlabel('mIoU')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig(plots_dir / 'miou_distribution.png', dpi=300)
plt.close()

# 2. Scatter Plot
plt.figure(figsize=(6, 6))
sns.scatterplot(x=text_f1, y=oracle_f1, alpha=0.6, color='#005f73')
# Draw y=x line
lims = [
    np.min([plt.xlim(), plt.ylim()]),  
    np.max([plt.xlim(), plt.ylim()]), 
]
plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='y = x (Equal Performance)')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('Sample-wise mIoU Comparison')
plt.xlabel('Text Descriptions mIoU')
plt.ylabel('Oracle Points mIoU')
plt.legend()
plt.tight_layout()
plt.savefig(plots_dir / 'miou_scatter.png', dpi=300)
plt.close()

print(f"Generated plots for {len(valid_samples)} overlapping samples.")
