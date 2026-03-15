import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
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
    protocol = img.get('protocol')
    sample_id = img.get('sample_id')
    
    if not protocol or not sample_id:
        continue
    
    if sample_id not in samples:
        samples[sample_id] = {}
    samples[sample_id][protocol] = img['f1']

# Load baseline from CSV
baseline_path = Path('/home/nada/PycharmProjects/research-site/experiments/full_all/dense/per_sample_metrics.csv')
baseline_df = pd.read_csv(baseline_path)
for _, row in baseline_df.iterrows():
    crop_str = str(row['crop_name']).replace('.jpg', '').replace('.png', '')
    try:
        sample_id = int(crop_str)
        if sample_id in samples:
            samples[sample_id]['baseline'] = row['miou_agg']
    except ValueError:
        pass

# Filter only those that have the core protocols + clustering
# Baseline might be in 'samples' already from the per_image loop if it was in training_data.json
# (Actually training_data.json has baseline now too)

# Recalculate lists for plotting
oracle_f1 = [v['oracle_points'] for v in samples.values() if 'oracle_points' in v]
text_f1 = [v['text'] for v in samples.values() if 'text' in v]
baseline_f1 = [v['baseline'] for v in samples.values() if 'baseline' in v]
fc_f1 = [v['feature_cluster_global'] for v in samples.values() if 'feature_cluster_global' in v]

sns.set_theme(style="whitegrid")

# 1. Histogram / KDE Plot
plt.figure(figsize=(8, 5))
sns.histplot(baseline_f1, color='green', label='Unprompted Baseline', kde=True, stat="density", linewidth=0, alpha=0.3)
sns.histplot(text_f1, color='orange', label='Text Descriptions', kde=True, stat="density", linewidth=0, alpha=0.5)
sns.histplot(oracle_f1, color='blue', label='Oracle Points', kde=True, stat="density", linewidth=0, alpha=0.5)
sns.histplot(fc_f1, color='purple', label='Feature Clustering', kde=True, stat="density", linewidth=0, alpha=0.5)
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
lims = [
    np.min([plt.xlim(), plt.ylim()]),  
    np.max([plt.xlim(), plt.ylim()]), 
]
plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='y = x (Equal Performance)')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('Sample-wise mIoU Comparison (Text vs Oracle)')
plt.xlabel('Text Descriptions mIoU')
plt.ylabel('Oracle Points mIoU')
plt.legend()
plt.tight_layout()
plt.savefig(plots_dir / 'miou_scatter.png', dpi=300)
plt.close()

# 3. Boxplot Comparison
plt.figure(figsize=(10, 6))
plot_data = pd.DataFrame({
    'Protocol': (['1. Baseline']*len(baseline_f1) + 
                 ['2. Text']*len(text_f1) + 
                 ['3. Oracle']*len(oracle_f1) + 
                 ['4. Feature Clustering']*len(fc_f1)),
    'mIoU': baseline_f1 + text_f1 + oracle_f1 + fc_f1
})
sns.boxplot(x='Protocol', y='mIoU', data=plot_data, palette=['#2ca02c', '#ff7f0e', '#1f77b4', '#9467bd'], hue='Protocol', legend=False)
plt.title('mIoU Performance Spread by Protocol')
plt.ylabel('mIoU')
plt.tight_layout()
plt.savefig(plots_dir / 'miou_boxplot.png', dpi=300)
plt.close()

# 4. Delta / Improvement Plot
plt.figure(figsize=(8, 5))
delta_text = np.array(text_f1) - np.array(baseline_f1)
delta_oracle = np.array(oracle_f1) - np.array(baseline_f1)

sns.histplot(delta_text, color='orange', label='Text vs Baseline', kde=True, stat="density", linewidth=0, alpha=0.5)
sns.histplot(delta_oracle, color='blue', label='Oracle vs Baseline', kde=True, stat="density", linewidth=0, alpha=0.5)
plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='No Change')
plt.title('Absolute mIoU Improvement over Unprompted Baseline')
plt.xlabel('Δ mIoU (Prompted - Baseline)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig(plots_dir / 'miou_delta.png', dpi=300)
plt.close()

print(f"Generated plots for {len(samples)} samples.")
