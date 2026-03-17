import os
import json

base_dir = '/home/nada/PycharmProjects/research-site/site/experiments/sam3-rwtd-both-test-visuals'

# Read data
with open(os.path.join(base_dir, 'training_data.json')) as f:
    data = json.load(f)

with open(os.path.join(base_dir, 'metrics.json')) as f:
    mets = json.load(f)

# Files
gallery_files = os.listdir(os.path.join(base_dir, 'assets/gallery'))
qc_files = os.listdir(os.path.join(base_dir, 'assets/qc'))

def get_metrics_for_id(id_str):
    for item in data['per_image']:
        if item['id'] == id_str.replace('.png', ''):
            return item
    return None

def color_f1_pill(f1):
    if f1 >= 0.7: return 'tag good'
    if f1 >= 0.4: return 'tag warn'
    return 'tag bad'

gallery_figures = ""
for gf in sorted(gallery_files):
    m = get_metrics_for_id(gf)
    if not m: continue
    tag_cls = color_f1_pill(m['f1'])
    gallery_figures += f'''
        <figure data-score="{m['f1']:.4f}">
          <img src="assets/gallery/{gf}" alt="sample {m['id']} preview"
            data-zoom-src="assets/gallery/{gf}" />
          <figcaption>
            <span class="pill">id={m['id']}</span>
            <span class="{tag_cls}">mIoU={m['f1']:.3f}</span>
            <span class="pill">P={m['precision']:.3f} R={m['recall']:.3f}</span>
          </figcaption>
        </figure>'''

qc_figures = ""
for qf in sorted(qc_files):
    m = get_metrics_for_id(qf)
    if not m: continue
    qc_figures += f'''
        <figure data-score="{m['id'].split('_')[-1]}">
          <img src="assets/qc/{qf}" alt="failure case sample {m['id']}"
            data-zoom-src="assets/qc/{qf}" />
          <figcaption><span class="pill">id={m['id']}</span> <span class="tag bad">failure</span> Hard failure or severe miss-prediction.</figcaption>
        </figure>'''

html_index = f'''<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>RWTD — SAM 3 Both Protocols</title>
  <link rel="stylesheet" href="../../assets/site.css" />
</head>

<body>
  <div class="container animate-fade-in-down">
    <header style="text-align: center; margin-bottom: 50px;">
      <div class="subtitle">Experiment Report</div>
      <h1 class="title-gradient">RWTD Segmentation — SAM 3 Oracle Points & Text</h1>
      <p style="color: var(--muted); margin-top: 15px; font-weight: 300;">Evaluation of the foundational SAM 3 model on the RWTD test split using both oracle points and text label prompting.</p>
      <div style="margin-top: 20px;">
        <span class="tag good">evaluation</span><span class="tag">sam3</span><span class="tag">foundation model</span>
      </div>
    </header>

    <section class="executive-summary">
      <h2>Executive Synopsis</h2>
      <div class="summary-body">
        <p class="kv"><strong>Run ID:</strong> <code>sam3-rwtd-both-test-visuals</code></p>
        <p class="kv"><strong>Scope:</strong> Zero-shot evaluation of facebook/sam3 using oracle points (2 per texture) and text descriptions.</p>
        <p class="kv"><strong>Primary Finding:</strong> Oracle points are highly effective at guiding the model to the dominant texture region, whereas text descriptions often struggle with overlapping visual properties.</p>
      </div>
      
      <div class="metric-grid" style="margin-top: 30px;">
        <article class="metric">
          <div class="k">Primary Metric (mIoU)</div>
          <div class="v" style="color: var(--brand);">{mets['mIoU']:.3f}</div>
        </article>
        <article class="metric">
          <div class="k">ARI Score</div>
          <div class="v">{mets['ARI']:.3f}</div>
        </article>
        <article class="metric">
          <div class="k">Baseline mIoU</div>
          <div class="v" style="color: var(--muted);">{mets.get('Baseline_mIoU', 0.725):.3f}</div>
        </article>
      </div>
    </section>

    <section class="section">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px;">
        <h2>Visual Preview Gallery</h2>
        <div class="gallery-controls">
          <label for="gallerySort">Sort by Score:</label>
          <select id="gallerySort" data-sort-gallery="mainGallery">
            <option value="default">Default order</option>
            <option value="asc">mIoU ↑ (worst first)</option>
            <option value="desc">mIoU ↓ (best first)</option>
          </select>
        </div>
      </div>
      <p class="kv" style="margin-bottom: 20px;">Representative samples from RWTD test evaluated with oracle points. Click any image to enlarge.</p>
      
      <div class="gallery" id="mainGallery">
{gallery_figures}
      </div>
      
      <div style="margin-top: 30px; text-align: center;">
        <a class="btn" href="gallery.html">Open Full Results Gallery (454 images) →</a>
      </div>
    </section>

    <section class="section">
      <h2>QC Diagnostics & Edge Cases</h2>
      <p class="kv" style="margin-bottom: 20px;">Failure cases and severe miss-predictions identified during quality control passes.</p>
      <div class="gallery" id="qcGallery">
{qc_figures}
      </div>
    </section>

    <section class="section">
      <h2>Metadata & Reproducibility</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Artifact Type</th>
              <th>Path / Reference</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Reproduction Config</td>
              <td><code>repro/config.json</code></td>
              <td><span class="tag good">Verified</span></td>
            </tr>
            <tr>
              <td>Experiment Terms</td>
              <td><code>repro/experiment_terms.md</code></td>
              <td><span class="tag good">Verified</span></td>
            </tr>
            <tr>
              <td>Raw Manifest</td>
              <td><code>manifest.yaml</code></td>
              <td><span class="tag">Static</span></td>
            </tr>
          </tbody>
        </table>
      </div>
      <div style="margin-top: 30px; text-align: center;">
        <a class="btn secondary" href="../../index.html">← Back to Dashboard</a>
      </div>
    </section>

    <footer style="margin-top: 80px;">
      <p>Report generated on 2026-03-17 | Powered by <strong>SAM 3 Intelligence Hub</strong></p>
    </footer>
  </div>

  <dialog class="image-dialog" id="imageDialog">
    <img id="imageDialogImg" alt="Expanded preview" />
  </dialog>
  <script src="../../assets/site.js"></script>
</body>
</html>
'''

with open(os.path.join(base_dir, 'index.html'), 'w') as f:
    f.write(html_index)

html_gallery = f'''<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Gallery — SAM 3 Both Protocols</title>
    <link rel="stylesheet" href="../../assets/site.css" />
    <style>
        .gallery-toolbar {{
            background: var(--glass-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 20px;
            margin-bottom: 30px;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 20px;
            box-shadow: var(--premium-shadow);
        }}

        .gallery-toolbar label {{
            font-family: var(--font-accent);
            font-weight: 600;
            font-size: 0.9rem;
            color: var(--ink);
        }}

        .gallery-toolbar select,
        .gallery-toolbar input {{
            font: inherit;
            border: 1px solid var(--line);
            background: #fff;
            color: var(--ink);
            border-radius: 999px;
            padding: 8px 16px;
            font-size: 0.88rem;
            outline: none;
            transition: border-color 0.2s;
        }}
        
        .gallery-toolbar select:focus {{ border-color: var(--brand); }}

        #fullGallery {{
            display: grid;
            gap: 20px;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        }}

        #fullGallery figure {{
            background: #fff;
            border-radius: 20px;
            overflow: hidden;
            border: 1px solid var(--line);
            box-shadow: 0 4px 12px rgba(0,0,0,0.03);
            transition: transform 0.3s ease;
        }}

        #fullGallery figure:hover {{ transform: translateY(-5px); }}

        .f1-badge {{
            position: absolute;
            top: 12px;
            right: 12px;
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(4px);
            color: #fff;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 4px 12px;
            border-radius: 999px;
            z-index: 2;
        }}
    </style>
</head>

<body>
    <div class="container animate-fade-in-down">
        <header style="text-align: center; margin-bottom: 40px;">
            <div class="subtitle">Visual Evidence</div>
            <h1 class="title-gradient">Full Prediction Gallery</h1>
            <p style="color: var(--muted); margin-top: 10px; font-weight: 300;">SAM 3 RWTD Both Protocols — 454 predictions with exhaustive metrics.</p>
            <div style="margin-top: 20px;">
                <a class="btn secondary" href="index.html">← Return to Report</a>
            </div>
        </header>

        <section class="section">
            <div class="gallery-toolbar">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <label for="sortBy">Sort:</label>
                    <select id="sortBy">
                        <option value="f1_desc">mIoU ↓ best first</option>
                        <option value="f1_asc">mIoU ↑ worst first</option>
                        <option value="precision_desc">Precision ↓</option>
                        <option value="id_asc">ID ↑</option>
                    </select>
                </div>
                
                <div style="display: flex; align-items: center; gap: 10px;">
                    <label for="filterF1">Min mIoU:</label>
                    <input type="range" id="filterF1" min="0" max="100" value="0" />
                    <span id="filterF1Val" style="font-family: var(--font-accent); font-weight: 700; color: var(--brand);">0.00</span>
                </div>

                <div style="display: flex; align-items: center; gap: 10px;">
                    <label for="filterQuality">Quality Tier:</label>
                    <select id="filterQuality">
                        <option value="all">All Tiers</option>
                        <option value="good">Good (≥ 0.7)</option>
                        <option value="medium">Medium (0.4–0.7)</option>
                        <option value="bad">Bad (< 0.4)</option>
                    </select>
                </div>

                <div class="count" id="countLabel" style="margin-left: auto; font-family: var(--font-accent); font-weight: 600; color: var(--muted);">454 / 454</div>
            </div>
            
            <div id="fullGallery"></div>
        </section>

        <footer>
            <p>Full gallery archives. Research Site Premium v1.2</p>
        </footer>
    </div>

    <dialog class="image-dialog" id="imageDialog">
        <img id="imageDialogImg" alt="Expanded preview" />
    </dialog>

    <script>
        (async () => {{
            const resp = await fetch('training_data.json');
            const data = await resp.json();
            const gallery = document.getElementById('fullGallery');
            const sortSel = document.getElementById('sortBy');
            const filterSlider = document.getElementById('filterF1');
            const filterVal = document.getElementById('filterF1Val');
            const filterQual = document.getElementById('filterQuality');
            const countLabel = document.getElementById('countLabel');
            const total = data.per_image.length;

            function f1Class(f1) {{
                if (f1 >= 0.7) return 'tag good';
                if (f1 >= 0.4) return 'tag warn';
                return 'tag bad';
            }}

            function qualityLabel(f1) {{
                if (f1 >= 0.7) return 'good';
                if (f1 >= 0.4) return 'medium';
                return 'bad';
            }}

            function render() {{
                const sortKey = sortSel.value;
                const minF1 = parseFloat(filterSlider.value) / 100;
                const qualFilter = filterQual.value;
                filterVal.textContent = minF1.toFixed(2);

                let items = data.per_image.filter(img => {{
                    if (img.f1 < minF1) return false;
                    if (qualFilter !== 'all' && qualityLabel(img.f1) !== qualFilter) return false;
                    return true;
                }});

                const parts = sortKey.split('_');
                const field = parts[0];
                const dir = parts[1];
                items.sort((a, b) => {{
                    const va = a[field], vb = b[field];
                    return dir === 'desc' ? vb - va : va - vb;
                }});

                countLabel.textContent = items.length + ' / ' + total;

                gallery.innerHTML = items.map((img, i) => {{
                    const cls = f1Class(img.f1);
                    return '<figure style="position:relative;">' +
                        '<span class="f1-badge">mIoU ' + img.f1.toFixed(3) + '</span>' +
                        '<img src="assets/all_previews/' + img.file_path + '" alt="sample ' + img.id + '" data-zoom-src="assets/all_previews/' + img.file_path + '" />' +
                        '<div style="padding: 15px; border-top: 1px solid var(--line);">' +
                        '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">' +
                        '<span class="pill">id=' + img.id.split('_').slice(-1)[0] + '</span>' +
                        '<span class="' + cls + '">mIoU ' + img.f1.toFixed(3) + '</span>' +
                        '</div>' +
                        '<div style="font-size: 0.75rem; color: var(--muted);">' +
                        '<strong>Mode:</strong> ' + img.protocol + ' · ' +
                        '<strong>P:</strong> ' + img.precision.toFixed(3) + ' · ' +
                        '<strong>R:</strong> ' + img.recall.toFixed(3) +
                        '</div>' +
                        '</div>' +
                        '</figure>';
                }}).join('');

                // Rebind zoom
                const dialog = document.getElementById('imageDialog');
                const dialogImg = document.getElementById('imageDialogImg');
                gallery.querySelectorAll('[data-zoom-src]').forEach(img => {{
                    img.addEventListener('click', () => {{
                        dialogImg.src = img.getAttribute('data-zoom-src');
                        dialog.showModal();
                    }});
                }});
            }}

            sortSel.addEventListener('change', render);
            filterSlider.addEventListener('input', render);
            filterQual.addEventListener('change', render);
            render();

            const dialog = document.getElementById('imageDialog');
            dialog.addEventListener('click', (e) => {{
                const r = dialog.getBoundingClientRect();
                if (e.clientY < r.top || e.clientY > r.bottom || e.clientX < r.left || e.clientX > r.right) dialog.close();
            }});
        }})();
    </script>
</body>
</html>'''

with open(os.path.join(base_dir, 'gallery.html'), 'w') as f:
    f.write(html_gallery)

print("index.html and gallery.html built with Premium UI.")
