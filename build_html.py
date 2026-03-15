import os
import json

base_dir = '/home/nada/PycharmProjects/research-site/site/experiments/sam3-rwtd-both-test-visuals'

# Read training data for the gallery images
with open(os.path.join(base_dir, 'training_data.json')) as f:
    data = json.load(f)

with open(os.path.join(base_dir, 'metrics.json')) as f:
    mets = json.load(f)

# Find top 8 images in the gallery directory
gallery_files = os.listdir(os.path.join(base_dir, 'assets/gallery'))
qc_files = os.listdir(os.path.join(base_dir, 'assets/qc'))

def get_metrics_for_id(id_str):
    for item in data['per_image']:
        if item['id'] == id_str.replace('.png', ''):
            return item
    return None

def color_f1(f1):
    if f1 >= 0.7: return 'background:#edf9f1;border-color:#a6dfb8;color:#1d7a3e;'
    if f1 >= 0.4: return 'background:#fff7ec;border-color:#ffd29b;color:#9a4e02;'
    return 'background:#fce8e8;border-color:#e8a0a0;color:#9b2226;'

gallery_figures = ""
for gf in sorted(gallery_files):
    m = get_metrics_for_id(gf)
    if not m: continue
    color = color_f1(m['f1'])
    gallery_figures += f'''
        <figure data-score="{m['f1']:.4f}">
          <img src="assets/gallery/{gf}" alt="sample {m['id']} preview"
            data-zoom-src="assets/gallery/{gf}" />
          <figcaption><span class="pill">id={m['id']}</span><span class="pill"
              style="{color}">F1={m['f1']:.3f}</span><span class="pill">P={m['precision']:.3f}
              R={m['recall']:.3f}</span></figcaption>
        </figure>'''

qc_figures = ""
for qf in sorted(qc_files):
    m = get_metrics_for_id(qf)
    if not m: continue
    qc_figures += f'''
        <figure data-score="{m['id'].split('_')[-1]}">
          <img src="assets/qc/{qf}" alt="failure case sample {m['id']}"
            data-zoom-src="assets/qc/{qf}" />
          <figcaption><span class="pill">id={m['id']}</span><span class="pill">failure</span> Hard failure or severe over-prediction.</figcaption>
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
  <main>
    <section class="hero">
      <h1>RWTD Segmentation — SAM 3 Oracle Points & Text</h1>
      <p>Evaluation of the foundational SAM 3 model on the RWTD test split using both oracle points and text label prompting.</p>
      <p style="margin-top:8px;"><span class="tag good">evaluation</span><span class="tag">sam3</span><span
          class="tag">foundation model</span></p>
    </section>

    <section class="section">
      <h2>Executive Read</h2>
      <div class="summary">
        <p class="kv"><strong>Run:</strong> <code>sam3-rwtd-both-test-visuals</code> (reported on 2026-03-12)</p>
        <p class="kv"><strong>What was done:</strong> Zero-shot evaluation of facebook/sam3 using oracle points (2 per texture) and text descriptions.</p>
        <p class="kv"><strong>What works:</strong> Points are highly effective at guiding the model to the dominant texture region.</p>
        <p class="kv"><strong>Where it fails:</strong> The text descriptions alone are sometimes not sufficient to distinguish closely overlapping visual properties.</p>
      </div>
      <div class="metric-grid" style="margin-top:10px;">
        <article class="metric">
          <div class="k">Primary Metric (mIoU)</div>
          <div class="v">{mets['mIoU']:.3f}</div>
        </article>
        <article class="metric">
          <div class="k">ARI</div>
          <div class="v">{mets['ARI']:.3f}</div>
        </article>
        <article class="metric">
          <div class="k">Macro Dice</div>
          <div class="v">{mets['Macro_Dice']:.3f}</div>
        </article>
      </div>
    </section>

    <section class="section">
      <h2>Visual Preview Gallery</h2>
      <p class="kv">Representative samples from RWTD test evaluated with oracle points. Click any image to enlarge.</p>
      <div class="gallery-controls" style="margin-bottom:10px;">
        <label for="gallerySort"><strong>Sort:</strong></label>
        <select id="gallerySort" data-sort-gallery="mainGallery">
          <option value="default">Default order</option>
          <option value="asc">F1 ↑ (worst first)</option>
          <option value="desc">F1 ↓ (best first)</option>
        </select>
        <a class="btn" href="gallery.html" style="margin-left:auto;">Open full gallery (454 images) →</a>
      </div>
      <div class="gallery" id="mainGallery">
{gallery_figures}
      </div>
    </section>

    <section class="section">
      <h2>QC Diagnostics</h2>
      <p class="kv">Failure cases and severe miss-predictions from the sample pool.</p>
      <div class="gallery-controls" style="margin-bottom:10px;">
        <label for="qcSort"><strong>Sort:</strong></label>
        <select id="qcSort" data-sort-gallery="qcGallery">
          <option value="default">Default order</option>
          <option value="asc">Sample ID ↑ (best first)</option>
          <option value="desc">Sample ID ↓ (worst first)</option>
        </select>
      </div>
      <div class="gallery" id="qcGallery">
{qc_figures}
      </div>
    </section>

    <section class="section">
      <h2>Data and Repro (Details)</h2>
      <div class="actions" style="margin-top:10px;">
        <button data-toggle-target="reproDetails">Show details</button>
      </div>
      <div class="details" id="reproDetails">
        <div class="table-wrap">
          <table>
            <tbody>
              <tr>
                <th>Repro Config</th>
                <td><code>repro/config.json</code></td>
              </tr>
              <tr>
                <th>Experiment Terms</th>
                <td><code>repro/experiment_terms.md</code></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
      <p style="margin-top:10px;"><a class="btn" href="../../index.html">Back to overview</a></p>
    </section>

    <footer>Report page generated on 2026-03-12.</footer>
  </main>

  <dialog class="image-dialog" id="imageDialog">
    <img id="imageDialogImg" alt="Expanded preview" />
  </dialog>
  <script src="../../assets/site.js"></script>
</body>
</html>
'''

with open(os.path.join(base_dir, 'index.html'), 'w') as f:
    f.write(html_index)

html_gallery = '''<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Gallery — SAM 3 Both Protocols</title>
    <link rel="stylesheet" href="../../assets/site.css" />
    <style>
        .gallery-toolbar {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 10px;
            background: #f1f7fa;
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 12px 14px;
            margin-bottom: 12px;
        }

        .gallery-toolbar label {
            font-weight: 700;
            font-size: 0.88rem;
            color: var(--ink);
        }

        .gallery-toolbar select,
        .gallery-toolbar input {
            font: inherit;
            border: 1px solid #7bb2c0;
            background: #edf7fb;
            color: #0d3b4c;
            border-radius: 999px;
            padding: 5px 12px;
            font-size: 0.86rem;
        }

        .gallery-toolbar input[type="range"] {
            width: 120px;
        }

        .gallery-toolbar .count {
            font-size: 0.85rem;
            color: var(--muted);
            margin-left: auto;
        }

        #fullGallery {
            display: grid;
            gap: 10px;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
        }

        #fullGallery figure {
            margin: 0;
            border: 1px solid var(--line);
            border-radius: 10px;
            background: #fbfdff;
            overflow: hidden;
            position: relative;
        }

        #fullGallery figure img {
            width: 100%;
            display: block;
            cursor: zoom-in;
            border-radius: 10px 10px 0 0;
        }

        #fullGallery figure .metrics-bar {
            padding: 6px 8px;
            font-size: 0.78rem;
            line-height: 1.5;
            background: #f7fafd;
            border-top: 1px solid var(--line);
        }

        .metrics-bar .metric-label {
            font-weight: 700;
            color: var(--ink);
        }

        .metrics-bar .metric-good {
            color: var(--good);
            font-weight: 700;
        }

        .metrics-bar .metric-bad {
            color: var(--bad);
            font-weight: 700;
        }

        .metrics-bar .metric-mid {
            color: var(--accent);
            font-weight: 700;
        }

        .f1-badge {
            position: absolute;
            top: 6px;
            right: 6px;
            background: rgba(0, 0, 0, 0.72);
            color: #fff;
            font-size: 0.78rem;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 999px;
        }
    </style>
</head>

<body>
    <main>
        <section class="hero">
            <h1>Full Preview Gallery</h1>
            <p>SAM 3 RWTD Both Protocols — 454 predictions with per-image metrics</p>
            <p style="margin-top:8px;"><a class="btn" href="index.html"
                    style="border-color:#8ecae6;background:#edf7fc;color:#0b4f6c;">← Back to report</a></p>
        </section>

        <section class="section">
            <div class="gallery-toolbar">
                <label for="sortBy">Sort:</label>
                <select id="sortBy">
                    <option value="f1_desc">F1 ↓ best first</option>
                    <option value="f1_asc">F1 ↑ worst first</option>
                    <option value="precision_desc">Precision ↓</option>
                    <option value="recall_desc">Recall ↓</option>
                    <option value="id_asc">Sample ID ↑</option>
                    <option value="id_desc">Sample ID ↓</option>
                </select>
                <label for="filterF1">Min F1:</label>
                <input type="range" id="filterF1" min="0" max="100" value="0" />
                <span id="filterF1Val">0.00</span>
                <label for="filterQuality">Quality:</label>
                <select id="filterQuality">
                    <option value="all">All</option>
                    <option value="good">Good (F1 ≥ 0.7)</option>
                    <option value="medium">Medium (0.4–0.7)</option>
                    <option value="bad">Bad (F1 < 0.4)</option>
                </select>
                <span class="count" id="countLabel">454 / 454</span>
            </div>
            <div id="fullGallery"></div>
        </section>

        <footer>Full gallery generated from per-image evaluation data. 2026-03-12.</footer>
    </main>

    <dialog class="image-dialog" id="imageDialog">
        <img id="imageDialogImg" alt="Expanded preview" />
    </dialog>

    <script>
        (async () => {
            const resp = await fetch('training_data.json');
            const data = await resp.json();
            const gallery = document.getElementById('fullGallery');
            const sortSel = document.getElementById('sortBy');
            const filterSlider = document.getElementById('filterF1');
            const filterVal = document.getElementById('filterF1Val');
            const filterQual = document.getElementById('filterQuality');
            const countLabel = document.getElementById('countLabel');
            const total = data.per_image.length;

            function f1Class(f1) {
                if (f1 >= 0.7) return 'metric-good';
                if (f1 >= 0.4) return 'metric-mid';
                return 'metric-bad';
            }

            function qualityLabel(f1) {
                if (f1 >= 0.7) return 'good';
                if (f1 >= 0.4) return 'medium';
                return 'bad';
            }

            function render() {
                const sortKey = sortSel.value;
                const minF1 = parseFloat(filterSlider.value) / 100;
                const qualFilter = filterQual.value;
                filterVal.textContent = minF1.toFixed(2);

                let items = data.per_image.filter(img => {
                    if (img.f1 < minF1) return false;
                    if (qualFilter !== 'all' && qualityLabel(img.f1) !== qualFilter) return false;
                    return true;
                });

                const [field, dir] = sortKey.split('_');
                items.sort((a, b) => {
                    const va = a[field], vb = b[field];
                    return dir === 'desc' ? vb - va : va - vb;
                });

                countLabel.textContent = items.length + ' / ' + total;

                gallery.innerHTML = items.map((img, i) => {
                    const cls = f1Class(img.f1);
                    return '<figure>' +
                        '<span class="f1-badge">F1 ' + img.f1.toFixed(3) + '</span>' +
                        '<img src="assets/all_previews/' + img.id + '.png" alt="sample ' + img.id + '" data-zoom-src="assets/all_previews/' + img.id + '.png" />' +
                        '<div class="metrics-bar">' +
                        '<span class="pill">id=' + img.id.split('_').slice(-1)[0] + '</span>' +
                        '<span class="pill">rank ' + (i + 1) + '/' + items.length + '</span><br/>' +
                        '<span class="metric-label">Mode:</span> <span class="metric-mid">' + img.id.split('_')[0] + '</span><br/>' +
                        '<span class="metric-label">F1 (mIoU):</span> <span class="' + cls + '">' + img.f1.toFixed(3) + '</span> · ' +
                        '<span class="metric-label">P:</span> ' + img.precision.toFixed(3) + ' · ' +
                        '<span class="metric-label">R:</span> ' + img.recall.toFixed(3) +
                        '</div>' +
                        '</figure>';
                }).join('');

                // Rebind zoom
                const dialog = document.getElementById('imageDialog');
                const dialogImg = document.getElementById('imageDialogImg');
                gallery.querySelectorAll('[data-zoom-src]').forEach(img => {
                    img.addEventListener('click', () => {
                        dialogImg.src = img.getAttribute('data-zoom-src');
                        dialog.showModal();
                    });
                });
            }

            sortSel.addEventListener('change', render);
            filterSlider.addEventListener('input', render);
            filterQual.addEventListener('change', render);
            render();

            const dialog = document.getElementById('imageDialog');
            dialog.addEventListener('click', (e) => {
                const r = dialog.getBoundingClientRect();
                if (e.clientY < r.top || e.clientY > r.bottom || e.clientX < r.left || e.clientX > r.right) dialog.close();
            });
        })();
    </script>
</body>
</html>'''

with open(os.path.join(base_dir, 'gallery.html'), 'w') as f:
    f.write(html_gallery)

print("index.html and gallery.html built.")
