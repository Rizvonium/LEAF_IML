"""Build LeafIDMir.ipynb from the cell sources below.

Run:  python tools/build_notebook.py            -> writes LeafIDMir.ipynb
The notebook is the deliverable; this script only exists so the cells can be
edited as plain Python and the notebook regenerated.
"""
import sys
from pathlib import Path

import nbformat as nbf

CELLS = []


def md(s):
    CELLS.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    CELLS.append(nbf.v4.new_code_cell(s.strip("\n")))


# ==========================================================================
md(r"""
# LeafID: identifying tree species from scanned leaves (CNN trained from scratch)

**Task.** 600 labelled leaf scans from 15 species (40 per species) and 525 unlabelled test scans.
The starter notebook trains a small CNN that performs poorly. This notebook
1. runs the starter as given (baseline),
2. diagnoses why it performs poorly,
3. fixes and improves it **one change at a time**, scoring every change with the same 5-fold stratified cross-validation, paired fold-by-fold and against a measured seed-noise floor,
4. backs each change with a specific piece of evidence, and
5. trains the final ensemble and writes `predictions.csv`.

No pretrained weights or external data: every network is trained from random initialisation on the 600 labelled images only.

**To run:** Colab → Runtime → Change runtime type → **T4 GPU**; upload `leaves.zip` to `/content`; Runtime → **Run all**.
""")

md("## 0. Setup")
code(r"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')   # deterministic cuBLAS
import time, math, random, zipfile, copy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy import ndimage, stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score
try:
    display
except NameError:            # outside Jupyter
    display = print

QUICK = os.environ.get('LEAF_QUICK') == '1'      # tiny smoke-test settings (local CPU testing only)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', DEVICE, '| torch', torch.__version__, '| QUICK =', QUICK)
if DEVICE.type == 'cuda':
    print(torch.cuda.get_device_name(0))
elif not QUICK and os.environ.get('LEAF_ALLOW_CPU') != '1':
    raise RuntimeError('No GPU. Runtime > Change runtime type > T4 GPU, then Run all.')

# reproducibility: same seed -> same weights, so paired comparisons between rows are not polluted by GPU noise
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
T0 = time.time()
""")

# --------------------------------------------------------------------------
md(r"""
## 1. Load the data
Upload `leaves.zip` to `/content` (Files panel) before running.
""")
code(r"""
DATA_DIR = Path(os.environ.get('LEAF_DIR', '/content/leaves_data'))
ZIP_PATH = Path(os.environ.get('LEAF_ZIP', '/content/leaves.zip'))
if not DATA_DIR.exists():
    if not ZIP_PATH.exists():
        zips = sorted(Path('/content').glob('*.zip')) if Path('/content').exists() else []
        if zips:
            ZIP_PATH = zips[0]
        else:
            from google.colab import files
            uploaded = files.upload()
            if not uploaded:
                raise FileNotFoundError('no zip uploaded')
            ZIP_PATH = Path(next(iter(uploaded))).resolve(); del uploaded
    print('extracting', ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(DATA_DIR)

IMG_EXT = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
all_imgs = sorted(p for p in DATA_DIR.rglob('*')
                  if p.suffix.lower() in IMG_EXT and '__MACOSX' not in p.parts and not p.name.startswith('.'))
is_test = lambda p: any(part.lower().startswith('test') for part in p.relative_to(DATA_DIR).parts[:-1])
train_paths = [p for p in all_imgs if not is_test(p)]
test_paths = [p for p in all_imgs if is_test(p)]

# Labels: from a CSV whose rows name the training images, if there is one; otherwise from the parent folder.
# A CSV that names the *test* images is treated as the submission template (its columns and row order are kept).
def _stem_set(paths):
    return {p.stem for p in paths} | {p.name for p in paths}
label_map, SUB_TEMPLATE = None, None
for c in sorted(p for p in DATA_DIR.rglob('*.csv') if '__MACOSX' not in p.parts):
    df = pd.read_csv(c)
    for col in df.columns:
        vals = df[col].astype(str)
        if len(train_paths) and vals.isin(_stem_set(train_paths)).mean() > 0.9:
            other = [k for k in df.columns if k != col and df[k].nunique() <= 100]
            if other:
                label_map = {str(k): str(v) for k, v in zip(vals, df[other[0]])}
                print(f'labels from {c.name}: {col} -> {other[0]}')
        if len(test_paths) and vals.isin(_stem_set(test_paths)).mean() > 0.9:
            SUB_TEMPLATE = (df, col); print(f'submission template: {c.name} (id column {col!r}, columns {list(df.columns)})')
if label_map:
    train_labels_str = [label_map.get(p.name, label_map.get(p.stem)) for p in train_paths]
else:
    train_labels_str = [p.parent.name for p in train_paths]

CLASSES = sorted(set(train_labels_str))
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
y_all = np.array([CLS2IDX[s] for s in train_labels_str])
N_CLASSES = len(CLASSES)
print(f'{len(train_paths)} labelled images, {len(test_paths)} test images, {N_CLASSES} classes')
print(pd.Series(train_labels_str).value_counts().sort_index().to_string())
assert N_CLASSES == 15, f'expected 15 species, found {N_CLASSES}: check the folder/CSV layout'
assert len({p.name for p in test_paths}) == len(test_paths), 'test file names are not unique'
if (len(train_paths), len(test_paths)) != (600, 525):
    print('WARNING: expected 600 train / 525 test images')
""")

md("### 1.1 What do the images look like?")
code(r"""
info = []
for p in train_paths + test_paths:
    with Image.open(p) as im:
        info.append((im.size[0], im.size[1], im.mode, str(im.info.get('dpi', ''))))
meta = pd.DataFrame(info, columns=['w', 'h', 'mode', 'dpi'])
meta['split'] = ['train'] * len(train_paths) + ['test'] * len(test_paths)
print(meta.groupby('split')[['w', 'h']].describe().T.round(0))
print('modes:', meta['mode'].value_counts().to_dict(), '| dpi:', meta['dpi'].value_counts().head(3).to_dict())

fig, axes = plt.subplots(3, 5, figsize=(15, 9))
for ax, c in zip(axes.flat, CLASSES):
    p = train_paths[int(np.where(y_all == CLS2IDX[c])[0][0])]
    ax.imshow(Image.open(p).convert('RGB')); ax.set_title(c, fontsize=9); ax.axis('off')
plt.suptitle('One labelled example per species'); plt.tight_layout(); plt.show()

fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))
for ax, p in zip(axes.flat, test_paths[:16]):
    ax.imshow(Image.open(p).convert('RGB')); ax.set_title(p.name, fontsize=7); ax.axis('off')
plt.suptitle('Unlabelled test images'); plt.tight_layout(); plt.show()
""")

# --------------------------------------------------------------------------
md(r"""
## 2. The starter notebook, run as given (baseline)

*(Starter code goes here unchanged, with its outputs.)*
""")
code(r"""
# STARTER CODE PLACEHOLDER: replaced with the provided starter cells once available.
""")

md(r"""
## 3. Diagnosis: why the starter performs poorly

*(Filled in after reading and running the starter.)*
""")

# --------------------------------------------------------------------------
md(r"""
## 4. Improved pipeline

### 4.1 Preprocessing
Every scan is decoded **once** at full resolution and turned into two square versions at `BASE_RES`:
* **no-crop**: the whole scan padded to a square with its own background colour, so the aspect ratio is kept (squashing to a square would distort leaf shape differently for each scan);
* **crop**: the leaf is found by thresholding against the background colour (small specks removed) and cut out of the *full-resolution* scan. It is centred in a square whose side is the diameter of the leaf's enclosing circle plus a margin, so rotating it never pushes the tip or stem out of the frame.

Measurements of each leaf (the angle of its long axis, and its length in original pixels) are also stored; 4.2 uses them to check orientation and size.
""")
code(r"""
BASE_RES = 96 if QUICK else 384
THR, MARGIN = 40, 0.06

def border_bg(a):
    b = np.concatenate([a[:3].reshape(-1, 3), a[-3:].reshape(-1, 3), a[:, :3].reshape(-1, 3), a[:, -3:].reshape(-1, 3)])
    return np.median(b, axis=0)

def leaf_mask(a, bg):
    m = np.abs(a.astype(np.int16) - bg.astype(np.int16)[None, None]).max(axis=2) > THR
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    keep = np.isin(lab, 1 + np.nonzero(sizes >= 0.02 * sizes.max())[0])   # drop dust / specks
    return keep

def to_square(arr, cy, cx, s, bg, res):
    # place arr into an s x s canvas (filled with bg) so that (cy, cx) lands at the centre; resize to res
    canvas = np.empty((s, s, 3), np.uint8); canvas[:] = bg.astype(np.uint8)
    oy, ox = int(round(cy - s / 2)), int(round(cx - s / 2))
    H, W = arr.shape[:2]
    sy0, sx0, sy1, sx1 = max(0, oy), max(0, ox), min(H, oy + s), min(W, ox + s)
    canvas[sy0 - oy:sy1 - oy, sx0 - ox:sx1 - ox] = arr[sy0:sy1, sx0:sx1]
    im = Image.fromarray(canvas)
    while im.width > 4 * res:
        im = im.reduce(2)
    return np.asarray(im.resize((res, res), Image.LANCZOS))

def decode(path):
    img = Image.open(path); img.draft('RGB', (2 * BASE_RES * 4, 2 * BASE_RES * 4))
    img = ImageOps.exif_transpose(img).convert('RGB')
    full = np.asarray(img); H, W = full.shape[:2]
    small = img.copy(); small.thumbnail((512, 512), Image.BILINEAR); f = W / small.width
    a = np.asarray(small); bg = border_bg(a); m = leaf_mask(a, bg)
    ys, xs = np.nonzero(m)
    nocrop = to_square(full, H / 2, W / 2, max(H, W), bg, BASE_RES)
    if len(ys) < 50:                                   # leaf not found: fall back to the whole scan
        return nocrop, nocrop, bg / 255, (np.nan, np.nan, np.nan)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2).max()
    s = int(math.ceil(2 * r * (1 + 2 * MARGIN) * f)) + 2
    crop = to_square(full, cy * f, cx * f, s, bg, BASE_RES)
    # shape measurements (on the thumbnail mask, lengths converted to original pixels)
    pts = np.stack([xs - xs.mean(), ys - ys.mean()], 1).astype(np.float64)
    evals, evecs = np.linalg.eigh(np.cov(pts.T)); v = evecs[:, -1]
    angle = math.degrees(math.atan2(v[0], v[1]))         # 0 = long axis vertical
    angle = (angle + 90) % 180 - 90
    proj = pts @ v
    length = (proj.max() - proj.min()) * f
    return nocrop, crop, bg / 255, (angle, length, m.mean())

t = time.time()
with ThreadPoolExecutor(8) as ex:
    dec = list(ex.map(decode, train_paths + test_paths))
NOCROP = torch.from_numpy(np.stack([d[0] for d in dec])).permute(0, 3, 1, 2).contiguous()
CROP = torch.from_numpy(np.stack([d[1] for d in dec])).permute(0, 3, 1, 2).contiguous()
BG = torch.tensor(np.stack([d[2] for d in dec]), dtype=torch.float32)
LEAF = pd.DataFrame([d[3] for d in dec], columns=['angle', 'length_px', 'area_frac'])
LEAF['split'] = meta['split'].values
del dec
NTR = len(train_paths)
print(f'decoded {len(LEAF)} images in {time.time() - t:.0f}s; leaf not found in {LEAF.length_px.isna().sum()} images')

fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))
for i in range(8):
    j = i * (NTR // 8)
    axes[0, i].imshow(NOCROP[j].permute(1, 2, 0)); axes[0, i].set_title('no-crop ' + CLASSES[y_all[j]], fontsize=7)
    axes[1, i].imshow(CROP[j].permute(1, 2, 0)); axes[1, i].set_title('crop', fontsize=7)
for ax in axes.flat: ax.axis('off')
plt.tight_layout(); plt.show()
""")

md(r"""
### 4.2 Are the scans upright, and does leaf size carry information?
Two facts about the data drive later choices:
* **Orientation.** If train *and* test leaves all lie with the long axis near-vertical, orientation is consistent, and full 360° rotation augmentation would throw away a free cue (which end is the tip and which is the stem). The rotation range is then chosen by cross-validation (section 5.2).
* **Size.** Cropping to the leaf and resizing makes every leaf the same size in pixels. If the scans share one resolution (DPI), the leaf's true length is a cheap species cue that the crop removes. Section 5.2 tests feeding it back in as an extra input.
""")
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(17, 4))
for sp, col in [('train', 'C0'), ('test', 'C1')]:
    axes[0].hist(LEAF.loc[LEAF.split == sp, 'angle'], bins=36, range=(-90, 90), alpha=.6, density=True, label=sp, color=col)
axes[0].set_xlabel('angle of leaf long axis from vertical (deg)'); axes[0].legend(); axes[0].set_title('Orientation, train vs test')
tr_len = LEAF.length_px.values[:NTR]
order = np.argsort([np.nanmedian(tr_len[y_all == k]) for k in range(N_CLASSES)])
axes[1].boxplot([tr_len[y_all == k][~np.isnan(tr_len[y_all == k])] for k in order], showfliers=False)
axes[1].set_xticks(range(1, N_CLASSES + 1)); axes[1].set_xticklabels([CLASSES[k] for k in order], rotation=90, fontsize=7)
axes[1].set_ylabel('leaf length (original px)'); axes[1].set_title('Leaf length per species (train)')
for sp, col in [('train', 'C0'), ('test', 'C1')]:
    axes[2].hist(LEAF.loc[LEAF.split == sp, 'length_px'].dropna(), bins=30, alpha=.6, density=True, label=sp, color=col)
axes[2].set_xlabel('leaf length (original px)'); axes[2].legend(); axes[2].set_title('Leaf length, train vs test')
plt.tight_layout(); plt.show()
near_vertical = (LEAF.angle.abs() < 30).groupby(LEAF.split).mean()
print('fraction of leaves within 30 deg of vertical:', near_vertical.round(3).to_dict())
h = stats.kruskal(*[tr_len[y_all == k][~np.isnan(tr_len[y_all == k])] for k in range(N_CLASSES)])
print(f'leaf length differs between species: Kruskal-Wallis H={h.statistic:.0f}, p={h.pvalue:.1e}')
""")

md(r"""
### 4.3 Augmentation (on the GPU)
Changes that should not alter the species label: left-right mirroring, a small rotation (or any rotation, if scans are not upright), slight zoom and shift, and brightness/contrast changes (scanner variation). Each batch gets a random affine warp; the area exposed by the warp is filled with that image's own background colour, so the model never sees black corners. The crop square is large enough that no rotation or zoom here cuts off the leaf. Up-down flips are *not* used: they swap the tip and the stem.
""")
code(r"""
def gpu_augment(x, bg, a):
    # x: (B,3,H,W) float in [0,1]; bg: (B,3) in [0,1]
    B, dev = x.size(0), x.device
    ang = (torch.rand(B, device=dev) * 2 - 1) * math.radians(a['rot_deg'])
    sc = torch.empty(B, device=dev).uniform_(*a['scale'])          # sc > 1 zooms in
    fl = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0) if a['flip'] else torch.ones(B, device=dev)
    tx, ty = (torch.rand(2, B, device=dev) * 2 - 1) * a['translate']
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos * fl, -sin, tx], 1), torch.stack([sin * fl, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    bgv = bg.view(B, 3, 1, 1)
    x = F.grid_sample(x - bgv, grid, mode='bilinear', padding_mode='zeros', align_corners=False) + bgv
    if a['jitter'] > 0:
        b = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * a['jitter']
        c = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * a['jitter']
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        x = ((x - m) * c + m) * b
    return x.clamp(0, 1)

def make_aug(rot_deg):
    return dict(rot_deg=rot_deg, scale=(0.85, 1.05), flip=True, translate=0.04, jitter=0.15)
AUG = make_aug(20)

def tta_views(x, aug):
    # test-time views that match what the model was trained to be invariant to
    if aug is None:
        return [x]
    views = [x, x.flip(3)] if aug['flip'] else [x]
    if aug['rot_deg'] >= 90:
        views = [torch.rot90(v, k, dims=(2, 3)) for v in views for k in range(4)]
    return views

_x = CROP[:8].float().div(255); _bg = BG[:8]
_xa = gpu_augment(_x.repeat(2, 1, 1, 1), _bg.repeat(2, 1), AUG)
fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))
for i, ax in enumerate(axes.flat):
    ax.imshow(_xa[i].permute(1, 2, 0)); ax.axis('off')
plt.suptitle('Random augmentations of 8 training images (two draws each)'); plt.tight_layout(); plt.show()
""")

md(r"""
### 4.4 Models
* `plain`: conv → ReLU → max-pool blocks, flattened into a large fully connected layer, no normalisation (the starter's design style).
* `leafnet`: a small residual CNN trained from scratch: BatchNorm after every convolution, residual blocks, global average pooling instead of the large flatten → dense layer, dropout before the classifier. It can optionally take the leaf's length as one extra input next to the pooled image features.
""")
code(r"""
class PlainCNN(nn.Module):
    def __init__(self, n_classes, res, widths=(16, 32, 64), hidden=128):
        super().__init__()
        layers, c = [], 3
        for w in widths:
            layers += [nn.Conv2d(c, w, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c = w
        self.features = nn.Sequential(*layers)
        s = res // 2 ** len(widths)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c * s * s, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, n_classes))
    def forward(self, x, size=None):
        return self.head(self.features(x))

def conv_bn(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.short = (nn.Identity() if stride == 1 and cin == cout else
                      nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))
    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(out)) + self.short(x))

class LeafNet(nn.Module):
    def __init__(self, n_classes, widths=(32, 64, 128, 256), dropout=0.3, use_size=False):
        super().__init__()
        self.stem = nn.Sequential(conv_bn(3, widths[0], 2), conv_bn(widths[0], widths[0]), nn.MaxPool2d(2))
        blocks, c = [], widths[0]
        for i, w in enumerate(widths):
            blocks.append(ResBlock(c, w, 1 if i == 0 else 2)); c = w
        self.blocks = nn.Sequential(*blocks)
        self.use_size = use_size
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c + int(use_size), n_classes)
    def forward(self, x, size=None):
        f = self.blocks(self.stem(x)).mean(dim=(2, 3))            # global average pooling
        if self.use_size:
            f = torch.cat([f, size.view(-1, 1).to(f.dtype)], 1)
        return self.fc(self.drop(f))

def build_model(cfg):
    if cfg['arch'] == 'plain':
        return PlainCNN(N_CLASSES, cfg['res'])
    return LeafNet(N_CLASSES, dropout=cfg['dropout'], use_size=cfg['use_size'])

n_params = lambda m: sum(p.numel() for p in m.parameters())
print('plain   params:', f"{n_params(PlainCNN(N_CLASSES, 64)):,}", '(at 64px)')
print('leafnet params:', f"{n_params(LeafNet(N_CLASSES)):,}")
""")

md(r"""
### 4.5 Training and evaluation
* **5-fold stratified cross-validation, same folds for every configuration.** With 40 images per species, one 80/20 split leaves 8 validation images per species, so a single split can swing by several points by chance. Five folds validate on all 600 images once.
* **Paired comparison.** Rows are compared fold by fold (mean difference with a 95% t-interval over the 5 folds), and by an exact McNemar test on the 600 out-of-fold predictions (does the new row fix more images than it breaks?). Section 5.2 also re-runs the final configuration with **fresh seeds**: the gap between two runs of the same configuration is the noise floor, and a gain smaller than that is not evidence.
* **Honest numbers.** Each model is scored **after its last epoch**, never at its best validation epoch. Normalisation statistics (and the leaf-size scaling) come from the training fold only.
* **Learning curves** score a fixed subset of training images exactly like validation images (eval mode, no augmentation, plain cross-entropy), so the train-vs-val gap measures memorisation rather than measurement differences.
""")
code(r"""
BASE = dict(arch='leafnet', res=192, crop=True, epochs=120, bs=32, opt='adamw', lr=2e-3, wd=0.05,
            sched='onecycle', label_smoothing=0.1, aug=AUG, normalize=True, dropout=0.3, use_size=False)
N_FOLDS = 2 if QUICK else 5
FOLDS = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=42).split(np.zeros(len(y_all)), y_all))

def quick(cfg):
    return dict(cfg, res=min(cfg['res'], 48), epochs=2) if QUICK else cfg

_res_cache = {}
def data_at(cfg):
    # (X uint8 (N,3,r,r), bg (N,3), log-length (N,)) for all train+test images at this resolution / crop setting
    key = (cfg['crop'], cfg['res'])
    if key not in _res_cache:
        X = CROP if cfg['crop'] else NOCROP
        if X.shape[-1] != cfg['res']:
            X = torch.cat([F.interpolate(X[i:i + 64].float(), size=(cfg['res'],) * 2, mode='bilinear',
                                         antialias=True, align_corners=False).round().clamp(0, 255).to(torch.uint8)
                           for i in range(0, len(X), 64)])
        L = torch.tensor(np.log(LEAF.length_px.fillna(LEAF.length_px.median()).values), dtype=torch.float32)
        _res_cache[key] = (X, BG, L)
    return _res_cache[key]

class Stats:
    # per-model statistics, computed on that model's training images only
    def __init__(self, X, L):
        x = X.float().div(255)
        self.mean = x.mean(dim=(0, 2, 3)).to(DEVICE); self.std = x.std(dim=(0, 2, 3)).to(DEVICE)
        self.lmean, self.lstd = L.mean().item(), L.std().item() + 1e-6

def prep(xb, bgb, cfg, st, train):
    x = xb.to(DEVICE, non_blocking=True).float().div_(255)
    if train and cfg['aug'] is not None:
        x = gpu_augment(x, bgb.to(DEVICE), cfg['aug'])
    if cfg['normalize']:
        x = (x - st.mean.view(1, 3, 1, 1)) / st.std.view(1, 3, 1, 1)
    return x

def make_opt(model, cfg):
    if cfg['opt'] == 'adamw':
        decay = [p for p in model.parameters() if p.ndim > 1]
        no_decay = [p for p in model.parameters() if p.ndim <= 1]
        return torch.optim.AdamW([{'params': decay, 'weight_decay': cfg['wd']},
                                  {'params': no_decay, 'weight_decay': 0.0}], lr=cfg['lr'])
    return torch.optim.Adam(model.parameters(), lr=cfg['lr'])

@torch.no_grad()
def predict_proba(model, X, bg, L, cfg, st, tta=False, bs=64):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        x = prep(X[i:i + bs], bg[i:i + bs], cfg, st, train=False)
        s = ((L[i:i + bs] - st.lmean) / st.lstd).to(DEVICE)
        views = tta_views(x, cfg['aug']) if tta else [x]
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == 'cuda'):
            p = torch.stack([F.softmax(model(v, s).float(), 1) for v in views]).mean(0)
        out.append(p.cpu())
    return torch.cat(out).numpy()

def nll(p, y):
    return float(-np.log(p[np.arange(len(y)), y] + 1e-8).mean())

def train_model(cfg, idx, seed, va_idx=None, eval_every=5):
    # idx / va_idx index into the training images
    seed_everything(seed)
    X, bg, L = data_at(cfg)
    Xt, bgt, Lt = X[idx].to(DEVICE), bg[idx].to(DEVICE), L[idx].to(DEVICE)
    yt = torch.as_tensor(y_all[idx]).to(DEVICE)
    st = Stats(X[idx], L[idx])
    model = build_model(cfg).to(DEVICE).to(memory_format=torch.channels_last)
    opt = make_opt(model, cfg)
    steps = len(idx) // cfg['bs']; total = steps * cfg['epochs']           # drop_last: every step is a full batch
    sched = (torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg['lr'], total_steps=total, pct_start=0.15)
             if cfg['sched'] == 'onecycle' else None)
    scaler = torch.amp.GradScaler('cuda', enabled=DEVICE.type == 'cuda')
    crit = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
    sub = np.arange(0, len(idx), max(1, len(idx) // 120))                  # fixed train subset for clean curves
    hist = {k: [] for k in ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']}
    for ep in range(cfg['epochs']):
        model.train(); perm = torch.randperm(len(idx), device=DEVICE)
        for b in range(steps):
            j = perm[b * cfg['bs']:(b + 1) * cfg['bs']]
            x = prep(Xt[j], bgt[j], cfg, st, train=True).contiguous(memory_format=torch.channels_last)
            s = (Lt[j] - st.lmean) / st.lstd
            with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == 'cuda'):
                loss = crit(model(x, s).float(), yt[j])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if sched: sched.step()
        if va_idx is not None and ((ep + 1) % eval_every == 0 or ep == cfg['epochs'] - 1):
            pt = predict_proba(model, X[idx[sub]], bg[idx[sub]], L[idx[sub]], cfg, st)
            pv = predict_proba(model, X[va_idx], bg[va_idx], L[va_idx], cfg, st)
            hist['epoch'].append(ep + 1)
            hist['train_loss'].append(nll(pt, y_all[idx[sub]])); hist['train_acc'].append(float((pt.argmax(1) == y_all[idx[sub]]).mean()))
            hist['val_loss'].append(nll(pv, y_all[va_idx])); hist['val_acc'].append(float((pv.argmax(1) == y_all[va_idx]).mean()))
    model.st = st
    return model, hist

def cross_validate(cfg, name='', keep_models=False, seed_offset=0, verbose=True):
    cfg = quick(cfg); t = time.time()
    X, bg, L = data_at(cfg)
    oof, oof_tta = np.zeros((NTR, N_CLASSES)), np.zeros((NTR, N_CLASSES))
    hists, models = [], []
    for f, (tr, va) in enumerate(FOLDS):
        model, h = train_model(cfg, tr, seed=f + seed_offset, va_idx=va)
        oof[va] = predict_proba(model, X[va], bg[va], L[va], cfg, model.st)
        oof_tta[va] = predict_proba(model, X[va], bg[va], L[va], cfg, model.st, tta=True)
        hists.append(h)
        if keep_models: models.append(model.cpu())
        del model
    fold_acc = lambda P: [float((P[va].argmax(1) == y_all[va]).mean()) for _, va in FOLDS]
    r = dict(name=name, cfg=cfg, oof=oof, oof_tta=oof_tta, fold_acc=fold_acc(oof), fold_acc_tta=fold_acc(oof_tta),
             hists=hists, models=models, secs=time.time() - t)
    r['mean'] = float(np.mean(r['fold_acc']))
    if verbose:
        print(f"{name:52s} {r['mean']*100:5.1f}%  folds {[round(a * 100, 1) for a in r['fold_acc']]}  "
              f"(TTA {np.mean(r['fold_acc_tta'])*100:.1f}%, {r['secs']:.0f}s)")
    return r

def with_tta(r, name):
    # inference-only change: same models, predictions averaged over test-time views
    return dict(r, name=name, oof=r['oof_tta'], fold_acc=r['fold_acc_tta'], mean=float(np.mean(r['fold_acc_tta'])), secs=0.0)

def compare(a, b):
    # paired comparison of result b against result a on the same folds / images
    d = np.array(b['fold_acc']) - np.array(a['fold_acc'])
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else np.nan
    ca, cb = a['oof'].argmax(1) == y_all, b['oof'].argmax(1) == y_all
    fixed, broke = int((~ca & cb).sum()), int((ca & ~cb).sum())
    p = stats.binomtest(fixed, fixed + broke, 0.5).pvalue if fixed + broke else 1.0
    return dict(gain=d.mean() * 100, ci=half * 100, folds_up=f'{int((d > 0).sum())}/{len(d)}', fixed=fixed, broke=broke, p_mcnemar=p)
""")

# --------------------------------------------------------------------------
md(r"""
## 5. Experiments

### 5.1 Ablation: one change at a time
Each row adds exactly **one** change to the row above, on the same five folds and seeds. The order follows the diagnosis: fix what is broken first, then reduce overfitting, then refine. The cheap rows run at 64 px; resolution is raised once, near the end.
""")
code(r"""
def cfg_with(base, **kw):
    c = copy.deepcopy(base); c.update(kw); return c

STARTER_LIKE = cfg_with(BASE, arch='plain', res=64, crop=False, epochs=30, opt='adam', lr=1e-3, wd=0.0, sched=None,
                        label_smoothing=0.0, aug=None, normalize=False, dropout=0.0)
steps = [('A0 starter-like: plain CNN, 64px, raw pixels, 30 ep', STARTER_LIKE)]
def add(name, **kw):
    steps.append((name, cfg_with(steps[-1][1], **kw)))

add('A1 + standardise inputs (train-fold mean/std)', normalize=True)
add('A2 + crop to leaf (rotation-safe square)', crop=True)
add('A3 + LeafNet (BatchNorm, residual, GAP, dropout .3)', arch='leafnet', dropout=0.3)
add('A4 + train 4x longer (120 epochs)', epochs=BASE['epochs'])
add('A5 + augmentation (mirror, +-20 deg, zoom, shift, light)', aug=AUG)
add('A6 + AdamW weight decay 0.05', opt='adamw', wd=BASE['wd'])
add('A7 + one-cycle LR schedule (peak 2e-3)', sched='onecycle', lr=BASE['lr'])
add('A8 + label smoothing 0.1', label_smoothing=0.1)
add('A9 + resolution 64 -> 192 px', res=BASE['res'])

# rows whose models are needed later for the evidence plots, picked by config rather than by name
NOAUG_ROW = [n for n, c in steps if c['arch'] == 'leafnet' and c['aug'] is None][-1]
AUG_ROW = next(n for n, c in steps if c['aug'] is not None)
HIRES_ROW = steps[-1][0]

results = {}
for name, cfg in steps:
    results[name] = cross_validate(cfg, name, keep_models=name in (NOAUG_ROW, AUG_ROW, HIRES_ROW))
TTA_ROW = 'A10 + test-time augmentation (mirror)'
results[TTA_ROW] = with_tta(results[HIRES_ROW], TTA_ROW)
""")
code(r"""
def ablation_table(names):
    rows = []
    for i, n in enumerate(names):
        r = results[n]; row = {'step': n, 'CV acc %': r['mean'] * 100, 'fold sd': np.std(r['fold_acc'], ddof=1) * 100,
                               'macro F1': f1_score(y_all, r['oof'].argmax(1), average='macro')}
        if i:
            c = compare(results[names[i - 1]], r)
            row.update({'gain %': c['gain'], '95% CI ±': c['ci'], 'folds up': c['folds_up'],
                        'fixed/broke': f"{c['fixed']}/{c['broke']}", 'McNemar p': c['p_mcnemar']})
        row['train s'] = round(r['secs']); rows.append(row)
    return pd.DataFrame(rows)

names = list(results)
display(ablation_table(names).round(3))

fig, ax = plt.subplots(figsize=(12, 4.5))
m = [results[n]['mean'] * 100 for n in names]
ax.bar(range(len(names)), m, color='#4C72B0', alpha=.8)
for i, n in enumerate(names):
    ax.scatter([i] * N_FOLDS, np.array(results[n]['fold_acc']) * 100, color='k', s=10, zorder=3)
    ax.text(i, m[i] + 1, f'{m[i]:.1f}', ha='center', fontsize=8)
ax.set_xticks(range(len(names))); ax.set_xticklabels([n.split(' ', 1)[0] for n in names])
ax.set_ylabel('5-fold CV accuracy (%)'); ax.set_title('Ablation: each bar adds one change (dots = the 5 folds)')
ax.grid(axis='y', alpha=.3); plt.tight_layout(); plt.show()
""")

md(r"""
### 5.2 Choices made by cross-validation, and the noise floor
Starting from A9 (192 px), each experiment changes one setting:
* **noise floor:** A9 again, with fresh seeds. Any gain smaller than this gap is noise;
* **rotation range:** ±45° and ±180° (with matching test-time views) against ±20°;
* **resolution:** 128 and 256 px against 192;
* **leaf size:** the leaf's length given to the classifier as an extra input.
""")
code(r"""
A9 = results[HIRES_ROW]
exp = {'A9 (reference)': A9}
exp['noise: A9 with fresh seeds'] = cross_validate(steps[-1][1], 'noise: A9 with fresh seeds', seed_offset=1000)
for deg in (45, 180):
    exp[f'rotation +-{deg}'] = cross_validate(cfg_with(steps[-1][1], aug=make_aug(deg)), f'rotation +-{deg} deg')
for res in (128, 256):
    exp[f'res {res}'] = cross_validate(cfg_with(steps[-1][1], res=res), f'resolution {res}px')
exp['+ leaf size input'] = cross_validate(cfg_with(steps[-1][1], use_size=True), '+ leaf length as extra input')

rows = []
for n, r in exp.items():
    c = compare(A9, r) if n != 'A9 (reference)' else {}
    rows.append({'experiment': n, 'CV acc %': r['mean'] * 100, 'CV acc with TTA %': np.mean(r['fold_acc_tta']) * 100,
                 'gain vs A9 %': c.get('gain', np.nan), '95% CI ±': c.get('ci', np.nan),
                 'fixed/broke': f"{c['fixed']}/{c['broke']}" if c else '', 'McNemar p': c.get('p_mcnemar', np.nan)})
exp_table = pd.DataFrame(rows); display(exp_table.round(3))
NOISE = abs(exp['noise: A9 with fresh seeds']['mean'] - A9['mean']) * 100
print(f'noise floor (same config, different seeds): {NOISE:.2f} points of CV accuracy')
""")

# --------------------------------------------------------------------------
md("## 6. Evidence for each change")
md(r"""
### 6.1 Learning curves: memorisation before vs after
Train (fixed subset, scored like validation) vs validation, fold 1. A large gap with ~100% train accuracy means the model memorises the training images.
""")
code(r"""
show = [names[0], NOAUG_ROW, AUG_ROW, HIRES_ROW]
fig, axes = plt.subplots(2, len(show), figsize=(4.3 * len(show), 7), squeeze=False)
for j, n in enumerate(show):
    h = results[n]['hists'][0]; tag = n.split(' ', 1)[0]
    axes[0, j].plot(h['epoch'], h['train_loss'], label='train'); axes[0, j].plot(h['epoch'], h['val_loss'], label='val')
    axes[0, j].set_title(f'{tag}: cross-entropy'); axes[0, j].legend()
    axes[1, j].plot(h['epoch'], h['train_acc'], label='train'); axes[1, j].plot(h['epoch'], h['val_acc'], label='val')
    axes[1, j].set_title(f'{tag}: accuracy'); axes[1, j].set_ylim(0, 1.02); axes[1, j].legend(); axes[1, j].set_xlabel('epoch')
plt.tight_layout(); plt.show()
for n in show:
    h = results[n]['hists'][0]
    print(f"{n.split(' ', 1)[0]}: final train acc {h['train_acc'][-1]:.2f}, val acc {h['val_acc'][-1]:.2f}, "
          f"gap {h['train_acc'][-1] - h['val_acc'][-1]:.2f}")
""")

md(r"""
### 6.2 Robustness to small changes in how the leaf lies
The fold-1 models of the row before augmentation and the row with augmentation (same epochs, same everything else) classify the fold-1 validation leaves after a fixed rotation. The crop square is rotation-safe, so no part of the leaf leaves the frame.
""")
code(r"""
def rotate_u8(X, bg, deg):
    x = X.float().div(255); B = len(x); a = math.radians(deg)
    theta = torch.tensor([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0]], dtype=torch.float32).repeat(B, 1, 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False); bgv = bg.view(B, 3, 1, 1)
    return ((F.grid_sample(x - bgv, grid, align_corners=False) + bgv).clamp(0, 1) * 255).round().to(torch.uint8)

angles = list(range(-60, 61, 10))
tr, va = FOLDS[0]
fig, ax = plt.subplots(figsize=(7.5, 4))
for n in (NOAUG_ROW, AUG_ROW):
    r = results[n]; cfg = r['cfg']; model = r['models'][0].to(DEVICE)
    X, bg, L = data_at(cfg)
    acc = [float((predict_proba(model, rotate_u8(X[va], bg[va], d), bg[va], L[va], cfg, model.st).argmax(1) == y_all[va]).mean())
           for d in angles]
    ax.plot(angles, np.array(acc) * 100, marker='o', label=f"{n.split(' ', 1)[0]} ({'no aug' if cfg['aug'] is None else 'aug'})")
    model.cpu()
ax.axvspan(-20, 20, color='g', alpha=.07, label='training rotation range')
ax.set_xlabel('rotation applied to validation leaves (deg)'); ax.set_ylabel('accuracy (%)'); ax.legend(); ax.grid(alpha=.3)
ax.set_title('Accuracy vs rotation (fold 1)'); plt.tight_layout(); plt.show()
""")

md("### 6.3 Where the final configuration still makes mistakes (out-of-fold confusion matrix)")
code(r"""
best = results[TTA_ROW]
pred = best['oof'].argmax(1)
cm = confusion_matrix(y_all, pred, labels=range(N_CLASSES))
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap='Blues'); plt.colorbar(im, fraction=.046)
ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
ax.set_xticklabels(CLASSES, rotation=90, fontsize=8); ax.set_yticklabels(CLASSES, fontsize=8)
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=7, color='w' if cm[i, j] > cm.max() / 2 else 'k')
ax.set_xlabel('predicted'); ax.set_ylabel('true'); ax.set_title('Out-of-fold predictions, A10')
plt.tight_layout(); plt.show()
per_class = pd.Series(cm.diagonal() / cm.sum(1), index=CLASSES).sort_values()
print('per-class accuracy, worst first (%):'); print((per_class * 100).round(1).head(8).to_string())
off = sorted([(int(cm[i, j]), CLASSES[i], CLASSES[j]) for i in range(N_CLASSES) for j in range(N_CLASSES) if i != j and cm[i, j]], reverse=True)
print('most confused (count, true -> predicted):', off[:6])
if off:
    _, a, b = off[0]
    fig, axes = plt.subplots(1, 6, figsize=(14, 2.8))
    ia, ib = np.where(y_all == CLS2IDX[a])[0][:3], np.where(y_all == CLS2IDX[b])[0][:3]
    for ax, k in zip(axes, list(ia) + list(ib)):
        ax.imshow(CROP[k].permute(1, 2, 0)); ax.set_title(CLASSES[y_all[k]], fontsize=8); ax.axis('off')
    plt.suptitle(f'The most confused pair: {a} vs {b}'); plt.tight_layout(); plt.show()
""")

md(r"""
### 6.4 What the network looks at (Grad-CAM)
Grad-CAM weights a convolutional feature map by the gradient of the predicted class score. It is computed on the third residual stage (12×12 at 192 px) of the fold-1 A9 model. The number printed below is the share of the heat that falls inside the leaf mask, averaged over all fold-1 validation leaves. If it is well above the leaf's share of the image, the model is using the leaf rather than the background or scan artefacts.
""")
code(r"""
def grad_cam(model, x, size, cls, layer):
    model.eval(); feats = {}
    h = layer.register_forward_hook(lambda m, i, o: feats.__setitem__('a', o))
    logits = model(x, size); a = feats['a']; h.remove()
    g = torch.autograd.grad(logits[torch.arange(len(x)), cls].sum(), a)[0]
    cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)
    return (cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8))[:, 0].detach().cpu()

r = results[HIRES_ROW]; cfg = r['cfg']; model = r['models'][0].to(DEVICE).float()
tr, va = FOLDS[0]
X, bg, L = data_at(cfg)
cams, preds = [], []
for i in range(0, len(va), 32):
    vi = va[i:i + 32]
    xin = prep(X[vi], bg[vi], cfg, model.st, train=False)
    s = ((L[vi] - model.st.lmean) / model.st.lstd).to(DEVICE)
    with torch.enable_grad():
        p = model(xin, s).argmax(1); cams.append(grad_cam(model, xin, s, p, model.blocks[2])); preds.append(p.cpu())
cams, preds = torch.cat(cams), torch.cat(preds)
Xv = X[va].float().div(255)
leaf = ((Xv - bg[va].view(-1, 3, 1, 1)).abs().amax(1) > THR / 255).float()
inside = (cams * leaf).sum((1, 2)) / (cams.sum((1, 2)) + 1e-8)
print(f'share of Grad-CAM heat on the leaf: {inside.mean():.2f} (leaf covers {leaf.mean():.2f} of the image)')
fig, axes = plt.subplots(2, 6, figsize=(16, 6))
for k, ax in enumerate(axes.flat):
    ax.imshow(Xv[k].permute(1, 2, 0)); ax.imshow(cams[k], cmap='jet', alpha=.4); ax.axis('off')
    ok = preds[k].item() == y_all[va][k]
    ax.set_title(f"{'✓' if ok else '✗'} {CLASSES[preds[k]]}", fontsize=8, color='g' if ok else 'r')
plt.suptitle('Grad-CAM, fold-1 validation leaves (A9 model, stage-3 features)'); plt.tight_layout(); plt.show()
_ = model.cpu()
""")

# --------------------------------------------------------------------------
md(r"""
## 7. Final model and test predictions
The final configuration takes A9, plus each choice from 5.2 whose gain is larger than the noise floor. It is cross-validated once more with **fresh seeds**, which gives an unbiased estimate of accuracy after choosing settings on the earlier runs. The test prediction averages the softmax outputs of an ensemble: the 5 cross-validation models plus `N_FULL` models trained on all 600 images (different seeds), each with the test-time views that match its augmentation.
""")
code(r"""
FINAL = copy.deepcopy(steps[-1][1])
for key, cand in [('aug', ['rotation +-45', 'rotation +-180']), ('res', ['res 128', 'res 256']), ('use_size', ['+ leaf size input'])]:
    best_c = max(cand, key=lambda n: np.mean(exp[n]['fold_acc_tta']))
    gain = (np.mean(exp[best_c]['fold_acc_tta']) - np.mean(A9['fold_acc_tta'])) * 100
    if gain > NOISE:
        FINAL[key] = copy.deepcopy(exp[best_c]['cfg'][key])
        print(f'{key}: adopt {best_c!r} (+{gain:.2f} > noise {NOISE:.2f})')
    else:
        print(f'{key}: keep A9 setting ({best_c!r} gain {gain:+.2f} <= noise {NOISE:.2f})')
print('FINAL config:', {k: FINAL[k] for k in ('res', 'crop', 'epochs', 'use_size')}, 'rotation', FINAL['aug']['rot_deg'])

final_cv = cross_validate(FINAL, 'FINAL, fresh seeds', keep_models=True, seed_offset=2000)
print(f"FINAL CV accuracy with TTA: {np.mean(final_cv['fold_acc_tta']) * 100:.1f}% "
      f"(folds {[round(a * 100, 1) for a in final_cv['fold_acc_tta']]})")

N_FULL = 0 if QUICK else 3
ensemble = list(final_cv['models'])
for s in range(N_FULL):
    m, _ = train_model(quick(FINAL), np.arange(NTR), seed=3000 + s); ensemble.append(m.cpu())
    print(f'full-data model {s + 1}/{N_FULL} trained')

X, bg, L = data_at(quick(FINAL))
te = np.arange(NTR, len(X))
probs = []
for m in ensemble:
    m.to(DEVICE); probs.append(predict_proba(m, X[te], bg[te], L[te], quick(FINAL), m.st, tta=True)); m.cpu()
test_prob = np.mean(probs, 0); test_pred = test_prob.argmax(1)
agree = np.mean([(p.argmax(1) == test_pred).mean() for p in probs])
print(f'{len(ensemble)} models; single models agree with the ensemble on {agree * 100:.1f}% of test images')
""")
code(r"""
# predictions.csv: one row per test image (columns follow the handout / template if one was provided)
if SUB_TEMPLATE is not None:
    tmpl, idcol = SUB_TEMPLATE
    name2pred = {p.name: CLASSES[k] for p, k in zip(test_paths, test_pred)} | {p.stem: CLASSES[k] for p, k in zip(test_paths, test_pred)}
    sub = tmpl.copy(); labcol = [c for c in sub.columns if c != idcol][0]
    sub[labcol] = sub[idcol].astype(str).map(name2pred)
    assert sub[labcol].notna().all(), 'some template rows did not match a test image'
else:
    sub = pd.DataFrame({'filename': [p.name for p in test_paths], 'label': [CLASSES[k] for k in test_pred]})
sub.to_csv('predictions.csv', index=False)
print(sub.shape); display(sub.head())
print('predicted class counts:', sub.iloc[:, -1].value_counts().sort_index().to_dict())
print(f'total runtime: {(time.time() - T0) / 60:.1f} min')
""")

md(r"""
## 8. Summary
*(Filled in from the results above: baseline accuracy, what was wrong, what each change did and the evidence, final CV accuracy.)*
""")

# ==========================================================================
nb = nbf.v4.new_notebook()
nb.cells = CELLS
nb.metadata = {
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}
out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "LeafIDMir.ipynb")
nbf.write(nb, out)
print("wrote", out, f"({len(CELLS)} cells)")
