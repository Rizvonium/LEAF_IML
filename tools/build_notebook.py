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


# --------------------------------------------------------------------------
md(r"""
# LeafID: identifying tree species from scanned leaves (CNN trained from scratch)

**Task.** 600 labelled leaf scans from 15 species (40 per species) and 525 unlabelled test scans.
The starter notebook trains a small CNN that performs poorly. This notebook
1. runs the starter as given to get a baseline,
2. diagnoses why it performs poorly,
3. fixes and improves it **one change at a time**, measuring each change with the same 5-fold stratified cross-validation, and
4. trains the final model and writes `predictions.csv`.

No pretrained weights or external data are used: every network here is trained from random initialisation on the 600 labelled images only.

**Runtime:** Colab, T4 GPU. Runtime → Change runtime type → T4 GPU, upload `leaves.zip`, then Runtime → Run all.
""")

md("## 0. Setup")
code(r"""
import os, time, json, math, random, zipfile, copy
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score

QUICK = os.environ.get('LEAF_QUICK') == '1'   # tiny smoke-test settings, used only for local CPU testing
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', DEVICE, '| torch', torch.__version__, '| QUICK =', QUICK)
if DEVICE.type == 'cuda':
    print(torch.cuda.get_device_name(0))
torch.backends.cudnn.benchmark = True

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

T0 = time.time()
""")

md(r"""
## 1. Load the data

Upload `leaves.zip` to the Colab session (Files panel) before running. The cell below unzips it and finds the labelled training images and the unlabelled test images.
""")
code(r"""
ZIP_PATH = Path(os.environ.get('LEAF_ZIP', '/content/leaves.zip'))
DATA_DIR = Path(os.environ.get('LEAF_DIR', '/content/leaves_data'))

if not ZIP_PATH.exists():
    try:
        from google.colab import files
        print('leaves.zip not found in /content, please choose it:')
        uploaded = files.upload()
        ZIP_PATH = Path(next(iter(uploaded))).resolve()
    except ImportError:
        raise FileNotFoundError(f'{ZIP_PATH} not found')

if not DATA_DIR.exists():
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(DATA_DIR)

IMG_EXT = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
all_imgs = sorted(p for p in DATA_DIR.rglob('*')
                  if p.suffix.lower() in IMG_EXT and '__MACOSX' not in p.parts and not p.name.startswith('.'))
is_test = lambda p: any(part.lower().startswith('test') for part in p.relative_to(DATA_DIR).parts[:-1])
train_paths = [p for p in all_imgs if not is_test(p)]
test_paths  = [p for p in all_imgs if is_test(p)]

# labels: from a CSV if one is provided, otherwise from the parent folder name
csvs = [p for p in DATA_DIR.rglob('*.csv') if '__MACOSX' not in p.parts]
if csvs:
    print('CSV files found:', [str(c.relative_to(DATA_DIR)) for c in csvs])
train_labels_str = [p.parent.name for p in train_paths]

CLASSES = sorted(set(train_labels_str))
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
y_all = np.array([CLS2IDX[s] for s in train_labels_str])
N_CLASSES = len(CLASSES)
print(f'{len(train_paths)} labelled images, {len(test_paths)} test images, {N_CLASSES} classes')
print(pd.Series(train_labels_str).value_counts().sort_index().to_string())
""")

md("### 1.1 What do the images look like?")
code(r"""
sizes = pd.DataFrame([Image.open(p).size for p in train_paths + test_paths], columns=['w', 'h'])
sizes['split'] = ['train'] * len(train_paths) + ['test'] * len(test_paths)
print(sizes.groupby('split').describe().T.round(0))

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

md(r"""
## 4. Improved pipeline

### 4.1 Preprocessing
Scans differ in size and aspect ratio. Squashing them to a fixed square distorts leaf shape differently for each image, and leaf shape (length:width ratio, margin, tip) is the main cue for species. So each scan is
1. optionally cropped to the leaf's bounding box (found by thresholding against the background colour),
2. padded to a square with its own background colour (aspect ratio preserved), and
3. resized with anti-aliasing.

Images are decoded once at `BASE_RES` and kept in memory; lower resolutions are derived from that copy.
""")
code(r"""
BASE_RES = 96 if QUICK else 384

def border_bg(arr):
    b = np.concatenate([arr[:3].reshape(-1, 3), arr[-3:].reshape(-1, 3),
                        arr[:, :3].reshape(-1, 3), arr[:, -3:].reshape(-1, 3)])
    return np.median(b, axis=0)

def leaf_bbox(arr, bg, thr=40, margin=0.05):
    diff = np.abs(arr.astype(np.int16) - bg.astype(np.int16)[None, None]).max(axis=2)
    ys, xs = np.nonzero(diff > thr)
    H, W = diff.shape
    if len(ys) < 100:
        return 0, 0, W, H
    y0, y1 = np.percentile(ys, [0.1, 99.9]); x0, x1 = np.percentile(xs, [0.1, 99.9])
    m = margin * max(y1 - y0, x1 - x0)
    return (int(max(0, x0 - m)), int(max(0, y0 - m)), int(min(W, x1 + m + 1)), int(min(H, y1 + m + 1)))

def load_square(path, res, crop):
    img = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    img.thumbnail((1024, 1024), Image.BILINEAR)            # cheap first downscale of large scans
    arr = np.asarray(img)
    bg = border_bg(arr)
    if crop:
        x0, y0, x1, y1 = leaf_bbox(arr, bg)
        arr = arr[y0:y1, x0:x1]
    h, w = arr.shape[:2]; s = max(h, w)
    canvas = np.empty((s, s, 3), np.uint8); canvas[:] = bg.astype(np.uint8)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = arr
    out = Image.fromarray(canvas).resize((res, res), Image.LANCZOS)
    return np.asarray(out), bg

_cache = {}
def get_arrays(crop):
    # uint8 tensors (N,3,BASE_RES,BASE_RES) and background colours (N,3) in [0,1]
    if crop not in _cache:
        t = time.time()
        def load(paths):
            out = [load_square(p, BASE_RES, crop) for p in paths]
            X = torch.from_numpy(np.stack([o[0] for o in out])).permute(0, 3, 1, 2).contiguous()
            bg = torch.tensor(np.stack([o[1] for o in out]), dtype=torch.float32) / 255
            return X, bg
        _cache[crop] = (*load(train_paths), *load(test_paths))
        print(f'decoded {len(train_paths) + len(test_paths)} images (crop={crop}) in {time.time() - t:.0f}s')
    return _cache[crop]

def at_res(X, res):
    # X uint8 (N,3,R,R) -> uint8 (N,3,res,res), anti-aliased
    if X.shape[-1] == res:
        return X
    out = [F.interpolate(X[i:i + 64].float(), size=(res, res), mode='bilinear', antialias=True, align_corners=False)
           for i in range(0, len(X), 64)]
    return torch.cat(out).round().clamp(0, 255).to(torch.uint8)

Xtr_c, bgtr_c, Xte_c, bgte_c = get_arrays(crop=True)
fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))
for i, ax in enumerate(axes.flat):
    j = i * (len(Xtr_c) // 16)
    ax.imshow(Xtr_c[j].permute(1, 2, 0)); ax.set_title(CLASSES[y_all[j]], fontsize=7); ax.axis('off')
plt.suptitle('After crop + pad-to-square'); plt.tight_layout(); plt.show()
""")

md(r"""
### 4.2 Augmentation (on the GPU)
A scanned leaf has no fixed orientation, mirror image or exact scale, so the label should not change under rotation, flipping or small zoom/shift. Each batch gets a random affine warp (rotation, flip, scale, translation). The area exposed by the warp is filled with that image's own background colour, so the model does not learn black corners as a feature. Small brightness/contrast jitter covers scanner variation.
""")
code(r"""
def gpu_augment(x, bg, a):
    # x: (B,3,H,W) float in [0,1]; bg: (B,3) in [0,1]; a: augmentation settings dict
    B, dev = x.size(0), x.device
    ang = (torch.rand(B, device=dev) * 2 - 1) * math.radians(a['rot_deg'])
    sc = torch.empty(B, device=dev).uniform_(*a['scale'])
    fl = (torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0) if a['flip']
          else torch.ones(B, device=dev))
    tx, ty = (torch.rand(2, B, device=dev) * 2 - 1) * a['translate']
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos * fl, -sin, tx], 1),
                         torch.stack([sin * fl,  cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    bgv = bg.view(B, 3, 1, 1)
    x = F.grid_sample(x - bgv, grid, mode='bilinear', padding_mode='zeros', align_corners=False) + bgv
    if a['jitter'] > 0:
        b = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * a['jitter']
        c = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * a['jitter']
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        x = ((x - m) * c + m) * b
    return x.clamp(0, 1)

NO_AUG = dict(rot_deg=0, scale=(1, 1), flip=False, translate=0.0, jitter=0.0)
FULL_AUG = dict(rot_deg=180, scale=(0.85, 1.15), flip=True, translate=0.08, jitter=0.15)

_x = Xtr_c[:8].float().div(255); _bg = bgtr_c[:8]
_xa = gpu_augment(_x.repeat(2, 1, 1, 1), _bg.repeat(2, 1), FULL_AUG)
fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))
for i, ax in enumerate(axes.flat):
    ax.imshow(_xa[i].permute(1, 2, 0)); ax.axis('off')
plt.suptitle('Random augmentations of the first 8 training images (two draws each)'); plt.tight_layout(); plt.show()
""")

md(r"""
### 4.3 Models
* `plain`: conv → ReLU → max-pool blocks, flattened into a large fully connected layer, no normalisation (the starter's design style).
* `leafnet`: a small residual CNN trained from scratch. Batch normalisation after every convolution keeps activations well scaled, so a higher learning rate trains stably. Residual connections ease optimisation. Global average pooling replaces the huge flatten→dense layer. On 600 images, that dense layer holds most of the parameters and is where the network memorises the training set.
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
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c * s * s, hidden), nn.ReLU(inplace=True),
                                  nn.Linear(hidden, n_classes))
    def forward(self, x):
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
        out = F.relu(self.b1(self.c1(x)), inplace=True)
        return F.relu(self.b2(self.c2(out)) + self.short(x), inplace=True)

class LeafNet(nn.Module):
    def __init__(self, n_classes, widths=(32, 64, 128, 256), depth=1, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(conv_bn(3, widths[0], 2), conv_bn(widths[0], widths[0]), nn.MaxPool2d(2))
        blocks, c = [], widths[0]
        for i, w in enumerate(widths):
            for d in range(depth):
                blocks.append(ResBlock(c, w, 2 if (i > 0 and d == 0) else 1)); c = w
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, n_classes))
    def features(self, x):
        return self.blocks(self.stem(x))
    def forward(self, x):
        return self.head(self.features(x))

def build_model(cfg):
    if cfg['arch'] == 'plain':
        return PlainCNN(N_CLASSES, cfg['res'])
    return LeafNet(N_CLASSES, widths=cfg.get('widths', (32, 64, 128, 256)), depth=cfg.get('depth', 1),
                   dropout=cfg.get('dropout', 0.3))

n_params = lambda m: sum(p.numel() for p in m.parameters())
print('plain   params:', f"{n_params(PlainCNN(N_CLASSES, 128)):,}", '(at 128px)')
print('leafnet params:', f"{n_params(LeafNet(N_CLASSES)):,}")
""")

md(r"""
### 4.4 Training and evaluation
* **5-fold stratified cross-validation** with the same folds for every configuration. With 40 images per species, one 80/20 split gives only 8 validation images per species, and a single split's accuracy can swing by several points by chance. Five folds use all 600 images for validation once and give a mean ± spread. Because the folds are identical, configurations can be compared fold by fold.
* Inputs are scaled to [0, 1] and standardised with per-channel mean/std computed from the labelled training images.
* The model is evaluated **after the last epoch**, not at its best validation epoch, so the reported accuracy is not tuned on the validation fold.
""")
code(r"""
BASE = dict(arch='leafnet', res=192, crop=True, epochs=120, bs=32, opt='adamw', lr=2e-3, wd=0.05,
            sched='onecycle', label_smoothing=0.1, aug=FULL_AUG, normalize=True, tta=False, dropout=0.3)
N_FOLDS = 5
if QUICK:
    BASE.update(res=48, epochs=2); N_FOLDS = 2

def channel_stats(X):
    x = X.float().div(255)
    return x.mean(dim=(0, 2, 3)), x.std(dim=(0, 2, 3))

def prep_batch(xb, bgb, cfg, mean, std, train):
    x = xb.to(DEVICE, non_blocking=True).float().div_(255)
    if train and cfg['aug'] is not None:
        x = gpu_augment(x, bgb.to(DEVICE), cfg['aug'])
    if cfg['normalize']:
        x = (x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
    return x.contiguous(memory_format=torch.channels_last)

def make_opt(model, cfg):
    if cfg['opt'] == 'adamw':
        decay = [p for n, p in model.named_parameters() if p.ndim > 1]
        no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1]
        return torch.optim.AdamW([{'params': decay, 'weight_decay': cfg['wd']},
                                  {'params': no_decay, 'weight_decay': 0}], lr=cfg['lr'])
    if cfg['opt'] == 'adam':
        return torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    return torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, nesterov=True, weight_decay=5e-4)

@torch.no_grad()
def predict_proba(model, X, bg, cfg, mean, std, tta=False, bs=64):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        x = prep_batch(X[i:i + bs], bg[i:i + bs], cfg, mean, std, train=False)
        views = [x] if not tta else [torch.rot90(v, k, dims=(2, 3)) for v in (x, x.flip(3)) for k in range(4)]
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == 'cuda'):
            p = torch.stack([F.softmax(model(v).float(), 1) for v in views]).mean(0)
        out.append(p.cpu())
    return torch.cat(out).numpy()

def train_model(cfg, X, y, bg, Xva=None, yva=None, bgva=None, seed=0, eval_every=5):
    seed_everything(seed)
    model = build_model(cfg).to(DEVICE).to(memory_format=torch.channels_last)
    mean, std = channel_stats(X); mean, std = mean.to(DEVICE), std.to(DEVICE)
    opt = make_opt(model, cfg)
    steps = math.ceil(len(X) / cfg['bs']); total = steps * cfg['epochs']
    sched = (torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg['lr'], total_steps=total, pct_start=0.15)
             if cfg['sched'] == 'onecycle' else None)
    scaler = torch.amp.GradScaler('cuda', enabled=DEVICE.type == 'cuda')
    crit = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
    yt = torch.as_tensor(y)
    hist = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for ep in range(cfg['epochs']):
        model.train(); perm = torch.randperm(len(X)); tl = tc = 0
        for i in range(0, len(X), cfg['bs']):
            idx = perm[i:i + cfg['bs']]
            if len(idx) < 2: continue                      # BatchNorm needs >1 sample
            xb = prep_batch(X[idx], bg[idx], cfg, mean, std, train=True); yb = yt[idx].to(DEVICE)
            with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == 'cuda'):
                logits = model(xb); loss = crit(logits.float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if sched: sched.step()
            tl += loss.item() * len(idx); tc += (logits.argmax(1) == yb).sum().item()
        if Xva is not None and ((ep + 1) % eval_every == 0 or ep == cfg['epochs'] - 1):
            p = predict_proba(model, Xva, bgva, cfg, mean, std)
            hist['epoch'].append(ep + 1); hist['train_loss'].append(tl / len(X)); hist['train_acc'].append(tc / len(X))
            hist['val_loss'].append(float(F.nll_loss(torch.log(torch.as_tensor(p) + 1e-8), torch.as_tensor(yva))))
            hist['val_acc'].append(float((p.argmax(1) == yva).mean()))
    model.norm_stats = (mean, std)
    return model, hist

FOLDS = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=42).split(np.zeros(len(y_all)), y_all))

def cross_validate(cfg, name='', keep_models=False, verbose=True):
    t = time.time()
    Xtr, bgtr, _, _ = get_arrays(cfg['crop']); Xtr = at_res(Xtr, cfg['res'])
    oof = np.zeros((len(y_all), N_CLASSES)); accs, hists, models = [], [], []
    for f, (tr, va) in enumerate(FOLDS):
        model, h = train_model(cfg, Xtr[tr], y_all[tr], bgtr[tr], Xtr[va], y_all[va], bgtr[va], seed=f)
        p = predict_proba(model, Xtr[va], bgtr[va], cfg, *model.norm_stats, tta=cfg['tta'])
        oof[va] = p; accs.append(float((p.argmax(1) == y_all[va]).mean())); hists.append(h)
        if keep_models: models.append(model.cpu())
        del model
    r = dict(name=name, cfg=cfg, fold_acc=accs, mean=float(np.mean(accs)), std=float(np.std(accs)),
             macro_f1=float(f1_score(y_all, oof.argmax(1), average='macro')), oof=oof, hists=hists,
             models=models, secs=time.time() - t)
    if verbose:
        print(f"{name:42s} acc {r['mean']*100:5.1f} ± {r['std']*100:4.1f}  "
              f"folds {[round(a*100, 1) for a in accs]}  ({r['secs']:.0f}s)")
    return r
""")

md(r"""
## 5. Ablation: one change at a time

Each row adds **one** change to the row above, and every row uses the same five folds. The order follows the diagnosis: correctness fixes first, then generalisation, then the final extras.
""")
code(r"""
def cfg_with(base, **kw):
    c = copy.deepcopy(base); c.update(kw); return c

STARTER_LIKE = cfg_with(BASE, arch='plain', res=64 if not QUICK else 32, crop=False, epochs=30 if not QUICK else 2,
                        opt='adam', lr=1e-3, sched=None, label_smoothing=0.0, aug=None, normalize=False)
steps = [('A0 starter-like (plain CNN, 64px, no aug)', STARTER_LIKE)]
def add(name, **kw):
    steps.append((name, cfg_with(steps[-1][1], **kw)))

add('A1 + input standardisation', normalize=True)
add('A2 + crop to leaf, pad to square', crop=True)
add('A3 + BatchNorm/residual/GAP net (LeafNet)', arch='leafnet')
add('A4 + rotation/flip/scale augmentation', aug=FULL_AUG, epochs=BASE['epochs'])
add('A5 + AdamW + one-cycle LR + label smoothing', opt='adamw', lr=BASE['lr'], sched='onecycle', label_smoothing=0.1)
add('A6 + higher resolution (192px)', res=BASE['res'])
add('A7 + test-time augmentation (8 views)', tta=True)

results = {}
for name, cfg in steps:
    results[name] = cross_validate(cfg, name, keep_models=name.startswith(('A3', 'A4')))
""")
code(r"""
abl = pd.DataFrame([{'step': n, 'cv_acc_%': r['mean'] * 100, 'std_%': r['std'] * 100, 'macro_F1': r['macro_f1'],
                     'gain_vs_prev_%': np.nan, 'folds_improved': '', 'secs': round(r['secs'])}
                    for n, r in results.items()])
names = list(results)
for i in range(1, len(names)):
    a, b = np.array(results[names[i - 1]]['fold_acc']), np.array(results[names[i]]['fold_acc'])
    abl.loc[i, 'gain_vs_prev_%'] = (b - a).mean() * 100
    abl.loc[i, 'folds_improved'] = f'{int((b > a).sum())}/{len(a)}'
display(abl.round(2))

fig, ax = plt.subplots(figsize=(11, 4.5))
m = [results[n]['mean'] * 100 for n in names]; s = [results[n]['std'] * 100 for n in names]
ax.bar(range(len(names)), m, yerr=s, capsize=4, color='#4C72B0', alpha=.8)
for i, n in enumerate(names):
    ax.scatter([i] * N_FOLDS, np.array(results[n]['fold_acc']) * 100, color='k', s=10, zorder=3)
ax.set_xticks(range(len(names))); ax.set_xticklabels([n.split(' ', 1)[0] for n in names])
ax.set_ylabel('5-fold CV accuracy (%)'); ax.set_title('Ablation: each bar adds one change (dots = folds)')
ax.grid(axis='y', alpha=.3); plt.tight_layout(); plt.show()
""")

md("## 6. Evidence for each change")
md(r"""
### 6.1 Learning curves: overfitting before vs after
If the training accuracy reaches ~100% while the validation loss rises, the model is memorising. Each panel shows the fold-1 curves for a step of the ablation.
""")
code(r"""
show = [n for n in names if n.split(' ', 1)[0] in ('A0', 'A3', 'A4', 'A6')]
fig, axes = plt.subplots(2, len(show), figsize=(4.2 * len(show), 7), squeeze=False)
for j, n in enumerate(show):
    h = results[n]['hists'][0]
    axes[0, j].plot(h['epoch'], h['train_loss'], label='train'); axes[0, j].plot(h['epoch'], h['val_loss'], label='val')
    axes[0, j].set_title(n.split(' ', 1)[0] + ' loss'); axes[0, j].legend()
    axes[1, j].plot(h['epoch'], h['train_acc'], label='train'); axes[1, j].plot(h['epoch'], h['val_acc'], label='val')
    axes[1, j].set_title(n.split(' ', 1)[0] + ' accuracy'); axes[1, j].set_ylim(0, 1.02); axes[1, j].legend()
plt.tight_layout(); plt.show()
""")

md(r"""
### 6.2 Why rotation augmentation helps: accuracy vs rotation angle
Validation images from fold 1 are rotated by a fixed angle, and the fold-1 models from A3 (no augmentation) and A4 (with augmentation) classify them. A model that has only seen leaves in one orientation should lose accuracy as the leaf is turned.
""")
code(r"""
def rotate_batch(X, bg, deg):
    x = X.float().div(255); B = len(x); a = math.radians(deg)
    theta = torch.tensor([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0]], dtype=torch.float32).repeat(B, 1, 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False); bgv = bg.view(B, 3, 1, 1)
    x = F.grid_sample(x - bgv, grid, align_corners=False) + bgv
    return (x.clamp(0, 1) * 255).round().to(torch.uint8)

angles = list(range(0, 361, 30))
tr, va = FOLDS[0]
fig, ax = plt.subplots(figsize=(7, 4))
for n in [k for k in names if k.startswith(('A3', 'A4'))]:
    r = results[n]; cfg = r['cfg']; model = r['models'][0].to(DEVICE)
    Xtr, bgtr, _, _ = get_arrays(cfg['crop']); Xv = at_res(Xtr, cfg['res'])[va]; bgv = bgtr[va]
    acc = [float((predict_proba(model, rotate_batch(Xv, bgv, d), bgv, cfg, *model.norm_stats).argmax(1) == y_all[va]).mean())
           for d in angles]
    ax.plot(angles, np.array(acc) * 100, marker='o', label=n.split(' ', 1)[0] + (' (no aug)' if 'A3' in n else ' (+ aug)'))
    model.cpu()
ax.set_xlabel('rotation applied to validation images (degrees)'); ax.set_ylabel('accuracy (%)')
ax.set_xticks(angles); ax.legend(); ax.grid(alpha=.3); ax.set_title('Robustness to rotation (fold 1)')
plt.tight_layout(); plt.show()
""")

md("### 6.3 Where the final model still makes mistakes (out-of-fold confusion matrix)")
code(r"""
best_name = names[-1]; best = results[best_name]
pred = best['oof'].argmax(1)
cm = confusion_matrix(y_all, pred, labels=range(N_CLASSES))
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap='Blues'); plt.colorbar(im, fraction=.046)
ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
ax.set_xticklabels(CLASSES, rotation=90, fontsize=8); ax.set_yticklabels(CLASSES, fontsize=8)
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        if cm[i, j]: ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=7, color='w' if cm[i, j] > cm.max() / 2 else 'k')
ax.set_xlabel('predicted'); ax.set_ylabel('true'); ax.set_title(f'{best_name.split(" ", 1)[0]}: out-of-fold predictions')
plt.tight_layout(); plt.show()
per_class = pd.Series(cm.diagonal() / cm.sum(1), index=CLASSES).sort_values()
print('per-class OOF accuracy (worst first):'); print((per_class * 100).round(1).to_string())
off = [(cm[i, j], CLASSES[i], CLASSES[j]) for i in range(N_CLASSES) for j in range(N_CLASSES) if i != j and cm[i, j]]
print('most confused (count, true -> predicted):', sorted(off, reverse=True)[:6])
""")

md(r"""
### 6.4 What the network looks at (Grad-CAM)
Grad-CAM weights the last convolutional feature maps by the gradient of the predicted class score. If the heat sits on the leaf blade and margin, and not on the background, scanner artefacts or the image border, the model is using the leaf itself.
""")
code(r"""
def grad_cam(model, x, cls):
    model.eval(); feats = {}
    h = model.blocks.register_forward_hook(lambda m, i, o: feats.__setitem__('a', o))
    x = x.clone().requires_grad_(False)
    logits = model(x); a = feats['a']; h.remove()
    g = torch.autograd.grad(logits[torch.arange(len(x)), cls].sum(), a)[0]
    cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)
    cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)
    return cam[:, 0].detach().cpu()

r = results[[k for k in names if k.startswith('A4')][0]]; cfg = r['cfg']; model = r['models'][0].to(DEVICE).float()
Xtr, bgtr, _, _ = get_arrays(cfg['crop']); Xv = at_res(Xtr, cfg['res'])[va][:12]; bgv = bgtr[va][:12]
xin = prep_batch(Xv, bgv, cfg, *model.norm_stats, train=False).float()
with torch.enable_grad():
    pred = model(xin).argmax(1)
    cams = grad_cam(model, xin, pred)
fig, axes = plt.subplots(2, 6, figsize=(16, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(Xv[i].permute(1, 2, 0)); ax.imshow(cams[i], cmap='jet', alpha=.4); ax.axis('off')
    ok = pred[i].item() == y_all[va][i]
    ax.set_title(f"{'✓' if ok else '✗'} {CLASSES[pred[i]]}", fontsize=8, color='g' if ok else 'r')
plt.suptitle('Grad-CAM on fold-1 validation images (A4 model)'); plt.tight_layout(); plt.show()
model.cpu()
""")

md(r"""
## 7. Final model and test predictions
The final configuration is the last ablation row. The test prediction averages the softmax outputs of an ensemble, each with 8-view TTA (4 rotations × mirror):
* the 5 cross-validation models (each trained on 4/5 of the data, with a known out-of-fold accuracy), plus
* `N_FULL` models trained on all 600 labelled images with different seeds.
""")
code(r"""
FINAL = copy.deepcopy(results[best_name]['cfg'])
N_FULL = 0 if QUICK else 3
final_cv = cross_validate(FINAL, 'FINAL (CV models kept for ensemble)', keep_models=True)
Xtr, bgtr, Xte, bgte = get_arrays(FINAL['crop'])
Xtr, Xte = at_res(Xtr, FINAL['res']), at_res(Xte, FINAL['res'])

ensemble = list(final_cv['models'])
for s in range(N_FULL):
    m, _ = train_model(FINAL, Xtr, y_all, bgtr, seed=100 + s); ensemble.append(m.cpu())
    print(f'full-data model {s + 1}/{N_FULL} trained')

probs = []
for m in ensemble:
    m.to(DEVICE); probs.append(predict_proba(m, Xte, bgte, FINAL, *m.norm_stats, tta=True)); m.cpu()
test_prob = np.mean(probs, 0)
test_pred = test_prob.argmax(1)
agree = np.mean([(p.argmax(1) == test_pred).mean() for p in probs])
print(f'{len(ensemble)} models; mean agreement of single models with the ensemble: {agree * 100:.1f}%')
""")
code(r"""
# predictions.csv: one row per test image (format to match the handout)
sub = pd.DataFrame({'filename': [p.name for p in test_paths], 'label': [CLASSES[i] for i in test_pred]})
sub.to_csv('predictions.csv', index=False)
print(sub.shape); display(sub.head())
print(sub['label'].value_counts().sort_index().to_string())
print(f'total runtime: {(time.time() - T0) / 60:.1f} min')
""")

md(r"""
## 8. Summary
*(Filled in from the results above: the baseline accuracy, what was wrong, what each change did with evidence, the final CV accuracy.)*
""")

# --------------------------------------------------------------------------
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
