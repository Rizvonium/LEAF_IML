"""Build LeafID_Rizwan_Mir.ipynb from the cell sources below.

    python tools/build_notebook.py        # writes LeafID_Rizwan_Mir.ipynb

The notebook is the deliverable. This script only exists so the cells can be
edited as plain Python and the notebook regenerated. Never regenerate over an
executed copy you want to keep: the generated notebook has no outputs.
"""
import json
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
CELLS = []


def md(s):
    CELLS.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    CELLS.append(nbf.v4.new_code_cell(s.strip("\n")))


# ==========================================================================
md(r"""
# Leaf species recognition with CNNs (Assignment 2)

**Goal.** Predict which of 15 tree species a scanned leaf comes from: 600 labelled training images (40 per species) and 525 unlabelled test images, all 128×128 RGB.

**Plan**
1. Run the starter notebook **unchanged** and record its results (plus the same recipe with 3 seeds).
2. Data: class counts, image sizes, and a look at the training and test images. Key finding: **training leaves are upright, test leaves are rotated about the image centre to any angle.**
3. Why the starter performs poorly.
4. Shared code: validation split, a rotated (test-like) validation set, augmentation, models, training loop.
5. Experiment log: fix the starter's small CNN **one change per row**, 3 random seeds per row.
6. An ImageNet-pretrained ResNet-18 (allowed by the handout) with the same fixes.
7. Final model: 3 seeds, mean ± sd; confusion matrix; misclassified examples; `predictions.csv` (header `id,label`, 525 rows).
8. Summary of results; references.

Every accuracy is given as **mean ± sample standard deviation over 3 seeds**, unless a line says otherwise.

**Runtime:** Colab → Runtime → Change runtime type → **T4 GPU**, upload `leaves.zip` to `/content`, then Runtime → **Run all**. (On Kaggle: add `leaves.zip` as an input dataset, choose a GPU T4 accelerator with Internet on; the first cell copies the data to `/content`.)
""")

code(r"""
import time, os, glob, shutil
T_START = time.time()      # total run time is printed at the end of the notebook

# Only on Kaggle: the uploaded dataset appears under /kaggle/input. Copy it to /content, where the starter
# expects it (as leaves.zip, or as the already-unzipped folder holding train_labels.csv). Does nothing on Colab.
if os.path.isdir('/kaggle/input') and not os.path.isdir('/content/leaves'):
    os.makedirs('/content', exist_ok=True)
    zips = glob.glob('/kaggle/input/**/leaves.zip', recursive=True)
    if zips:
        shutil.copy(zips[0], '/content/leaves.zip')
    else:
        shutil.copytree(os.path.dirname(glob.glob('/kaggle/input/**/train_labels.csv', recursive=True)[0]), '/content/leaves')
    print('Kaggle: data copied to /content')
""")

# --------------------------------------------------------------------------
md(r"""
## 1. The starter notebook, unchanged
The next six cells are the starter notebook's code cells, copied without any change. (They write a `predictions.csv`; the final model overwrites it at the end.)
""")
STARTER = json.load(open(ROOT / "starter" / "LeafID_starter.ipynb"))
for c in STARTER["cells"]:
    if c["cell_type"] == "code":
        code("".join(c["source"]))

code(r"""
# Record the starter's numbers before changing anything
print(f'Starter (unchanged, one run): train accuracy {accuracy(model, X_tr, y_tr):.3f}, '
      f'validation accuracy {accuracy(model, X_val, y_val):.3f}  (chance = 1/15 = {1/15:.3f})')
print('Starter predictions on the test set, per class:', pd.read_csv('predictions.csv').label.value_counts().to_dict())
""")
md(r"""
The handout asks for the spread of every number, so the cell below repeats the starter's exact recipe (same split procedure, model, optimiser and loop) with seeds 0, 1, 2. Seed 0 reproduces the run above.
""")
code(r"""
res = []
for s in [0, 1, 2]:
    torch.manual_seed(s); np.random.seed(s)
    perm = torch.randperm(len(X)); v, t = perm[:len(X) // 5], perm[len(X) // 5:]
    m = SmallCNN().to(DEV); o = torch.optim.SGD(m.parameters(), lr=LR)
    for ep in range(EPOCHS):
        m.train(); order = t[torch.randperm(len(t))]
        for i in range(0, len(order), BATCH_SIZE):
            b = order[i:i + BATCH_SIZE]
            loss = F.cross_entropy(m(X[b].to(DEV)), y[b].to(DEV)); o.zero_grad(); loss.backward(); o.step()
    res.append((accuracy(m, X[t], y[t]), accuracy(m, X[v], y[v])))
res = np.array(res)
STARTER_TRAIN, STARTER_VAL = res[:, 0], res[:, 1]
print(f'Starter recipe, 3 seeds: train {STARTER_TRAIN.mean():.3f} ± {STARTER_TRAIN.std(ddof=1):.3f}, '
      f'validation {STARTER_VAL.mean():.3f} ± {STARTER_VAL.std(ddof=1):.3f}')
""")

# --------------------------------------------------------------------------
md(r"""
## 2. Data: counts, sizes, and what the images look like
From here on, all code is ours. Images are loaded once at their full 128×128 resolution and kept as `uint8` tensors.
""")
code(r"""
import os, math, random, copy
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torchvision.transforms.v2 import functional as TF
from PIL import Image

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
QUICK = os.environ.get('LEAF_QUICK') == '1'          # only for a fast local smoke test; False on Colab
SEEDS = [0, 1, 2]                                    # every experiment is repeated with these 3 seeds
torch.backends.cudnn.benchmark = False               # reproducible convolution algorithms
torch.backends.cudnn.deterministic = True

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def sd(v):
    return float(np.std(v, ddof=1))                  # sample standard deviation over seeds

DATA = '/content/leaves'
train_df = pd.read_csv(f'{DATA}/train_labels.csv')               # columns id, label
sub_df = pd.read_csv(f'{DATA}/sample_submission.csv')            # required id order / format
CLASSES = sorted(train_df.label.unique())
y_all = torch.tensor([CLASSES.index(c) for c in train_df.label])

def load_folder(folder, ids):
    return torch.from_numpy(np.stack([np.asarray(Image.open(f'{DATA}/{folder}/{i}.png').convert('RGB')) for i in ids])
                            ).permute(0, 3, 1, 2).contiguous()     # N x 3 x 128 x 128, uint8

X_all = load_folder('train', train_df.id)
X_te = load_folder('test', sub_df.id)
print('train:', tuple(X_all.shape), ' test:', tuple(X_te.shape), ' dtype:', X_all.dtype)
print('image sizes (train, test):', {tuple(X_all.shape[2:])}, {tuple(X_te.shape[2:])})
print('images per class:', train_df.label.value_counts().sort_index().to_dict())
""")
code(r"""
fig, axes = plt.subplots(15, 6, figsize=(9, 22))
for r, c in enumerate(CLASSES):
    idx = np.where(train_df.label == c)[0][:6]
    for k, i in enumerate(idx):
        axes[r, k].imshow(X_all[i].permute(1, 2, 0)); axes[r, k].axis('off')
    axes[r, 0].set_title(c, fontsize=8, loc='left')
plt.suptitle('Training images: 6 per species', y=1.0); plt.tight_layout(); plt.show()

fig, axes = plt.subplots(3, 10, figsize=(15, 4.8))
for k, ax in enumerate(axes.flat):
    ax.imshow(X_te[k].permute(1, 2, 0)); ax.set_title(sub_df.id[k], fontsize=7); ax.axis('off')
plt.suptitle('Test images (first 30)'); plt.tight_layout(); plt.show()
""")

md(r"""
### Measuring how training and test images differ
For each image we separate the leaf from the background. Every image has a single flat background colour along its border, so a pixel is "leaf" if it differs from that colour by more than 40 in some channel. From the leaf pixels we measure:
* **orientation:** the angle of the leaf's long axis (the main eigenvector of the covariance of the leaf-pixel coordinates); 0° means vertical. An axis has no direction, so this angle cannot tell up from down;
* **stem direction:** along that axis, which end is thinner (the sign of the third moment, i.e. the skew of the pixel positions along the axis). For an upright leaf the thin end (the stem) points down;
* **position:** the centre of the leaf pixels, and its distance from the image centre;
* **size and colour:** the fraction of the image covered by the leaf, and the mean green value of the leaf pixels.
""")
code(r"""
def leaf_measurements(img):
    a = img.permute(1, 2, 0).numpy().astype(np.int16)                 # H x W x 3
    bg = a[0, 0]                                                      # flat background colour of this image
    mask = np.abs(a - bg).max(axis=2) > 40                            # leaf pixels
    ys, xs = np.nonzero(mask)
    c = np.stack([xs - xs.mean(), ys - ys.mean()])                    # centred pixel coordinates
    v = np.linalg.eigh(np.cov(c))[1][:, -1]                           # long-axis direction
    angle = (np.degrees(np.arctan2(v[0], v[1])) + 90) % 180 - 90      # 0 = vertical (an axis: up = down)
    v = v if v[1] >= 0 else -v                                        # make the axis point down the image
    stem_down = ((v @ c) ** 3).mean() > 0                             # the thin end lies in the downward direction
    return dict(angle=angle, stem_down=stem_down, cx=xs.mean(), cy=ys.mean(),
                centre_dist=np.hypot(xs.mean() - 63.5, ys.mean() - 63.5), area=mask.mean(), green=a[mask][:, 1].mean())

meas = pd.DataFrame([leaf_measurements(x) for x in X_all] + [leaf_measurements(x) for x in X_te])
meas['split'] = ['train'] * len(X_all) + ['test'] * len(X_te)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for s, col in [('train', 'C0'), ('test', 'C1')]:
    m = meas[meas.split == s]
    axes[0].hist(m.angle, bins=36, range=(-90, 90), alpha=.6, label=s, color=col)
    axes[1].hist(m.centre_dist, bins=30, range=(0, 30), alpha=.6, label=s, color=col)
    axes[2].hist(m.area, bins=30, alpha=.6, label=s, color=col)
axes[0].set_xlabel('leaf long-axis angle from vertical (deg)'); axes[0].set_title('Orientation'); axes[0].legend()
axes[1].set_xlabel('distance of leaf centre from image centre (px)'); axes[1].set_title('Position'); axes[1].legend()
axes[2].set_xlabel('fraction of image covered by the leaf'); axes[2].set_title('Size'); axes[2].legend()
plt.tight_layout(); plt.show()

near = meas[meas.angle.abs() < 30]
summary = pd.DataFrame({
    'near-vertical (|angle| < 30 deg)': meas.groupby('split').angle.apply(lambda a: (a.abs() < 30).mean()),
    'stem down, among near-vertical': near.groupby('split').stem_down.mean(),
    'centre distance, median (px)': meas.groupby('split').centre_dist.median(),
    'leaf area, mean': meas.groupby('split').area.mean(),
    'leaf green, mean': meas.groupby('split').green.mean()})
display(summary.round(3))
""")

md(r"""
**Findings**
* Classes are balanced (40 each); every image is 128×128 RGB with one flat, near-white background colour.
* **Training leaves are upright:** about 9 in 10 have their long axis within 30° of vertical, and among those the stem points down in most images. The few exceptions are nearly round leaves, whose "long axis" is ill-defined.
* **Test leaves are rotated to every angle:** only about 1 in 3 is within 30° of vertical (what uniformly random angles give), and among the near-vertical ones the stem points down only about half the time, so they are upside down as often as not.
* **No extra shift:** test leaf centres look scattered, but their distance from the image centre is the same as in training. A training leaf's centre sits a few pixels above the image centre (the stem hangs below), and rotating the image about its centre carries that offset round.
* Leaf size, colour and background are the same in both sets. **The test images look like training images rotated about the image centre, with the uncovered corners filled in the background colour.**

**Consequences**
1. A CNN is not rotation-invariant. Its filters respond to patterns in a particular orientation, so a model trained only on upright leaves can be expected to lose accuracy on rotated ones (tested below: E6 vs E7, P1 vs P2). We **train with random rotations over the full 360°**, filling the corners with the image's own background colour, exactly like the test images.
2. A validation set cut from the training images is upright, so its accuracy can overestimate test accuracy. Every experiment therefore also reports accuracy on a **rotated validation set**: the same validation images, each rotated about its centre by a random angle (fixed seed). This is our estimate of test accuracy.
""")

# --------------------------------------------------------------------------
md(r"""
## 3. Why the starter performs poorly: hypotheses, each tested by a row of the experiment log
1. **Very few gradient updates.** 480 training images with batch size 256 gives 2 updates per epoch, so 5 epochs give **10 updates**. The starter predicts the same species for every test image, and its training accuracy is as low as its validation accuracy (section 1): it is *under-fitting*, not over-fitting. Tested by E1 (batch 32) and E2 (40 epochs).
2. **Plain SGD at a fixed learning rate of 0.05, without momentum.** Tested by E3 (Adam).
3. **Images shrunk from 128 to 64 px.** This throws away fine detail such as leaf-margin teeth and veins. Tested by E4.
4. **No input standardisation and no batch normalisation.** The inputs are mostly bright background (mean pixel value about 0.85), which suits the default weight initialisation poorly. Tested by E5 and E6.
5. **Train/test mismatch:** the model only ever sees upright leaves, but the test leaves are rotated (section 2). Tested by E7, and by P1 vs P2 for the pretrained model.

Separately, the starter's *measurement* is unreliable. Its validation split is random rather than stratified (the per-class counts printed below are uneven), it uses one seed, and it contains only upright images. This doesn't make the model worse, but it makes its validation number a poor guide to test accuracy.
""")
code(r"""
print('updates per epoch in the starter:', math.ceil(len(X_tr) / 256), '-> total updates:', math.ceil(len(X_tr) / 256) * 5)
print('validation images per class in the starter split:', np.bincount(y_val.numpy(), minlength=15).tolist())
print(f'mean pixel value of the training images: {X_all.float().mean() / 255:.2f}')
fig, axes = plt.subplots(1, 2, figsize=(5, 2.6))
axes[0].imshow(X_all[3].permute(1, 2, 0)); axes[0].set_title('128 px (original)', fontsize=8)
axes[1].imshow(F.interpolate(X_all[3:4].float(), size=64, mode='bilinear', antialias=True)[0].permute(1, 2, 0).byte()); axes[1].set_title('64 px (starter)', fontsize=8)
for a in axes: a.axis('off')
plt.tight_layout(); plt.show()
""")

# --------------------------------------------------------------------------
md(r"""
## 4. Shared code for all experiments
* **Split:** stratified, 32 training and 8 validation images per species (480 / 120). The split is fixed for every experiment, so the rows of the log are comparable.
* **Rotation** (`rotate_shift`): rotates an image about its centre. The corners uncovered by the rotation are filled with that image's own background colour, like the test images. (Implementation detail: we rotate *image − background* with zero padding and add the background back. torchvision's `fill=` option instead left a thin dark seam along the old image edge under bilinear interpolation.)
* **Rotated validation set:** each validation image rotated by a random angle in [−180°, 180°] (fixed seed).
* **Augmentation** (training only, a fresh random draw for every image in every epoch): rotation by any angle, plus a left-right mirror half of the time. Neither changes the species.
* **Test-time augmentation (TTA):** average the predicted probabilities over the 8 rotations/mirrors that map a square image exactly onto itself (0°, 90°, 180°, 270°, each with and without a mirror). These are exact pixel rearrangements, with no interpolation.
* **`run_experiment`** trains one configuration with each of the 3 seeds and records both validation accuracies.
""")
code(r"""
# ---- fixed stratified split: 8 validation images per species ----
rng = np.random.RandomState(0)
val_idx = np.sort(np.concatenate([rng.choice(np.where(y_all == k)[0], 8, replace=False) for k in range(15)]))
tr_idx = np.setdiff1d(np.arange(len(y_all)), val_idx)
X_all_d, X_te_d, y_all_d = X_all.to(DEV), X_te.to(DEV), y_all.to(DEV)

def rotate_shift(x, angle, tx=0, ty=0, flip=False):
    # x: 3 x H x W float image in [0,1]. Rotate by `angle` degrees about the centre, shift by (tx, ty) pixels,
    # optionally mirror. Every image has one flat background colour (its top-left pixel); we rotate the
    # difference from it, so the corners uncovered by the rotation (zero-filled) become exactly that colour.
    bg = x[:, :1, :1]
    x = TF.affine(x - bg, angle=angle, translate=[tx, ty], scale=1.0, shear=[0.0, 0.0],
                  interpolation=TF.InterpolationMode.BILINEAR) + bg
    x = x.clamp(0, 1)
    return TF.horizontal_flip(x) if flip else x

def augment(xb):
    # a fresh random rotation (any angle) and a random mirror for every image in the batch
    return torch.stack([rotate_shift(x, random.uniform(-180, 180), flip=random.random() < 0.5) for x in xb])

# ---- validation sets ----
val_y = y_all_d[val_idx]
val_x = X_all_d[val_idx].float() / 255                                  # upright
random.seed(123)
val_x_rot = torch.stack([rotate_shift(x, random.uniform(-180, 180)) for x in val_x])   # rotated, test-like

fig, axes = plt.subplots(2, 8, figsize=(14, 3.8))
for k in range(8):
    axes[0, k].imshow(val_x[k * 15].cpu().permute(1, 2, 0)); axes[0, k].axis('off')
    axes[1, k].imshow(val_x_rot[k * 15].cpu().permute(1, 2, 0)); axes[1, k].axis('off')
axes[0, 0].set_title('validation', fontsize=8, loc='left'); axes[1, 0].set_title('rotated validation (test-like)', fontsize=8, loc='left')
plt.tight_layout(); plt.show()
""")
code(r"""
class SmallCNN(nn.Module):
    # The starter's network. bn=True adds batch normalisation after each convolution; width sets the channels.
    def __init__(self, n_classes=15, width=16, bn=False):
        super().__init__()
        layers, c_in = [], 3
        for c_out in (width, 2 * width, 4 * width):
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1)]
            if bn:
                layers += [nn.BatchNorm2d(c_out)]
            layers += [nn.ReLU(), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(4 * width, n_classes)

    def forward(self, x):
        return self.head(F.adaptive_avg_pool2d(self.features(x), 1).flatten(1))

class Wrapped(nn.Module):
    # Resize to the network's input size and standardise, so every model takes 128x128 images in [0,1].
    def __init__(self, net, size, mean, std):
        super().__init__()
        self.net, self.size = net, size
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        if self.size != x.shape[-1]:
            x = F.interpolate(x, size=self.size, mode='bilinear', antialias=True, align_corners=False)
        return self.net((x - self.mean) / self.std)

DATA_MEAN = (X_all[tr_idx].float() / 255).mean(dim=(0, 2, 3)).tolist()     # from training images only
DATA_STD = (X_all[tr_idx].float() / 255).std(dim=(0, 2, 3)).tolist()
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def build_model(cfg):
    if cfg['model'] == 'resnet18':
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Linear(net.fc.in_features, 15)                          # new 15-way classifier (512 -> 15)
        return Wrapped(net, cfg['size'], IMAGENET_MEAN, IMAGENET_STD)
    net = SmallCNN(width=cfg['width'], bn=cfg['bn'])
    mean, std = (DATA_MEAN, DATA_STD) if cfg['normalize'] else ([0, 0, 0], [1, 1, 1])
    return Wrapped(net, cfg['size'], mean, std)
""")
code(r"""
@torch.no_grad()
def predict_proba(model, X, tta=False):
    # X: N x 3 x 128 x 128 float in [0,1]; returns N x 15 class probabilities
    model.eval()
    views = [X] if not tta else [torch.rot90(v, k, dims=(2, 3)) for v in (X, X.flip(3)) for k in range(4)]
    return torch.stack([torch.cat([F.softmax(model(v[i:i + 128]), 1) for i in range(0, len(v), 128)]) for v in views]).mean(0)

def accuracy_of(model, X, y, tta=False):
    return (predict_proba(model, X, tta).argmax(1) == y).float().mean().item()

def train_model(cfg, seed, idx):
    # train one model on the training images idx; returns the model and its per-epoch history
    set_seed(seed)
    model = build_model(cfg).to(DEV)
    if cfg['opt'] == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg.get('wd', 0))
    else:
        opt = torch.optim.SGD(model.parameters(), lr=cfg['lr'])
    epochs = cfg['epochs'] if not QUICK else 2
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs) if cfg.get('cosine') else None
    idx = torch.as_tensor(idx, device=DEV)
    hist = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_rot_acc': []}
    for ep in range(epochs):
        model.train(); total, correct = 0.0, 0
        order = idx[torch.randperm(len(idx), device=DEV)]
        for i in range(0, len(order), cfg['bs']):
            b = order[i:i + cfg['bs']]
            xb, yb = X_all_d[b].float() / 255, y_all_d[b]
            if cfg['aug']:
                xb = augment(xb)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b); correct += (logits.argmax(1) == yb).sum().item()
        if sched:
            sched.step()
        hist['train_loss'].append(total / len(idx)); hist['train_acc'].append(correct / len(idx))
        if cfg.get('track', True):                    # accuracy on both validation sets after every epoch
            hist['val_acc'].append(accuracy_of(model, val_x, val_y))
            hist['val_rot_acc'].append(accuracy_of(model, val_x_rot, val_y))
    return model, hist

LOG, RUNS = [], {}
def log_row(step, change, reason, cfg, va, vr, models, hists, seconds):
    RUNS[step] = dict(cfg=cfg, val=va, val_rot=vr, models=models, hists=hists)
    LOG.append({'step': step, 'change': change, 'reason': reason,
                'val acc': f'{np.mean(va):.3f} ± {sd(va):.3f}', 'rotated val acc': f'{np.mean(vr):.3f} ± {sd(vr):.3f}',
                'seconds': round(seconds)})
    print(f"{step}: val {np.mean(va):.3f} ± {sd(va):.3f} | rotated val {np.mean(vr):.3f} ± {sd(vr):.3f} "
          f"| rotated val per seed {[round(v, 3) for v in vr]} | {seconds:.0f}s")

def run_experiment(step, change, reason, cfg):
    # train cfg once per seed; score on the upright and the rotated validation sets
    t0 = time.time(); va, vr, models, hists = [], [], [], []
    for s in SEEDS:
        model, h = train_model(cfg, s, tr_idx)
        va.append(accuracy_of(model, val_x, val_y)); vr.append(accuracy_of(model, val_x_rot, val_y))
        models.append(model); hists.append(h)
    log_row(step, change, reason, cfg, va, vr, models, hists, time.time() - t0)

def tta_row(step, from_step, change, reason):
    # inference-only change: the SAME trained models, predictions averaged over 8 views
    t0 = time.time(); r = RUNS[from_step]
    va = [accuracy_of(m, val_x, val_y, tta=True) for m in r['models']]
    vr = [accuracy_of(m, val_x_rot, val_y, tta=True) for m in r['models']]
    log_row(step, change, reason, dict(r['cfg'], tta=True), va, vr, r['models'], r['hists'], time.time() - t0)
""")

# --------------------------------------------------------------------------
md(r"""
## 5. Experiment log: fixing the starter's small CNN, one change per row
Each row keeps everything from the row above and changes **one** setting (`cfg = dict(cfg, key=value)`). Every row is trained with the 3 seeds. E0 re-implements the starter's settings in our pipeline (stratified split, resizing inside the model), so its numbers can differ slightly from section 1. E11 re-scores the E10 models with TTA; nothing is retrained.
""")
code(r"""
cfg = dict(model='small', width=16, bn=False, size=64, normalize=False, opt='sgd', lr=0.05, bs=256, epochs=5, aug=False)
run_experiment('E0', 'starter settings', 'baseline: 64 px, batch 256, 5 epochs, plain SGD lr 0.05', cfg)

cfg = dict(cfg, bs=32)
run_experiment('E1', 'batch size 256 -> 32', '15 updates per epoch instead of 2 (75 in total)', cfg)

cfg = dict(cfg, epochs=40)
run_experiment('E2', 'epochs 5 -> 40', '600 updates in total instead of 75', cfg)

cfg = dict(cfg, opt='adam', lr=1e-3)
run_experiment('E3', 'plain SGD lr 0.05 -> Adam lr 1e-3', 'Adam adapts each parameter\'s step size', cfg)

cfg = dict(cfg, size=128)
run_experiment('E4', 'input 64 -> 128 px', 'keep fine detail: margin teeth, veins, stem', cfg)

cfg = dict(cfg, normalize=True)
run_experiment('E5', 'standardise inputs (train mean/std)', 'inputs are mostly bright background; centre and scale them', cfg)

cfg = dict(cfg, bn=True)
run_experiment('E6', 'add batch normalisation', 'keep activations well scaled during training', cfg)

cfg = dict(cfg, aug=True)
run_experiment('E7', 'augmentation: rotation (any angle) + mirror', 'test leaves are rotated; train on rotated leaves', cfg)

cfg = dict(cfg, epochs=80)
run_experiment('E8', 'epochs 40 -> 80', 'augmented images never repeat exactly: more passes needed', cfg)

cfg = dict(cfg, cosine=True)
run_experiment('E9', 'cosine learning-rate decay', 'large steps early, small steps at the end to settle', cfg)

cfg = dict(cfg, width=32)
run_experiment('E10', 'width 16 -> 32 channels', 'more filters for leaves seen at every angle', cfg)

tta_row('E11', 'E10', 'test-time augmentation (8 views)', 'average over rotations and mirrors')
SMALL_FINAL = 'E11'
""")

# --------------------------------------------------------------------------
md(r"""
## 6. ImageNet-pretrained ResNet-18 (fine-tuned)
The handout allows ImageNet-pretrained backbones. A pretrained ResNet-18 already has general-purpose filters (edges, textures, shapes), so 480 images only need to adapt them rather than learn them from scratch.

Settings: the 1000-class ImageNet layer is replaced by a new 15-class layer and the whole network is fine-tuned. Images are upsampled from 128 to 224 px, the size the backbone was trained at, and standardised with ImageNet's mean/std, the statistics it was trained with. Adam is used with a small learning rate (3e-4), so the pretrained weights are adjusted gently rather than overwritten, plus mild weight decay (1e-4) and cosine decay over 25 epochs (pretrained networks converge quickly, and this keeps the notebook within the time limit). Batch size is 32, as for the small CNN.

P1 tests whether pretraining alone copes with rotated leaves; P2 adds the rotation augmentation; P3 re-scores the P2 models with TTA.
""")
code(r"""
rcfg = dict(model='resnet18', size=224, opt='adam', lr=3e-4, wd=1e-4, bs=32, epochs=25, cosine=True, aug=False)
run_experiment('P1', 'ImageNet-pretrained ResNet-18, no augmentation', 'transfer learning, trained on upright leaves only', rcfg)

rcfg = dict(rcfg, aug=True)
run_experiment('P2', '+ augmentation: rotation (any angle) + mirror', 'same fix as E7', rcfg)

tta_row('P3', 'P2', '+ test-time augmentation (8 views)', 'same as E11')
FINAL_CFG = rcfg
""")
code(r"""
log_df = pd.DataFrame(LOG)
pd.set_option('display.max_colwidth', 70)
display(log_df)

steps = [r['step'] for r in LOG]
fig, ax = plt.subplots(figsize=(13, 4))
for j, (key, lab) in enumerate([('val', 'validation (upright)'), ('val_rot', 'rotated validation (test-like)')]):
    m = [np.mean(RUNS[s][key]) for s in steps]; e = [sd(RUNS[s][key]) for s in steps]
    ax.bar(np.arange(len(steps)) + (j - .5) * .4, m, .4, yerr=e, capsize=3, label=lab)
ax.set_xticks(range(len(steps))); ax.set_xticklabels(steps); ax.set_ylabel('accuracy (mean ± sd, 3 seeds)')
ax.axhline(1 / 15, color='k', ls=':', lw=1, label='chance'); ax.legend(); ax.grid(axis='y', alpha=.3)
ax.set_title('Experiment log'); plt.tight_layout(); plt.show()

for s, name in [(SMALL_FINAL, "starter's small CNN after all changes (E11)"), ('P3', 'pretrained ResNet-18 (P3)')]:
    r = RUNS[s]
    print(f"{name}: validation {np.mean(r['val']):.3f} ± {sd(r['val']):.3f} | rotated validation {np.mean(r['val_rot']):.3f} ± {sd(r['val_rot']):.3f}")
""")

# --------------------------------------------------------------------------
md(r"""
### Evidence for the key changes
**(a) Learning curves** for all 3 seeds (one thin line each): E0 (starter settings), E2 (600 plain-SGD updates), E3 (Adam), E6 (best model without augmentation) and E7 (with augmentation). Solid lines: accuracy on the two validation sets; dashed: training accuracy on the (augmented) training batches.
""")
code(r"""
show = ['E0', 'E2', 'E3', 'E6', 'E7']
fig, axes = plt.subplots(1, len(show), figsize=(4.2 * len(show), 3.6), sharey=True)
for ax, s in zip(axes, show):
    for k, h in enumerate(RUNS[s]['hists']):
        ep = np.arange(1, len(h['train_loss']) + 1)
        ax.plot(ep, h['train_acc'], 'k--', lw=.8, alpha=.6, label='train (batches)' if k == 0 else None)
        ax.plot(ep, h['val_acc'], 'C0', lw=1, alpha=.8, label='val (upright)' if k == 0 else None)
        ax.plot(ep, h['val_rot_acc'], 'C1', lw=1, alpha=.8, label='val (rotated)' if k == 0 else None)
    ax.set_ylim(0, 1.02); ax.set_xlabel('epoch'); ax.set_title(s); ax.grid(alpha=.3)
axes[0].set_ylabel('accuracy'); axes[0].legend(fontsize=7)
plt.tight_layout(); plt.show()
""")
md(r"""
**(b) What the CNN is and is not invariant to.** The E6 (no augmentation) and E7 (augmentation) models classify the validation leaves after a fixed rotation, mean ± sd over the 3 seeds. A convolution applies the same filter at every position, so a shifted leaf gives a shifted feature map, and the global average pooling at the end discards position. The output is therefore nearly unchanged by shifts: not exactly, because of the stride-2 pooling and the image border. Nothing in the architecture does the same for *rotation*, so the model must learn it from rotated examples.
""")
code(r"""
angles = list(range(-180, 181, 30))
rot_sets = {a: torch.stack([rotate_shift(x, a) for x in val_x]) for a in angles}
fig, ax = plt.subplots(figsize=(8, 3.8))
for s in ['E6', 'E7']:
    acc = np.array([[accuracy_of(m, rot_sets[a], val_y) for a in angles] for m in RUNS[s]['models']])   # seeds x angles
    mu, e = acc.mean(0), acc.std(0, ddof=1)
    ax.plot(angles, mu, marker='o', label=f"{s} ({'with' if RUNS[s]['cfg']['aug'] else 'no'} augmentation)")
    ax.fill_between(angles, mu - e, mu + e, alpha=.2)
ax.set_xlabel('rotation applied to validation leaves (deg)'); ax.set_ylabel('accuracy (mean ± sd, 3 seeds)')
ax.set_ylim(0, 1.02); ax.set_xticks(angles); ax.grid(alpha=.3); ax.legend(); ax.set_title('Accuracy vs rotation')
plt.tight_layout(); plt.show()
shift_acc = np.array([[accuracy_of(m, torch.stack([rotate_shift(x, 0, tx=d) for x in val_x]), val_y) for d in (0, 5, 10)]
                      for m in RUNS['E6']['models']])
print('E6 (no augmentation) accuracy after shifting right by 0 / 5 / 10 px:',
      [f'{a:.3f} ± {b:.3f}' for a, b in zip(shift_acc.mean(0), shift_acc.std(0, ddof=1))])
""")

# --------------------------------------------------------------------------
md(r"""
## 7. Final model
**Final configuration (row P3):** ImageNet-pretrained ResNet-18 (torchvision `IMAGENET1K_V1`), with a new 15-class linear layer and the whole network fine-tuned.
* Preprocessing: 128 px images upsampled to 224 px, ImageNet mean/std standardisation.
* Augmentation: rotation about the centre by any angle (corners filled with the image's background colour), random left-right mirror.
* Optimiser and schedule: Adam, learning rate 3e-4, weight decay 1e-4, batch size 32, 25 epochs with cosine decay.
* Test-time augmentation: average over 8 rotations/mirrors.

For comparison (the handout asks for this when a pretrained backbone is used), the **starter's small CNN after all changes (row E11)**: 3 convolutional layers (32/64/128 channels) with batch normalisation, 128 px input standardised with the training mean/std, Adam lr 1e-3, batch 32, 80 epochs with cosine decay, the same augmentation and TTA.

Below: the confusion matrix on the rotated validation set (3 seeds pooled), then misclassified examples.
""")
code(r"""
from sklearn.metrics import confusion_matrix
P3 = RUNS['P3']
print(f"Final config (P3), 3 seeds: validation {np.mean(P3['val']):.3f} ± {sd(P3['val']):.3f}, "
      f"rotated validation {np.mean(P3['val_rot']):.3f} ± {sd(P3['val_rot']):.3f}")
yv = val_y.cpu().numpy()
preds = [predict_proba(m, val_x_rot, tta=True).argmax(1).cpu().numpy() for m in P3['models']]
cm = confusion_matrix(np.tile(yv, len(preds)), np.concatenate(preds), labels=range(15))
fig, ax = plt.subplots(figsize=(7.5, 6.5))
ax.imshow(cm, cmap='Blues')
for i in range(15):
    for j in range(15):
        if cm[i, j]: ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=7, color='w' if cm[i, j] > cm.max() / 2 else 'k')
ax.set_xticks(range(15)); ax.set_yticks(range(15))
ax.set_xticklabels([c[-2:] for c in CLASSES]); ax.set_yticklabels(CLASSES, fontsize=8)
ax.set_xlabel('predicted species'); ax.set_ylabel('true species')
ax.set_title('Rotated validation set, final config, 3 seeds pooled (8 x 3 = 24 per row)'); plt.tight_layout(); plt.show()
CONFUSIONS = sorted([(int(cm[i, j]), CLASSES[i], CLASSES[j]) for i in range(15) for j in range(15) if i != j and cm[i, j]], reverse=True)
print('most frequent confusions (count over 3 seeds, true -> predicted):', CONFUSIONS[:8])
N_WRONG = [int((p != yv).sum()) for p in preds]
print(f'misclassified rotated validation images per seed: {N_WRONG} (mean {np.mean(N_WRONG):.1f} ± {sd(N_WRONG):.1f} of {len(yv)})')
""")
md(r"""
**Misclassified examples.** Each column is one error (from any seed). Top: the rotated image the model saw, with true and predicted species. Middle: the same leaf upright, with the same model's prediction. If the upright leaf is classified correctly, the error is down to orientation. Bottom: a training example of the predicted species, for comparison.
""")
code(r"""
shown, seen = [], set()
for s, p in enumerate(preds):
    for k in np.where(p != yv)[0]:
        if k not in seen:
            seen.add(k); shown.append((s, k))
shown = shown[:8]
if not shown:   # no errors at all: show the least confident predictions instead
    conf = torch.stack([predict_proba(m, val_x_rot, tta=True) for m in P3['models']]).mean(0).max(1).values.cpu().numpy()
    shown = [(0, k) for k in np.argsort(conf)[:8]]; print('no errors: showing the 8 least confident predictions')
fig, axes = plt.subplots(3, len(shown), figsize=(2.2 * len(shown), 7.2), squeeze=False)
for j, (s, k) in enumerate(shown):
    p_up = predict_proba(P3['models'][s], val_x[k:k + 1], tta=True).argmax(1).item()
    ex = np.where(y_all.numpy() == preds[s][k])[0][0]
    axes[0, j].imshow(val_x_rot[k].cpu().permute(1, 2, 0))
    axes[0, j].set_title(f'seed {s}: true {CLASSES[yv[k]][-2:]}\npred {CLASSES[preds[s][k]][-2:]}', fontsize=8, color='r')
    axes[1, j].imshow(val_x[k].cpu().permute(1, 2, 0)); axes[1, j].set_title(f'upright: pred {CLASSES[p_up][-2:]}', fontsize=8)
    axes[2, j].imshow(X_all[ex].permute(1, 2, 0)); axes[2, j].set_title(f'a training {CLASSES[preds[s][k]][-2:]}', fontsize=8)
for a in axes.flat: a.axis('off')
plt.suptitle('Errors: as seen (top), same leaf upright (middle), a training leaf of the predicted species (bottom)')
plt.tight_layout(); plt.show()
""")
md(r"""
**Interpretation of the confusion matrix and the errors:** see section 8, written from the outputs above.

### Predictions for the test set
The final configuration is retrained on **all 600 labelled images** with each of the 3 seeds: same settings, no validation hold-out, so there is no validation number for these exact models (P3 is their estimate). The test prediction averages the 3 models' TTA probabilities. How often the seeds agree with each other shows how stable the predictions are.
""")
code(r"""
final_models = []
for s in SEEDS:
    m, _ = train_model(dict(FINAL_CFG, track=False), s, np.arange(len(y_all)))
    final_models.append(m)
X_te_f = X_te_d.float() / 255
probs = [predict_proba(m, X_te_f, tta=True) for m in final_models]
test_pred = torch.stack(probs).mean(0).argmax(1).cpu().numpy()
pair_agree = [(probs[a].argmax(1) == probs[b].argmax(1)).float().mean().item() for a, b in [(0, 1), (0, 2), (1, 2)]]
print(f'pairwise agreement between seeds on the test images: {np.mean(pair_agree):.3f} ± {sd(pair_agree):.3f} '
      f'(pairs 0-1, 0-2, 1-2: {[round(a, 3) for a in pair_agree]})')

pred_df = pd.DataFrame({'id': sub_df.id, 'label': [CLASSES[k] for k in test_pred]})
pred_df.to_csv('predictions.csv', index=False)
check = pd.read_csv('predictions.csv')
assert list(check.columns) == ['id', 'label'] and len(check) == 525 and check.id.tolist() == sub_df.id.tolist()
print('predictions.csv:', check.shape); display(check.head())
print('predicted test images per class:', check.label.value_counts().sort_index().to_dict())
""")

# --------------------------------------------------------------------------
md(r"""
## 8. Summary
The cell below prints the key numbers from this run (mean ± sd over 3 seeds). The written interpretation follows it.
""")
code(r"""
def ms(v):
    return f'{np.mean(v):.3f} ± {sd(v):.3f}'
R = RUNS
print('Starter, unchanged recipe (3 seeds):     train', ms(STARTER_TRAIN), '| validation', ms(STARTER_VAL))
for s in steps:
    print(f"{s:4s} {LOG[steps.index(s)]['change'][:55]:55s} val {ms(R[s]['val'])} | rotated val {ms(R[s]['val_rot'])}")
print(f"Augmentation effect, small CNN (E6 -> E7), rotated val: {ms(R['E6']['val_rot'])} -> {ms(R['E7']['val_rot'])}")
print(f"Augmentation effect, ResNet-18 (P1 -> P2), rotated val: {ms(R['P1']['val_rot'])} -> {ms(R['P2']['val_rot'])}")
print(f"Small CNN final (E11) vs ResNet-18 final (P3), rotated val: {ms(R['E11']['val_rot'])} vs {ms(R['P3']['val_rot'])}")
print('Most frequent confusions of the final config:', CONFUSIONS[:4])
print(f'Total run time: {(time.time() - T_START) / 60:.1f} min')
""")
md(r"""
*(Written interpretation: added after the Colab run.)*

## References and code sources
* Starter notebook `LeafID_starter.ipynb` (course material): section 1 is its code, copied unchanged. Its `accuracy` function and its recipe are reused in the two cells after it, and our `SmallCNN` is the starter's network with options to add batch normalisation and change the width.
* Data: `leaves.zip` provided with the assignment on Classroom (train/, train_labels.csv, test/, sample_submission.csv). No other data was used, apart from the ImageNet-pretrained ResNet-18 weights from torchvision.
* PyTorch and torchvision documentation (models, `transforms.v2.functional.affine`, optimisers, LR schedulers): https://pytorch.org/docs, https://pytorch.org/vision. NumPy, pandas, matplotlib, Pillow; scikit-learn `confusion_matrix` (Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, 2011).
* K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016 (ResNet); weights `ResNet18_Weights.IMAGENET1K_V1`.
* S. Ioffe, C. Szegedy. *Batch Normalization.* ICML 2015.
* D. P. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015.
* I. Loshchilov, F. Hutter. *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR 2017 (cosine schedule).
* AI coding assistant (Claude, Anthropic) used as a coding aid. I checked and ran all code and can explain every part.
""")

# ==========================================================================
nb = nbf.v4.new_notebook()
nb.cells = CELLS
nb.metadata = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
               "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "LeafID_Rizwan_Mir.ipynb"
nbf.write(nb, out)
print("wrote", out, f"({len(CELLS)} cells)")
