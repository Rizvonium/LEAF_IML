"""Build LeafID_Rizwan_Mir.ipynb from the cell sources below.

    python tools/build_notebook.py        # writes LeafID_Rizwan_Mir.ipynb

The notebook is the deliverable. This script only exists so the cells can be
edited as plain Python and the notebook regenerated.
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

**Goal.** Predict which of 15 tree species a scanned leaf comes from. We have 600 labelled training images (40 per species) and 525 unlabelled test images, all 128×128 RGB.

**Plan of this notebook**
1. Run the starter notebook **unchanged** and record its results.
2. Load the data; report class counts and image sizes; examine the training and test images. Key finding: **training leaves are upright and centred, test leaves are rotated to any angle and shifted.**
3. Diagnose why the starter performs poorly.
4. Shared code: validation split, a rotated (test-like) validation set, augmentation, models, training loop.
5. Experiment log: fix the starter's small CNN **one change at a time**, with 3 random seeds per change.
6. Fine-tune an ImageNet-pretrained ResNet-18 (allowed by the handout) with the same fixes.
7. Final model: 3 seeds, mean ± spread; confusion matrix; misclassified examples; `predictions.csv` (header `id,label`, 525 rows).
8. Summary and references.

**Runtime:** Colab → Runtime → Change runtime type → **T4 GPU**, upload `leaves.zip` to `/content`, then Runtime → **Run all** (well under 30 minutes).
""")

# --------------------------------------------------------------------------
md(r"""
## 1. The starter notebook, unchanged
The next six cells are the starter notebook's code cells, copied without any change, so its results are recorded before anything is modified. (It writes a `predictions.csv`; that file is overwritten by the final model at the end.)
""")
STARTER = json.load(open(ROOT / "starter" / "LeafID_starter.ipynb"))
for c in STARTER["cells"]:
    if c["cell_type"] == "code":
        code("".join(c["source"]))

code(r"""
# Record the starter's numbers before changing anything
starter_train_acc = accuracy(model, X_tr, y_tr)
starter_val_acc = accuracy(model, X_val, y_val)
print(f'Starter (unchanged): train accuracy {starter_train_acc:.3f}, validation accuracy {starter_val_acc:.3f}')
print(f'Chance level for 15 balanced classes: {1/15:.3f}')
print('Starter predictions per class on the test set:', pd.read_csv('predictions.csv').label.value_counts().to_dict())
""")

# --------------------------------------------------------------------------
md(r"""
## 2. Data: counts, sizes, and what the images look like
From here on, all code is ours. Images are loaded once at their full 128×128 resolution and kept as `uint8` tensors.
""")
code(r"""
import os, time, math, random, copy
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
print('images per class:'); print(train_df.label.value_counts().sort_index().to_string())
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
### Measuring the difference between training and test images
For each image we separate the leaf from the background by thresholding against the background colour (median of the border pixels). From the leaf pixels we measure:
* **orientation:** the angle of the leaf's long axis (the main eigenvector of the pixel-coordinate covariance), with 0° meaning vertical;
* **position:** the centre of the leaf pixels;
* **size and colour:** the fraction of the image covered by the leaf, and the mean leaf colour.
""")
code(r"""
def leaf_measurements(img):
    a = img.permute(1, 2, 0).numpy().astype(np.int16)                 # H x W x 3
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(border, axis=0)
    mask = np.abs(a - bg).max(axis=2) > 40                            # leaf pixels
    ys, xs = np.nonzero(mask)
    cov = np.cov(np.stack([xs - xs.mean(), ys - ys.mean()]))
    vx, vy = np.linalg.eigh(cov)[1][:, -1]                            # long-axis direction
    angle = (np.degrees(np.arctan2(vx, vy)) + 90) % 180 - 90          # 0 = vertical
    return dict(angle=angle, cx=xs.mean(), cy=ys.mean(), area=mask.mean(),
                green=a[mask][:, 1].mean(), bg=bg.mean())

meas = pd.DataFrame([leaf_measurements(x) for x in X_all] + [leaf_measurements(x) for x in X_te])
meas['split'] = ['train'] * len(X_all) + ['test'] * len(X_te)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for s, col in [('train', 'C0'), ('test', 'C1')]:
    m = meas[meas.split == s]
    axes[0].hist(m.angle, bins=36, range=(-90, 90), alpha=.6, label=s, color=col)
    axes[1].scatter(m.cx, m.cy, s=6, alpha=.5, label=s, color=col)
    axes[2].hist(m.area, bins=30, alpha=.6, label=s, color=col)
axes[0].set_xlabel('leaf long-axis angle from vertical (deg)'); axes[0].set_title('Orientation'); axes[0].legend()
axes[1].set_xlabel('leaf centre x (px)'); axes[1].set_ylabel('leaf centre y (px)'); axes[1].set_title('Position'); axes[1].legend()
axes[1].set_xlim(30, 98); axes[1].set_ylim(98, 30)
axes[2].set_xlabel('fraction of image covered by the leaf'); axes[2].set_title('Size'); axes[2].legend()
plt.tight_layout(); plt.show()

summary = meas.groupby('split').agg(near_vertical=('angle', lambda a: (a.abs() < 30).mean()),
                                    centre_x_sd=('cx', 'std'), centre_y_sd=('cy', 'std'),
                                    leaf_area=('area', 'mean'), leaf_green=('green', 'mean'), background=('bg', 'mean'))
display(summary.round(3))
""")

md(r"""
**Findings**
* The classes are balanced (40 each) and every image is 128×128 RGB on a near-white background.
* **Training leaves are upright** (stem down, long axis within 30° of vertical for about 9 in 10 images) **and horizontally centred**. **Test leaves are rotated to any angle**, upside down included (only about 1 in 3 is near-vertical, which is what uniformly random angles give), **and shifted off centre**.
* Leaf size, leaf colour and background brightness are the same in both sets, so the shift is purely geometric.

**Consequences for the modelling**
1. A CNN is not rotation-invariant. Its filters respond to patterns in a particular orientation, so a model that only ever sees upright leaves has no reason to recognise a sideways one. We must **train with random rotations (full 360°) and shifts**.
2. A validation set cut from the training images is upright, so its accuracy **overestimates** test accuracy. Alongside the normal validation accuracy, every experiment therefore reports accuracy on a **rotated validation set**: the same validation images, each rotated by a random angle and shifted by a few pixels with a fixed seed, to mimic the test images. This is our best estimate of test accuracy.
""")

# --------------------------------------------------------------------------
md(r"""
## 3. Why the starter performs poorly
Each point below is backed by a number or a plot from this notebook.
1. **Far too few gradient updates.** 480 training images with batch size 256 is 2 updates per epoch, so 5 epochs give **10 SGD updates** in total, with plain SGD (no momentum). The network has barely moved from its random initialisation: training accuracy is as low as validation accuracy. That is under-fitting, not over-fitting. *Training long enough is the first change that makes it learn* (E1 in the log).
2. **Plain SGD at a fixed learning rate** makes slow progress even with enough steps. Adam adapts the step size per parameter (E2).
3. **Images are shrunk from 128 to 64 pixels**, which throws away fine detail such as leaf-margin teeth and veins (E3).
4. **Inputs are not standardised and there is no batch normalisation**, so the activation scale drifts from layer to layer and optimisation is harder (E4, E5).
5. **Train/test mismatch:** the model only ever sees upright, centred leaves, but the test leaves are rotated and shifted (section 2). This is the biggest problem once the model learns (E6).
6. **Validation is not representative and not repeated:** a random (unstratified) 20% split of upright images, with one seed. We use a stratified split (8 per species), add the rotated validation set, and repeat each experiment with 3 seeds.
""")
code(r"""
print('updates per epoch in the starter:', math.ceil(len(X_tr) / 256), '-> total updates:', math.ceil(len(X_tr) / 256) * 5)
print('validation images per class in the starter split:', np.bincount(y_val.numpy(), minlength=15).tolist())
fig, axes = plt.subplots(1, 2, figsize=(5, 2.6))
axes[0].imshow(X_all[3].permute(1, 2, 0)); axes[0].set_title('128 px (original)', fontsize=8)
axes[1].imshow(F.interpolate(X_all[3:4].float(), size=64, mode='bilinear', antialias=True)[0].permute(1, 2, 0).byte()); axes[1].set_title('64 px (starter)', fontsize=8)
for a in axes: a.axis('off')
plt.tight_layout(); plt.show()
""")

# --------------------------------------------------------------------------
md(r"""
## 4. Shared code for all experiments
* **Split:** stratified, 32 training and 8 validation images per species (480 / 120), fixed for every experiment so that the rows of the log are comparable.
* **Rotated validation set:** each validation image is rotated by a random angle in [−180°, 180°] and shifted by up to 10 px (fixed seed). The empty corners are filled with the background colour.
* **Augmentation** (training only, a fresh random draw for every image in every epoch): rotation by any angle, a shift of up to 10 px, and a left-right mirror half of the time. None of these change the species.
* **Test-time augmentation (TTA):** average the predicted probabilities over the 8 rotations/mirrors that map a square image exactly onto itself (0°, 90°, 180°, 270°, each with and without a mirror).
* **`run_experiment`** trains one configuration with each of the 3 seeds and records the mean and standard deviation of both validation accuracies.
""")
code(r"""
# ---- fixed stratified split: 8 validation images per species ----
rng = np.random.RandomState(0)
val_idx = np.sort(np.concatenate([rng.choice(np.where(y_all == k)[0], 8, replace=False) for k in range(15)]))
tr_idx = np.setdiff1d(np.arange(len(y_all)), val_idx)

BG = (X_all[:, :, 0, :].float().median(dim=0).values.median(dim=1).values / 255).tolist()   # background RGB in [0,1]
X_all_d, X_te_d, y_all_d = X_all.to(DEV), X_te.to(DEV), y_all.to(DEV)

def rotate_shift(x, angle, tx, ty, flip=False):
    # x: 3 x H x W float image in [0,1]; rotate (degrees), shift (pixels), optionally mirror
    x = TF.affine(x, angle=angle, translate=[tx, ty], scale=1.0, shear=[0.0, 0.0],
                  interpolation=TF.InterpolationMode.BILINEAR, fill=BG)
    return TF.horizontal_flip(x) if flip else x

def augment(xb):
    # a fresh random rotation / shift / mirror for every image in the batch
    return torch.stack([rotate_shift(x, random.uniform(-180, 180), random.randint(-10, 10), random.randint(-10, 10),
                                     random.random() < 0.5) for x in xb])

# ---- the rotated (test-like) validation set, made once with a fixed seed ----
random.seed(123)
val_x_rot = torch.stack([rotate_shift(x, random.uniform(-180, 180), random.randint(-10, 10), random.randint(-10, 10))
                         for x in X_all_d[val_idx].float() / 255])
val_y = y_all_d[val_idx]
val_x = X_all_d[val_idx].float() / 255                 # upright validation images

fig, axes = plt.subplots(2, 8, figsize=(14, 3.8))
for k in range(8):
    axes[0, k].imshow(X_all[val_idx[k * 15]].permute(1, 2, 0)); axes[0, k].axis('off')
    axes[1, k].imshow(val_x_rot[k * 15].cpu().permute(1, 2, 0)); axes[1, k].axis('off')
axes[0, 0].set_title('validation', fontsize=8, loc='left'); axes[1, 0].set_title('rotated validation (test-like)', fontsize=8, loc='left')
plt.tight_layout(); plt.show()
""")
code(r"""
class SmallCNN(nn.Module):
    # The starter's network. bn=True adds batch normalisation after each convolution.
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
    # resize to the network's input size and standardise, so every model takes 128x128 images in [0,1]
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
        net.fc = nn.Linear(net.fc.in_features, 15)                          # new 15-way classifier
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
    hist = {'train_loss': [], 'val_acc': [], 'val_rot_acc': []}
    for ep in range(epochs):
        model.train(); total = 0.0
        order = idx[torch.randperm(len(idx), device=DEV)]
        for i in range(0, len(order), cfg['bs']):
            b = order[i:i + cfg['bs']]
            xb, yb = X_all_d[b].float() / 255, y_all_d[b]
            if cfg['aug']:
                xb = augment(xb)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b)
        if sched:
            sched.step()
        hist['train_loss'].append(total / len(idx))
        if cfg.get('track', True):
            hist['val_acc'].append(accuracy_of(model, val_x, val_y))
            hist['val_rot_acc'].append(accuracy_of(model, val_x_rot, val_y))
    return model, hist

LOG, RUNS = [], {}
def log_row(step, change, reason, cfg, va, vr, models, hists, seconds):
    RUNS[step] = dict(cfg=cfg, val=va, val_rot=vr, models=models, hists=hists)
    LOG.append({'step': step, 'change': change, 'reason': reason,
                'val acc': f'{np.mean(va):.3f} ± {np.std(va):.3f}', 'rotated val acc': f'{np.mean(vr):.3f} ± {np.std(vr):.3f}',
                'seconds': round(seconds)})
    print(f"{step}: val {np.mean(va):.3f} ± {np.std(va):.3f} | rotated val {np.mean(vr):.3f} ± {np.std(vr):.3f} "
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
## 5. Experiment log: fixing the starter's small CNN, one change at a time
Each row keeps everything from the row above and changes **one** thing. Every row is trained with the 3 seeds, and the table shows mean ± standard deviation over the seeds.
""")
code(r"""
cfg = dict(model='small', width=16, bn=False, size=64, normalize=False, opt='sgd', lr=0.05, bs=256, epochs=5, aug=False)
run_experiment('E0', 'starter settings', 'baseline: 64 px, batch 256, 5 epochs, SGD lr 0.05', cfg)

cfg = dict(cfg, bs=32, epochs=40)
run_experiment('E1', 'batch 256 -> 32, epochs 5 -> 40', '10 updates are far too few: now 600 updates', cfg)

cfg = dict(cfg, opt='adam', lr=1e-3)
run_experiment('E2', 'SGD lr 0.05 -> Adam lr 1e-3', 'per-parameter adaptive steps converge faster', cfg)

cfg = dict(cfg, size=128)
run_experiment('E3', '64 -> 128 px input', 'keep fine detail: margin teeth, veins, stem', cfg)

cfg = dict(cfg, normalize=True)
run_experiment('E4', 'standardise inputs', 'zero-mean, unit-variance inputs suit the initialisation', cfg)

cfg = dict(cfg, bn=True)
run_experiment('E5', 'add batch normalisation', 'keeps activations well scaled; faster, more stable training', cfg)

cfg = dict(cfg, aug=True)
run_experiment('E6', 'rotation / shift / mirror augmentation', 'test leaves are rotated and shifted; train for that', cfg)

cfg = dict(cfg, epochs=80, cosine=True)
run_experiment('E7', '40 -> 80 epochs with cosine LR decay', 'augmented data is harder to fit: more time, then small steps', cfg)

cfg = dict(cfg, width=32)
run_experiment('E8', 'width 16 -> 32 channels', 'more filters to cover leaves seen at every angle', cfg)

tta_row('E9', 'E8', 'test-time augmentation (8 views)', 'average over rotations: steadier on rotated leaves')
""")

# --------------------------------------------------------------------------
md(r"""
## 6. ImageNet-pretrained ResNet-18 (fine-tuned)
The handout allows ImageNet-pretrained backbones. A pretrained ResNet-18 already has general-purpose filters (edges, textures, shapes), so 480 images only need to adapt them rather than learn them from scratch. The images are upsampled to 224 px, the size the backbone was trained on, and standardised with ImageNet statistics. The whole network is fine-tuned with Adam at a small learning rate.

P1 tests whether pretraining alone fixes the rotation problem; P2 adds the augmentation; P3 adds TTA (same models as P2).
""")
code(r"""
rcfg = dict(model='resnet18', size=224, opt='adam', lr=3e-4, wd=1e-4, bs=32, epochs=25, cosine=True, aug=False)
run_experiment('P1', 'ImageNet-pretrained ResNet-18, no augmentation', 'transfer learning, but trained on upright leaves only', rcfg)

rcfg = dict(rcfg, aug=True)
run_experiment('P2', '+ rotation / shift / mirror augmentation', 'same fix as E6', rcfg)

tta_row('P3', 'P2', '+ test-time augmentation (8 views)', 'same as E9')
FINAL_CFG = rcfg
""")
code(r"""
log_df = pd.DataFrame(LOG)
pd.set_option('display.max_colwidth', 80)
display(log_df)

steps = [r['step'] for r in LOG]
fig, ax = plt.subplots(figsize=(12, 4))
for j, (key, lab) in enumerate([('val', 'validation (upright)'), ('val_rot', 'rotated validation (test-like)')]):
    m = [np.mean(RUNS[s][key]) for s in steps]; sd = [np.std(RUNS[s][key]) for s in steps]
    ax.bar(np.arange(len(steps)) + (j - .5) * .4, m, .4, yerr=sd, capsize=3, label=lab)
ax.set_xticks(range(len(steps))); ax.set_xticklabels(steps); ax.set_ylabel('accuracy (mean ± sd over 3 seeds)')
ax.axhline(1 / 15, color='k', ls=':', lw=1, label='chance'); ax.legend(); ax.grid(axis='y', alpha=.3)
ax.set_title('Experiment log'); plt.tight_layout(); plt.show()
""")

# --------------------------------------------------------------------------
md(r"""
### Evidence for the key changes
**(a) Learning curves.** Seed-0 curves for the starter settings (E0), for E1 (enough updates), for E5 (no augmentation) and for E6 (augmentation). The curves show when the network starts to learn and when the upright and rotated validation accuracies diverge.
""")
code(r"""
fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
for ax, s in zip(axes, ['E0', 'E1', 'E5', 'E6']):
    h = RUNS[s]['hists'][0]; ep = np.arange(1, len(h['train_loss']) + 1)
    ax.plot(ep, h['val_acc'], label='val acc (upright)'); ax.plot(ep, h['val_rot_acc'], label='val acc (rotated)')
    ax2 = ax.twinx(); ax2.plot(ep, h['train_loss'], 'k--', lw=1, label='train loss'); ax2.set_ylabel('train loss')
    ax.set_ylim(0, 1); ax.set_xlabel('epoch'); ax.set_title(s); ax.legend(loc='center right', fontsize=7)
plt.tight_layout(); plt.show()
""")
md(r"""
**(b) What the CNN is and is not invariant to.** The seed-0 models from E5 (no augmentation) and E6 (augmentation) classify the validation leaves after a fixed rotation. A convolution slides the same filter over every position, and pooling summarises over neighbourhoods, so a CNN copes reasonably well with *shifts*. Nothing in the architecture handles *rotation*: the model has to learn it from rotated examples.
""")
code(r"""
angles = list(range(-180, 181, 30))
fig, ax = plt.subplots(figsize=(8, 3.8))
for s in ['E5', 'E6']:
    model = RUNS[s]['models'][0]
    acc = [accuracy_of(model, torch.stack([rotate_shift(x, a, 0, 0) for x in val_x]), val_y) for a in angles]
    ax.plot(angles, acc, marker='o', label=f"{s} ({'with' if RUNS[s]['cfg']['aug'] else 'no'} augmentation)")
shift_acc = [accuracy_of(RUNS['E5']['models'][0], torch.stack([rotate_shift(x, 0, d, d) for x in val_x]), val_y) for d in (0, 5, 10)]
ax.set_xlabel('rotation applied to validation leaves (deg)'); ax.set_ylabel('accuracy'); ax.set_ylim(0, 1.02)
ax.set_xticks(angles); ax.grid(alpha=.3); ax.legend(); ax.set_title('Accuracy vs rotation (seed 0)'); plt.tight_layout(); plt.show()
print('E5 (no augmentation) accuracy after shifting by 0 / 5 / 10 px:', [round(a, 3) for a in shift_acc])
""")

# --------------------------------------------------------------------------
md(r"""
## 7. Final model
**Final configuration:** ImageNet-pretrained ResNet-18 (torchvision `IMAGENET1K_V1`), with a new 15-way linear layer and the whole network fine-tuned.
* Preprocessing: 128 px images upsampled to 224 px, ImageNet mean/std standardisation.
* Augmentation: rotation by any angle, shift of up to 10 px, random left-right mirror.
* Optimiser and schedule: Adam, learning rate 3e-4, weight decay 1e-4, cosine decay over 25 epochs, batch size 32.
* Test-time augmentation: average over 8 rotations/mirrors.

Its validation numbers are row P3 of the log (3 seeds). Below: the confusion matrix on the rotated validation set (all 3 seeds pooled), then misclassified examples.
""")
code(r"""
from sklearn.metrics import confusion_matrix
P3 = RUNS['P3']
print(f"Final config, 3 seeds: validation {np.mean(P3['val']):.3f} ± {np.std(P3['val']):.3f}, "
      f"rotated validation {np.mean(P3['val_rot']):.3f} ± {np.std(P3['val_rot']):.3f}")
preds = [predict_proba(m, val_x_rot, tta=True).argmax(1).cpu().numpy() for m in P3['models']]
truth = np.tile(val_y.cpu().numpy(), len(preds)); pred_all = np.concatenate(preds)
cm = confusion_matrix(truth, pred_all, labels=range(15))
fig, ax = plt.subplots(figsize=(7.5, 6.5))
ax.imshow(cm, cmap='Blues')
for i in range(15):
    for j in range(15):
        if cm[i, j]: ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=7, color='w' if cm[i, j] > cm.max() / 2 else 'k')
ax.set_xticks(range(15)); ax.set_yticks(range(15))
ax.set_xticklabels([c[-2:] for c in CLASSES]); ax.set_yticklabels(CLASSES, fontsize=8)
ax.set_xlabel('predicted species'); ax.set_ylabel('true species')
ax.set_title('Rotated validation set, final config, 3 seeds pooled (8 x 3 = 24 per row)'); plt.tight_layout(); plt.show()
off = sorted([(int(cm[i, j]), CLASSES[i], CLASSES[j]) for i in range(15) for j in range(15) if i != j and cm[i, j]], reverse=True)
print('most frequent confusions (count, true -> predicted):', off[:8])

wrong = np.where(preds[0] != val_y.cpu().numpy())[0]
print(f'seed-0 model misclassifies {len(wrong)} of {len(val_y)} rotated validation images')
if len(wrong):
    fig, axes = plt.subplots(1, min(8, len(wrong)), figsize=(2.2 * min(8, len(wrong)), 2.8), squeeze=False)
    for ax, k in zip(axes[0], wrong[:8]):
        ax.imshow(val_x_rot[k].cpu().permute(1, 2, 0)); ax.axis('off')
        ax.set_title(f'true {CLASSES[val_y[k]][-2:]}\npred {CLASSES[preds[0][k]][-2:]}', fontsize=8, color='r')
    plt.suptitle('Misclassified rotated validation images (seed 0)'); plt.tight_layout(); plt.show()
""")
md(r"""
*(Interpretation of the confusion matrix and the misclassified examples: see the summary.)*

### Predictions for the test set
The final configuration is retrained on **all 600 labelled images** with each of the 3 seeds (same settings, no validation hold-out). The test prediction averages the 3 models' TTA probabilities. The agreement between the individual seeds shows how stable the predictions are.
""")
code(r"""
final_models = []
for s in SEEDS:
    m, _ = train_model(dict(FINAL_CFG, track=False), s, np.arange(len(y_all)))
    final_models.append(m)
X_te_f = X_te_d.float() / 255
probs = [predict_proba(m, X_te_f, tta=True) for m in final_models]
test_pred = torch.stack(probs).mean(0).argmax(1).cpu().numpy()
for s, p in zip(SEEDS, probs):
    print(f'seed {s}: agrees with the ensemble on {(p.argmax(1).cpu().numpy() == test_pred).mean():.3f} of test images')

pred_df = pd.DataFrame({'id': sub_df.id, 'label': [CLASSES[k] for k in test_pred]})
pred_df.to_csv('predictions.csv', index=False)
check = pd.read_csv('predictions.csv')
assert list(check.columns) == ['id', 'label'] and len(check) == 525 and check.id.tolist() == sub_df.id.tolist()
print(check.shape); display(check.head())
print('predicted test images per class:', check.label.value_counts().sort_index().to_dict())
""")

# --------------------------------------------------------------------------
md(r"""
## 8. Summary
*(Filled in from the outputs above after the Colab run.)*

## References and code sources
* Starter notebook `LeafID_starter.ipynb` (course material): Section 2 is copied from it unchanged, and `SmallCNN` keeps its architecture.
* PyTorch and torchvision documentation (models, `transforms.v2.functional.affine`, optimisers, LR schedulers): https://pytorch.org/docs, https://pytorch.org/vision
* K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016 (ResNet). Pretrained weights: torchvision `ResNet18_Weights.IMAGENET1K_V1`.
* S. Ioffe, C. Szegedy. *Batch Normalization.* ICML 2015.
* D. P. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015.
* I. Loshchilov, F. Hutter. *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR 2017 (cosine schedule).
* AI coding assistant (Claude) used as a coding aid; all code was checked and run by me.
""")

# ==========================================================================
nb = nbf.v4.new_notebook()
nb.cells = CELLS
nb.metadata = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
               "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "LeafID_Rizwan_Mir.ipynb"
nbf.write(nb, out)
print("wrote", out, f"({len(CELLS)} cells)")
