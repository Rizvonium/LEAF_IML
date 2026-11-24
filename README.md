# LeafID: tree species from scanned leaves (Assignment 2)

CNN trained **from scratch** (no pretrained weights, no external data) on 600 labelled leaf scans
(15 species × 40) to predict the species of 525 unlabelled test scans.

## Files
| file | what |
|---|---|
| `LeafIDMir.ipynb` | the submission notebook: baseline → diagnosis → one-change-at-a-time ablation → evidence → final ensemble → `predictions.csv` |
| `tools/build_notebook.py` | generates the notebook from plain-Python cell sources (edit here, then rebuild) |
| `tools/make_synthetic_leaves.py` | synthetic stand-in data, only for smoke-testing the pipeline before the real data is available |

## Run on Colab
1. Runtime → Change runtime type → **T4 GPU**
2. Upload `leaves.zip` to `/content` (Files panel)
3. Runtime → **Run all**. Writes `predictions.csv`.

## Rebuild / smoke-test locally
```bash
python tools/build_notebook.py
python tools/make_synthetic_leaves.py /tmp/synth/leaves.zip
LEAF_QUICK=1 LEAF_ZIP=/tmp/synth/leaves.zip LEAF_DIR=/tmp/synth/data jupyter nbconvert --to notebook --execute LeafIDMir.ipynb --output /tmp/synth/executed_quick.ipynb
```

## Method in one paragraph
Every configuration is scored with the same stratified 5-fold cross-validation, so changes can be compared fold by fold.
Starting from the starter's design, each ablation row adds a single change, in this order:
input standardisation → crop-to-leaf + aspect-preserving pad → BatchNorm/residual/global-average-pool network →
geometric augmentation → AdamW + one-cycle LR + label smoothing → higher resolution → test-time augmentation.
Each change comes with a specific piece of evidence: learning curves, accuracy vs rotation angle, confusion matrix, Grad-CAM.
