# LeafID: tree species from scanned leaves (Assignment 2)

| file | what |
|---|---|
| `LeafID_Rizwan_Mir.ipynb` | **the submission notebook**: starter run unchanged → data examination → diagnosis → experiment log (small CNN, one change per row, 3 seeds) → ImageNet-pretrained ResNet-18 → final model, confusion matrix, misclassified images → `predictions.csv` |
| `tools/build_notebook.py` | generates the notebook from plain-Python cell sources (edit there, then `python tools/build_notebook.py`) |
| `starter/` | the course's starter notebook (its code cells are copied unchanged into section 1 of the notebook) |

## Run on Colab
1. Runtime → Change runtime type → **T4 GPU**
2. Upload `leaves.zip` to `/content` (Files panel)
3. Runtime → **Run all**. It writes `predictions.csv` (header `id,label`, 525 rows).

## Key finding about the data
Training leaves are scanned **upright** (stem down); test leaves are **rotated about the image centre to any angle**.
About 9 in 10 training leaves lie within 30° of vertical, against about 1 in 3 test leaves (what uniformly random angles give).
Among near-vertical test leaves the stem points down only about half the time.
The leaf's distance from the image centre, its size, colour and background are the same in both sets.
A CNN is not rotation-invariant, so the model is trained with full-360° rotation (plus mirror) augmentation.
Every experiment is also scored on a *rotated validation set* that mimics the test images, because plain
(upright) validation accuracy overestimates test accuracy.
