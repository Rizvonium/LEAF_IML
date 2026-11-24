# Viva notes: LeafID (Assignment 2)

The viva is 25 of the 30 marks. If the TF decides you don't understand your own code, you get **0 out of 25**.
Read these notes next to the notebook and make sure you can explain every cell in your own words.
Numbers marked (see section 0) must be copied from your Colab outputs. Numbers in *italics* come from a local CPU run of the same code; they are only indicative.


## 0. Your numbers (Kaggle T4 GPU run, 3 seeds, mean ± sd)

| Step | Change | Upright val | Rotated (test-like) val |
|---|---|---|---|
| Starter | unchanged recipe | 0.067 ± 0.014 (train 0.067 ± 0.004) | n/a |
| E0 | starter settings in our pipeline | 0.067 | 0.067 |
| E1 | batch 256 → 32 | 0.089 | 0.083 |
| E2 | 5 → 40 epochs | 0.106 | 0.103 |
| **E3** | **Adam (first change that makes it learn)** | **0.639 ± 0.059** | 0.450 |
| E4 | 64 → 128 px | 0.567 ± 0.046 (no gain, noise) | 0.436 |
| E5 | standardise inputs | 0.731 ± 0.041 | 0.483 |
| E6 | batch norm | 0.786 ± 0.233 | 0.558 ± 0.217 |
| **E7** | **rotation + mirror augmentation** | 0.644 | **0.631 ± 0.168 (flat across angles)** |
| E8 | 80 epochs | 0.875 | 0.864 ± 0.017 |
| E9 | cosine LR | 0.867 | 0.864 ± 0.010 |
| E10 | width 32 | 0.900 | 0.906 ± 0.010 |
| E11 | TTA (small CNN final) | 0.900 ± 0.014 | **0.897 ± 0.005** |
| P1 | pretrained ResNet-18, no aug | 1.000 | **0.783 ± 0.017** |
| P2 | + augmentation | 1.000 | 0.997 ± 0.005 |
| **P3** | **+ TTA (final)** | 1.000 | **1.000 ± 0.000 (0 errors)** |

Other facts to quote:
- Run time 14.9 min (first run 16.1 min) on a Kaggle T4; outputs come from Kaggle because Colab gave no GPU that evening.
- Data: 92% of training leaves near-vertical vs 33% of test leaves; stem down 74% vs 52% (among near-vertical); centre distance 4.9 vs 4.7 px; sharpness 0.131 vs 0.131.
- Rotation: bilinear makes sharpness 0.093, bicubic 0.117. E6 (no augmentation) accuracy: 0.79 at 0°, about 0.37 at ±90°; E7 flat around 0.62–0.66.
- Shift test, E6: 0.786 / 0.800 / 0.786 for 0 / 5 / 10 px. Shifts barely matter.
- Final model: 0 errors on rotated validation (all seeds); test predictions: seeds agree on 99.4%, 34–36 per species.
- Small CNN errors (E11): 12.3 ± 0.6 of 120 per seed; pairs 09 → 11 (9), 01 ↔ 03 (7 + 6), some → 15; only 6 of 37 fixed by showing the leaf upright, so they come from the leaf (look-alike shape, folded bases).

---

## 1. Four-minute presentation outline

1. **Problem (20 s).** 15 species, 600 labelled scans (40 per species), 525 test scans, all 128×128 RGB.
2. **Starter (40 s).** Run unchanged: training accuracy ≈ validation accuracy ≈ chance (1/15 = 0.067). It predicts the same species for all 525 test images. *Locally: train 0.065, val 0.075.* With 3 seeds: (see section 0).
   It is **under-fitting**: it has hardly learned anything.
   Causes: only 10 updates (480 images ÷ batch 256 = 2 per epoch × 5 epochs), with plain SGD and no momentum, on inputs that are neither standardised nor normalised by batch norm.
3. **Data finding (60 s).** Training leaves are upright; test leaves are rotated about the image centre to any angle.
   Evidence:
   - ~92% of training leaves have their long axis within 30° of vertical, against ~33% of test leaves (uniform angles would give 1/3);
   - among near-vertical leaves, the stem points down in most training images but in only about half of the test images, so test leaves are upside down as often as not;
   - the distance of the leaf from the image centre is the same in train and test (median ~4.8 px), so there is no extra shift: test images are training-style scans rotated about their centre;
   - size, colour, background and **sharpness** are identical (mean |Laplacian| on leaf pixels: 0.131 in both). So the test leaves were rotated without blurring them.

   Consequences: (a) train with 360° rotation augmentation; (b) upright validation accuracy can overestimate test accuracy, so I also score every experiment on a **rotated validation set**.
4. **Experiment log (60 s).** One change per row, 3 seeds each, on the starter's own small CNN:
   - E1 batch 32, then E2 40 epochs: more updates, but still near chance (*E2 ≈ 0.11*);
   - E3 Adam: **the first change that makes it learn** (*≈ 0.56 upright*);
   - E4 128 px, E5 standardisation, E6 batch norm: upright accuracy climbs (*≈ 0.80*), but rotated validation stays around *0.45*;
   - E7 rotation augmentation: **the big jump on rotated validation**, *0.48 → 0.67*, with a much smaller spread between seeds;
   - E8–E10 longer training, cosine schedule, wider network; E11 TTA.

   Then an ImageNet-pretrained ResNet-18: P1 (no augmentation) is excellent upright but drops on rotated leaves; P2 adds augmentation; P3 adds TTA.
5. **Result (40 s).** Final model P3: rotated validation (see section 0) (mean ± sd, 3 seeds). Small CNN final E11: (see section 0).
   Confusion matrix: the remaining errors are between look-alike species (small CNN: 09/11, 01/03; final model: none). Say what the misclassified images show; only claim "similar outline" if the images show it.

---

## 2. Notebook walk-through (what each section does, where things happen)

| Section | What it does | Things you may be asked to point at |
|---|---|---|
| top | `T_START` timer | |
| 1. Starter unchanged | the starter's 6 code cells, untouched; a cell recording train/val accuracy; a cell repeating the starter's exact recipe with seeds 0, 1, 2 | `EPOCHS = 5`, `BATCH_SIZE = 256`, `LR = 0.05`, `IMG_SIZE = 64`, the `SGD` line |
| 2. Data | loads every image once as a `uint8` tensor `N×3×128×128` (`load_folder`), prints counts/sizes, example grids, measures each leaf (`leaf_measurements`) | the leaf mask: pixels that differ from the image's flat background colour (top-left pixel) by > 40; the angle: main eigenvector of the covariance of leaf-pixel coordinates; stem direction: sign of the third moment along that axis |
| 3. Diagnosis | hypotheses, each linked to a log row; prints updates per epoch, the starter's uneven validation class counts, the mean pixel value | |
| 4. Shared code | stratified split (`val_idx`, 8 per species), `rotate_batch` (bicubic rotation, prints the sharpness comparison), `augment`, `val_x` / `val_x_rot`, `SmallCNN`, `Wrapped`, `build_model`, `predict_proba`, `train_model`, `run_experiment`, `tta_row` | the training loop in `train_model`: forward → `F.cross_entropy` → `opt.zero_grad()` → `loss.backward()` → `opt.step()` |
| 5. Experiment log | rows E0–E11, one change per row | each `cfg = dict(cfg, key=value)` copies the previous config and changes one key |
| 6. Pretrained | P1–P3, torchvision ResNet-18 | `net.fc = nn.Linear(net.fc.in_features, 15)` (in_features = 512 for ResNet-18) replaces the 1000-class ImageNet layer |
| evidence | learning curves (3 seeds), accuracy vs rotation (E6 vs E7), shift test | |
| 7. Final | confusion matrix, misclassified images (rotated / upright / a training example of the predicted class), retrain on all 600 images with 3 seeds, ensemble + TTA, write and check `predictions.csv` | the `assert` checking 525 rows, the `id,label` header and the id order |
| 8. Summary | prints every key number as mean ± sd; total run time | |

**Flow:** images → `uint8` tensors on the GPU → each batch converted to float in [0,1] → (training only) random rotation + mirror →
`Wrapped` resizes if needed (64 px for E0–E3, 224 px for ResNet) and standardises (data mean/std from E5 on; ImageNet mean/std for ResNet) → network → logits → cross-entropy loss.
Prediction: softmax probabilities, averaged over the 8 TTA views, then over the 3 seeds → `argmax` → species name.

---

## 3. Why the starter performs poorly (each point tested by a log row)

1. **10 updates in total** (batch 256, 480 images, 5 epochs) ⇒ under-fitting: train ≈ val ≈ chance, one class predicted for everything.
   More updates (E1 batch 32; E2 40 epochs = 600 updates) are *necessary but not enough*: *E2 is still ≈ 0.11*.
2. **Plain SGD at lr 0.05 without momentum.** Switching to **Adam (E3) is the first change that makes it learn**. Adam keeps running averages of each parameter's gradient and squared gradient, and scales each step by them, so every parameter gets a sensibly sized step.
3. **64 px instead of 128 px** (E4): throws away margin teeth and veins. For the small CNN the measured gain was small (*≈ +1–2 points*, within seed noise). Say that honestly.
4. **No input standardisation / no batch norm** (E5, E6). Inputs are mostly bright background (mean ≈ 0.85), so every input is far from zero. Standardising to mean 0 and sd 1 suits the default initialisation; batch norm keeps each layer's activations at a stable scale during training.
5. **Train/test mismatch** (E7; P1 vs P2): upright training leaves, rotated test leaves.
6. *(Measurement, not performance)* The starter's validation split is random, not stratified (its class counts range from 3 to 15), uses one seed, and is upright only, so its number is a poor guide to the test set.

---

## 4. Concepts you must be able to explain

- **Convolution:** a small filter (3×3 here) slides over the image and computes a weighted sum at every position. The same weights are used everywhere (*weight sharing*), so the layer detects a pattern wherever it appears. `padding=1` keeps the image size.
- **ReLU:** max(0, x), the non-linearity. Without it, stacked layers collapse into one linear map.
- **Max pooling (2×2):** keeps the largest value in each 2×2 block. It halves the resolution, gives a little tolerance to small shifts, and widens what later layers can see.
- **Global average pooling** (`adaptive_avg_pool2d(..., 1)`): averages each channel over the whole map, giving one number per channel, so the classifier doesn't care *where* a feature occurred.
- **Batch normalisation:** for each channel, normalise with the batch mean and standard deviation, then apply a learned scale and shift. Benefit: activations stay well scaled, so training is faster and more stable. In `train()` mode it uses the current batch's statistics; in `eval()` mode it uses running averages collected during training. That's why `model.eval()` must be called before predicting; otherwise a prediction depends on the other images in the batch.
- **What a CNN is invariant to:** roughly to **translation**, because convolution is translation-*equivariant* (a shifted input gives a shifted feature map) and global average pooling discards position. Not exactly, because of the stride-2 pooling and the image border. Evidence: the E6 model's accuracy barely changes when leaves are shifted by 5–10 px (see section 0).
  **It is not invariant to rotation, scale or mirroring.** Evidence: in the accuracy-vs-rotation plot, E6 (no augmentation) is best near 0° and drops as leaves are rotated, while E7 (augmentation) is flat (see section 0). P1 vs P2 shows the same for the pretrained network. Accuracy doesn't fall to chance, because colour, texture and margin cues work at any angle.
- **Data augmentation:** random, label-preserving changes to training images: rotation by any angle about the centre (corners filled with that image's background colour) and a left-right mirror. A fresh draw is made for every image in every epoch. The training distribution then covers the test distribution, and augmentation also regularises (the model never sees exactly the same image twice).
  *Why no up-down flip?* Rotating by 180° plus a left-right mirror already produces it.
  *Why no shift?* The test leaves are not shifted (same distance from the centre as in training).
- **Test-time augmentation (TTA):** average the predicted probabilities over the 8 symmetries of a square (rotations by 0/90/180/270°, with and without a mirror). These are exact pixel rearrangements: no interpolation, no new corners.
- **Transfer learning:** ResNet-18 was trained on 1.28 M ImageNet photos. Its layers already detect edges, textures and shapes. We replace only the last layer (1000 → 15 classes) and fine-tune the whole network with a small learning rate (3e-4), so the pretrained features are adapted rather than destroyed. Images are upsampled 128 → 224 px because that is the scale the backbone was trained at, and standardised with ImageNet mean/std because those were its training statistics.
- **Cosine learning-rate schedule:** the learning rate decays smoothly from its starting value towards 0: big steps early, small steps at the end to settle.
- **Weight decay (1e-4):** pulls the weights slightly towards 0 at every step (L2 regularisation).
- **Cross-entropy loss:** −log(probability the model gives to the true class). `F.cross_entropy` applies softmax internally, so the network outputs raw logits.

---

## 5. Validation: what it can and cannot tell you

- **Split:** stratified, 8 validation images per species (120 in total), the same for every experiment, so rows are comparable.
- **Seeds:** every row is trained with seeds 0, 1 and 2. The seed controls the weight initialisation, batch order and augmentation draws.
  Reported as **mean ± sample standard deviation** (`np.std(..., ddof=1)`). With only 3 seeds, the sd itself is a rough estimate.
- **Two validation numbers:**
  - *upright validation*: same distribution as training; it measures learning but can overestimate test accuracy;
  - *rotated validation*: the same 120 images rotated about their centre by random angles (fixed seed), built the same way the test images appear to be; this is my estimate of test accuracy.
- **What it cannot tell you:**
  - 120 images is small. The seed sd covers only training randomness. A different set of 120 validation images would move accuracy by roughly ±2 points at 95% (binomial standard error √(p(1−p)/120)), or ±4.6 at 50%. All rows use the same images, so comparisons between rows are fairer than that, but differences of 1–2 points are not conclusive.
  - My rotated set imitates the test images; it is not identical to them.
  - I chose settings by looking at these numbers, so the final validation number is slightly optimistic.
- **Final models:** after choosing settings, I retrain on all 600 images with each of the 3 seeds. There is no hold-out, so these exact models have no validation number; P3 is their estimate. The test prediction averages the 3 models; pairwise seed agreement on the test set is printed as a stability check (see section 0).

---

## 6. Likely viva questions, with short answers

- **Why was the starter so bad?** Ten plain-SGD updates: it under-fits (train ≈ val ≈ chance, one class predicted for all test images). More updates alone barely helped (E2); Adam made it learn (E3); rotation augmentation fixed the train/test mismatch (E7).
- **Which change first made it learn?** E3, Adam. E1/E2 (600 plain-SGD updates) only reached about 0.11 (see section 0).
- **How did you find the rotation problem?** By looking at the test images, then measuring each leaf's long-axis angle and stem direction: training angles cluster near 0° with stems down; test angles are spread evenly and stems point down only half the time.
- **Why rotate the validation set?** The test images are rotated. Upright validation said the model was improving at E5/E6 while rotated validation stayed flat (compare the two bars in the log plot).
- **Why 360° and not ±20°?** The axis-angle histogram is flat and the stem direction is 50/50 on test images, so test leaves point every way, including upside down.
- **Why bicubic interpolation for the rotation?** Rotating moves pixels to non-integer positions, so new pixel values must be interpolated. Bilinear interpolation averages the 4 nearest pixels and visibly blurs the leaf: sharpness 0.131 drops to 0.093. Bicubic uses 16 neighbours with a sharper kernel: 0.117, much closer to the test images' 0.131. I first used bilinear, then noticed that augmented models scored *higher* on rotated (blurred) validation images than on sharp upright ones, measured the sharpness and switched. The notebook prints the comparison in section 4.
- **How does `rotate_batch` work?** For each image a 2×3 matrix `theta = [[cos, −sin, shift], [sin, cos, 0]]`. `F.affine_grid` turns it into, for every output pixel, the input position to read from, in [−1, 1] coordinates. `F.grid_sample` reads those positions (bicubic). Positions outside the image read 0; because we rotate *image − background* and add the background back, those corners become background-coloured.
- **What fills the corners after rotation?** That image's own background colour (each image has one flat background colour; `bg = x[:, :, :1, :1]`). `rotate_batch` rotates *(image − bg)* with zero padding and adds bg back, so the corners come out exactly background-coloured. (An earlier version using torchvision's `fill=` option left a thin dark seam along the old image edge, a cue the test images don't have; I found it by looking at the rotated images.)
- **Why batch size 32?** More updates per epoch (15 instead of 2), and the gradient noise helps generalisation. It's still large enough for stable batch-norm statistics.
- **Why 80 epochs and a cosine schedule?** Augmented images never repeat exactly, so the model needs more passes; decaying the learning rate at the end lets it settle (E8, E9 (see section 0)).
- **Why ResNet-18 and not something bigger?** It is small (11 M parameters), quick on a T4, and already (see section 0) accurate. The handout asks for understanding, not the largest model.
- **Did pretraining alone fix rotation?** No: P1 (pretrained, no augmentation) is near-perfect upright but much worse on rotated validation (see section 0). Augmentation (P2) is what fixes it.
- **What does `torch.no_grad()` do?** It stops PyTorch recording operations for backpropagation: less memory, faster prediction.
- **How do you ensure reproducibility?** `set_seed` seeds Python, NumPy and PyTorch (CPU and GPU), cuDNN is set to deterministic, and the split and rotated validation set use fixed seeds.
- **Where does the model still fail?** The final model makes no validation errors; the small CNN confuses 09 → 11 and 01 ↔ 03 (similar ovate/elliptic toothed leaves), and only 6 of its 37 errors are orientation errors. If the upright version of a misclassified leaf is classified correctly, that error is due to orientation; otherwise it's down to the leaf itself (damaged, unusual shape, look-alike species).
- **What would you try next?** More seeds or k-fold cross-validation for tighter error bars, a larger pretrained backbone, rotation-equivariant layers, or training the small CNN longer.

### Small code changes you might be asked to make live
- Change the number of epochs of a row: edit `epochs=` in that row's `cfg = dict(cfg, ...)`. Rows build on the row above, so the change carries into later rows until one sets that key again.
- Turn off augmentation: `aug=False`.
- Accuracy of one model at a specific rotation: `accuracy_of(RUNS['E7']['models'][0], rotate_batch(val_x, torch.full((len(val_x),), 90.0, device=DEV)), val_y)`.
- One test image with the final ensemble: `CLASSES[torch.stack([predict_proba(m, X_te_f[:1], tta=True) for m in final_models]).mean(0).argmax(1).item()]`.
- One rotated validation image with a P3 model: `CLASSES[predict_proba(RUNS['P3']['models'][0], val_x_rot[:1], tta=True).argmax(1).item()]`.
