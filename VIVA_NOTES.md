# Viva notes: LeafID (Assignment 2)

The viva is 25 of the 30 marks. If the TF decides you don't understand your own code, you get **0 out of 25**.
Read these notes next to the notebook and make sure you can explain every cell in your own words.
Numbers marked **[from run]** are filled in from the Colab outputs.

---

## 1. Four-minute presentation outline

1. **Problem (20 s).** 15 species, 600 labelled scans (40 per species), 525 test scans, all 128×128 RGB.
2. **Starter (40 s).** Run unchanged it gets validation accuracy **[from run]**, close to chance (1/15 = 6.7%).
   Its training accuracy is just as low, so it is **under-fitting**: it has barely learned anything.
   The main cause: 480 images ÷ batch 256 = 2 updates per epoch × 5 epochs = **10 gradient updates** in total, with plain SGD.
3. **Data finding (60 s).** Training leaves are upright and centred. Test leaves are rotated to any angle and shifted.
   Show the orientation histogram: ~92% of training leaves lie within 30° of vertical, against ~33% of test leaves.
   Size, colour and background match, so the only difference is geometric.
   Two consequences: (a) train with 360° rotation augmentation; (b) upright validation accuracy will overestimate
   test accuracy, so I built a **rotated validation set** that mimics the test images.
4. **Experiment log (60 s).** One change per row, 3 seeds each, on the starter's small CNN:
   more updates → Adam → 128 px → input standardisation → batch norm → **rotation augmentation (the big jump on rotated validation)** → longer training with a cosine schedule → wider → TTA.
   Then an ImageNet-pretrained ResNet-18 with the same fixes.
5. **Result (40 s).** Final model: ResNet-18 + augmentation + TTA. Rotated-validation accuracy **[from run]** (mean ± sd over 3 seeds).
   Confusion matrix: the remaining errors are between species **[from run]**, which look alike (similar outline/margin).

---

## 2. Notebook walk-through (what each section does, where things happen)

| Section | What it does | Things you may be asked to point at |
|---|---|---|
| 1. Starter unchanged | the starter's 6 code cells, untouched; then a cell that records train/val accuracy | `EPOCHS = 5`, `BATCH_SIZE = 256`, `LR = 0.05`, `IMG_SIZE = 64`, the `SGD` line |
| 2. Data | loads every image once as a `uint8` tensor `N×3×128×128` (`load_folder`), prints class counts/sizes, shows example grids, measures orientation/position/size (`leaf_measurements`) | how the leaf mask is made (pixels that differ from the border colour by > 40), how the angle is computed (main eigenvector of the covariance of leaf pixel coordinates) |
| 3. Diagnosis | prints updates per epoch, starter's validation class counts, 64 vs 128 px | why 10 updates is too few |
| 4. Shared code | stratified split (`val_idx`, 8 per species), `rotate_shift`, `augment`, the rotated validation set `val_x_rot`, `SmallCNN`, `Wrapped`, `build_model`, `predict_proba`, `train_model`, `run_experiment`, `tta_row` | the training loop inside `train_model`: forward → `F.cross_entropy` → `opt.zero_grad()` → `loss.backward()` → `opt.step()` |
| 5. Experiment log | rows E0–E9 (small CNN), one change per row | the `cfg = dict(cfg, ...)` line of each row: it copies the previous config and changes one key |
| 6. Pretrained | rows P1–P3 with torchvision ResNet-18 | `net.fc = nn.Linear(512, 15)`: replaces the 1000-class ImageNet layer |
| 7. Final | confusion matrix, misclassified images, retrain on all 600 images with 3 seeds, ensemble + TTA, write and check `predictions.csv` | the `assert` that checks 525 rows and the `id,label` header |

**Flow:** images → `uint8` tensors on the GPU → each batch converted to float in [0,1] → (training only) random rotation/shift/mirror →
`Wrapped` resizes (64/128/224) and standardises → network → logits → cross-entropy loss.
For prediction: softmax probabilities, averaged over the 8 TTA views, then over the 3 seeds → `argmax` → species name.

---

## 3. Why the starter performs poorly (in order of importance)

1. **10 updates in total** (batch 256, 480 images, 5 epochs). Training accuracy ≈ validation accuracy ≈ near chance ⇒ under-fitting.
   *The first change that made it learn:* E1, batch 32 and 40 epochs = 15 × 40 = **600 updates**.
2. **Plain SGD at lr 0.05 without momentum**: slow progress. Adam (E2) scales each parameter's step by a running estimate of its gradient size.
3. **64 px instead of 128 px**: throws away margin teeth, veins and the thin stem.
4. **No input standardisation, no batch norm**: activation scales drift from layer to layer, which slows and destabilises training.
5. **Train/test mismatch**: upright training leaves, rotated test leaves. Even a well-trained model fails on rotated leaves without rotation augmentation.
6. **Validation**: a random, unstratified 20% split (the class counts are uneven, printed in section 3), one seed, upright images only.

---

## 4. Concepts you must be able to explain

- **Convolution:** a small filter (here 3×3) slides over the image and computes a weighted sum at every position. The same weights are used everywhere (*weight sharing*), so the layer detects a pattern wherever it appears. `padding=1` keeps the size.
- **ReLU:** max(0, x), the non-linearity. Without it, stacked layers collapse into one linear map.
- **Max pooling (2×2):** keeps the largest value in each 2×2 block. It halves the resolution, gives a little tolerance to small shifts, and widens what later layers can see.
- **Global average pooling** (`adaptive_avg_pool2d(..., 1)`): averages each channel over the whole map, giving one number per channel. The classifier therefore doesn't care *where* a feature occurred.
- **Batch normalisation:** for each channel, subtract the batch mean and divide by the batch standard deviation, then apply a learned scale and shift. In `eval()` mode it uses running averages instead. Benefit: activations stay well scaled, so training is faster and more stable at a given learning rate. That's why `model.eval()` matters at test time.
- **What a CNN is invariant to:** roughly to **translation**, because convolution is translation-*equivariant* and pooling/GAP make the output roughly translation-*invariant*. Evidence: the E5 model barely changes when leaves are shifted by 5–10 px (section 5 evidence (b)).
  **It is not invariant to rotation, scale or mirroring.** Evidence: the E5 model's accuracy drops sharply once a leaf is rotated away from upright (the accuracy-vs-rotation plot). Rotation invariance has to be learned from data, which is what augmentation provides.
- **Data augmentation:** random label-preserving changes to training images (rotation 0–360°, shift ±10 px, left-right mirror), drawn fresh for every image in every epoch. It makes the training distribution cover the test distribution, and acts as a regulariser (the model sees "new" images each epoch).
  *Why not an up-down flip?* Rotation by 180° plus a left-right mirror already produces it, so it adds nothing.
- **Test-time augmentation (TTA):** average the predicted probabilities over the 8 symmetries of a square (rotations by 0/90/180/270° with and without a mirror). These are exact pixel rearrangements: no interpolation, no new corners.
- **Transfer learning:** the ResNet-18 was trained on 1.28 M ImageNet photos. Its early layers already detect edges, textures and shapes. We replace only the last layer (1000 → 15 classes) and fine-tune the whole network with a small learning rate (3e-4), so the pretrained features are adapted rather than destroyed. Images are upsampled 128 → 224 px because that is the scale the backbone was trained at, and standardised with ImageNet's mean/std because those were the statistics in training.
- **Adam:** keeps running averages of each parameter's gradient (momentum) and squared gradient, and divides the step by the square root of the latter, so every parameter gets a sensibly sized step.
- **Cosine learning-rate schedule:** the learning rate decays smoothly from its initial value to ~0 over training: big steps early to make progress, small steps at the end to settle.
- **Weight decay (1e-4):** a small pull of the weights towards 0 (L2 regularisation).
- **Cross-entropy loss:** −log(probability the model gives to the true class). `F.cross_entropy` applies softmax internally, which is why the model outputs raw logits.

---

## 5. Validation: what it can and cannot tell you

- **Split:** stratified, 8 validation images per species (120 in total), the same split for every experiment, so rows are directly comparable.
- **Seeds:** every row is trained with seeds 0, 1 and 2. The seed controls the weight initialisation, batch order and augmentation draws. I report mean ± standard deviation.
- **Two validation numbers:**
  - *upright validation* = same distribution as the training images; it measures learning, but it overestimates test accuracy;
  - *rotated validation* = the same 120 images, rotated by random angles and shifted (fixed seed); it mimics the test images and is my estimate of test accuracy.
- **What it cannot tell you:**
  - 120 images is small: one image is 0.83 percentage points, so differences of 1–2 points between rows are within noise. That's why I look at the seed spread.
  - My rotated set imitates the test images; it is not identical to them. Their rotation/fill details may differ slightly.
  - I chose settings by looking at these validation numbers, so the final number is slightly optimistic.
- **Final model:** after choosing settings, I retrain on all 600 images with each of the 3 seeds (no hold-out, so there's no validation number for these exact models). The test prediction averages the 3 models. Agreement between seeds is printed as a stability check.

---

## 6. Likely viva questions, with short answers

- **Why was the starter so bad?** Ten gradient updates with plain SGD: it under-fits (train ≈ val ≈ chance). Then: 64 px resolution, no normalisation or batch norm, and no rotation augmentation even though test leaves are rotated.
- **Which change first made it learn?** E1: batch 32 and 40 epochs (600 updates instead of 10).
- **How did you find the rotation problem?** By looking at the test images and measuring each leaf's long-axis angle: training angles cluster near 0°, test angles are spread uniformly.
- **Why rotate the validation set?** The test images are rotated. An upright validation set would tell me the model is good when it can fail on rotated leaves (compare the two bars in the log plot).
- **Why 360° and not ±20°?** Test leaves are at every angle, including upside down (the histogram is flat).
- **What fills the corners after rotation?** The median background colour of the training images (`BG`). `rotate_shift` rotates *(image − BG)*, so the uncovered corners come out as 0 and, after adding BG back, as exactly background colour. (Using torchvision's `fill=` option instead left a thin dark seam along the old image edge, a cue the test images don't have. I found it by looking at the rotated validation images.) Black corners would likewise be an artificial cue.
- **Why batch size 32?** More updates per epoch (15 instead of 2), and the gradient noise helps generalisation. It is also big enough for stable batch-norm statistics.
- **Why a cosine schedule / 80 epochs?** Augmented data never repeats exactly, so the model needs more passes; decaying the learning rate at the end lets it settle into a minimum.
- **Why ResNet-18 and not something bigger?** It is small (11 M parameters), fast on a T4, and already **[from run]** accurate. The handout asks for understanding, not the largest model.
- **Did pretraining alone fix rotation?** No: P1 (pretrained, no augmentation) still drops on the rotated validation set **[from run]**. Augmentation is what fixes it.
- **What does `model.eval()` change?** Batch norm switches from batch statistics to stored running averages (and dropout would be turned off). Without it, predictions depend on what else is in the batch.
- **What does `torch.no_grad()` do?** It stops PyTorch from recording operations for backprop: less memory, faster prediction.
- **How do you ensure reproducibility?** `set_seed` seeds Python, NumPy and PyTorch (CPU and GPU), cuDNN is set to deterministic, and the split and the rotated validation set use fixed seeds.
- **Where does the model still fail?** See the confusion matrix: **[from run]**. These species have similar outlines; at 128 px, fine margin detail is only a few pixels wide.
- **What would you try next?** More seeds or a 5-fold cross-validation for tighter error bars, higher input resolution, a larger pretrained backbone, rotation-equivariant layers.

### Small code changes you might be asked to make live
- Change the number of epochs: edit `epochs=` in that row's `cfg = dict(cfg, ...)` line.
- Turn off augmentation: `aug=False`.
- Evaluate one model at a specific rotation: `accuracy_of(RUNS['E6']['models'][0], torch.stack([rotate_shift(x, 90, 0, 0) for x in val_x]), val_y)`.
- Show a single prediction: `CLASSES[predict_proba(model, val_x_rot[:1], tta=True).argmax(1).item()]`.
