# M3 - Custom CNN (`notebooks/models/IT25101088_CNN.ipynb`)

**Member:** IT25101088 | **Category:** Deep learning (CNN, built from scratch) | **Branch:** A (images)

## Role in the project

The only model that learns **directly from the images**. Every other model uses hand-crafted features; the
CNN learns its own filters. It is also where the **CLAHE colour correction** (this member's preprocessing
step) is tested most directly, because the network reads colour and contrast itself.

```
preprocessing.load_images(96 | 128 px, CLAHE on/off)      uint8 images
  -> Keras model:
       Rescaling(1/255)                                    group normalization step
       [augmentation: flip, rotate, zoom, brightness]      group augmentation step (training only)
       [Conv3x3 -> BatchNorm -> ReLU -> MaxPool] x 2 or 3   filters 32 / 64 / 128
       GlobalAveragePooling -> Dropout -> Dense(20, softmax)
  -> trained on train (5,575), early stopping on validation (984)
  -> best variety re-checked with 3-fold CV -> scored once on the test set (1,612)
```

## Why a CNN suits AQUA20 (viva answer)

- Images are **spatial**: convolutions slide the same small filter over the whole image, so a fin is detected
  wherever it appears (translation invariance) with far fewer weights than a dense network.
- **Hierarchical features:** early layers learn edges and colours, deeper layers learn textures and body parts.
- **Augmentation** (random flips, rotations, zooms, brightness) creates new views of the training images and
  reduces overfitting - critical with only ~5.5k training images.
- **Built for a CPU:** small images (96/128 px), 3 conv blocks, and global average pooling instead of large
  dense layers keep it to ~100k parameters.

## Implementation details

| Item | Value |
|---|---|
| Libraries | TensorFlow / Keras (`keras.Model`, `Conv2D`, `BatchNormalization`, `MaxPooling2D`, `GlobalAveragePooling2D`, `Dropout`, `Dense`) |
| Loss / optimiser | sparse categorical cross-entropy, Adam (initial learning rate 0.001, or 0.0003 in V7) |
| Batch size / epochs | 64 / up to 50 |
| Early stopping | monitor `val_loss`, patience 10, best weights restored |
| Learning-rate schedule | `ReduceLROnPlateau`: halve the learning rate when training loss stalls for 3 epochs (min 1e-5) |
| Class weights | "balanced" weights in every variety (rare species count more in the loss) |
| Seeds | `set_seeds()` before every run (42) |

**Layer-by-layer explanation**
- `Rescaling(1/255)`: normalization inside the model, so raw uint8 images can be fed in.
- `Conv2D(filters, 3x3, padding="same")`: learns `filters` small 3x3 patterns.
- `BatchNormalization`: normalizes each layer's outputs per batch - faster, more stable training.
- `ReLU`: non-linearity. `MaxPooling2D`: halves the width and height, keeping the strongest responses.
- `GlobalAveragePooling2D`: averages each feature map to one number - far fewer parameters than `Flatten`.
- `Dropout`: randomly switches off neurons during training to reduce overfitting.
- `Dense(20, softmax)`: one probability per species.

## Varieties and tuning

Manual, staged tuning (each run takes several minutes on a CPU):

| Stage | Variety | What changes |
|---|---|---|
| Preprocessing | V1 / V2 | 96 px without vs with CLAHE |
| Preprocessing | V3 | + data augmentation (on the better CLAHE setting) |
| Preprocessing | V4 | 128 px instead of 96 px |
| Hyperparameters | V5 | 2 conv blocks instead of 3 |
| Hyperparameters | V6 | dropout 0.5 instead of 0.3 |
| Hyperparameters | V7 | initial learning rate 0.0003 instead of 0.001 |

- **7 trained CNN varieties**, compared on the hold-out validation set (macro F1).
- The winner is re-trained in **stratified 3-fold cross-validation** on train + validation (fixed epochs =
  its early-stopping epoch) to confirm the score is not a lucky split.

## Results

Results of the development run (all varieties; **bold** = best variety, chosen on validation / CV - never on the test set).

| Variety | Preprocessing | Best hyperparameters | Validation macro F1 | k-fold CV macro F1 | Test macro F1 | Test accuracy | Time |
|---|---|---|---|---|---|---|---|
| V1 96px | 96px, no CLAHE, no augmentation | `{'blocks': 3, 'dropout': 0.3, 'lr': 0.001, 'class_weight': 'balanced', 'best_epoch': 18, 'epochs_run': 28}` | 0.164 | - | 0.156 | 0.329 | 10.1 min |
| V2 96px + CLAHE | 96px, CLAHE, no augmentation | `{'blocks': 3, 'dropout': 0.3, 'lr': 0.001, 'class_weight': 'balanced', 'best_epoch': 14, 'epochs_run': 24}` | 0.140 | - | 0.155 | 0.261 | 8.7 min |
| **V3 96px + aug** | 96px, no CLAHE, augmentation | `{'blocks': 3, 'dropout': 0.3, 'lr': 0.001, 'class_weight': 'balanced', 'best_epoch': 33, 'epochs_run': 43}` | 0.184 | 0.142 +/- 0.011 | 0.171 | 0.360 | 16.2 min |
| V4 128px + aug | 128px, no CLAHE, augmentation | `{'blocks': 3, 'dropout': 0.3, 'lr': 0.001, 'class_weight': 'balanced', 'best_epoch': 6, 'epochs_run': 16}` | 0.153 | - | 0.122 | 0.261 | 11.5 min |
| V5 96px + aug, 2 blocks | 96px, no CLAHE, augmentation | `{'blocks': 2, 'dropout': 0.3, 'lr': 0.001, 'class_weight': 'balanced', 'best_epoch': 16, 'epochs_run': 26}` | 0.072 | - | 0.080 | 0.241 | 8.3 min |
| V6 96px + aug, dropout 0.5 | 96px, no CLAHE, augmentation | `{'blocks': 3, 'dropout': 0.5, 'lr': 0.001, 'class_weight': 'balanced', 'best_epoch': 22, 'epochs_run': 32}` | 0.172 | - | 0.149 | 0.290 | 11.8 min |
| V7 96px + aug, lr 3e-4 | 96px, no CLAHE, augmentation | `{'blocks': 3, 'dropout': 0.3, 'lr': 0.0003, 'class_weight': 'balanced', 'best_epoch': 30, 'epochs_run': 40}` | 0.176 | - | 0.160 | 0.303 | 14.8 min |

### Conclusion

**Best variety: V3 - 96x96 px, no CLAHE, data augmentation, 3 conv blocks, dropout 0.3, learning rate 0.001,
balanced class weights** (hold-out validation macro F1 = 0.184; **3-fold CV macro F1 = 0.142 +/- 0.011**).
On the test set: **macro F1 = 0.171**, accuracy = 0.360.

| Finding | Evidence |
|---|---|
| Augmentation helped | Validation macro F1 0.164 -> 0.184 (test 0.156 -> 0.171); the model kept improving for longer (best epoch 33 vs 18) because augmented images are harder to memorise |
| CLAHE hurt | 0.164 without vs 0.140 with CLAHE |
| Depth matters | 2 conv blocks: 0.072 vs 3 blocks: 0.184 - the shallower network cannot build up enough feature detectors |
| Bigger images did not help on a CPU | 128 px: 0.153 - training stopped at epoch 16, before the larger network converged |
| Dropout 0.5 / learning rate 3e-4 | 0.172 / 0.176 - close to, but not better than, the defaults |
| The training fixes worked | A first version (no class weights, max 30 epochs, patience 6) reached only 0.119 test macro F1 |

**What this means:** the learning curves show the CNN is **still underfitting** when it stops - training accuracy
is ~27% and still rising (validation accuracy is higher than training accuracy because augmentation and dropout
only make *training* harder). A small network trained from scratch on ~5.5k images, on a CPU, does not have
enough data or compute to beat hand-crafted HOG + colour features with classical models. The 3-fold CV score is
lower than the hold-out score because each fold trains on only two-thirds of the data. It is competitive on large
classes (fish 0.49, sea anemone 0.45 test F1) and on divers (0.17, better than most models), but scores 0 on six
species. With a GPU, larger images and longer training - or pretrained features - a CNN would be expected to
lead; here it shows the cost of learning features from scratch.

Figures: `results/model_figures/IT25101088_*.png` (variety comparison, confusion matrix, learning curves).

## Viva prep questions

- **Q: Why not 224x224 like the preprocessing pipeline?** No GPU. Each epoch costs ~(size)^2, so 224 px would
  be ~5x slower than 96 px. Image size is tested as a variety (96 vs 128).
- **Q: Why is augmentation inside the model?** Keras augmentation layers are active only during `fit()`;
  during `evaluate()`/`predict()` they pass images through unchanged, so validation and test images are never
  augmented - the same rule as the group pipeline.
- **Q: What does early stopping do?** It stops training when validation loss has not improved for 10 epochs
  and restores the weights from the best epoch, preventing overfitting and saving CPU time.
- **Q: Why class weights in every variety?** A first version of this notebook trained without them and stopped
  early (patience 6, 30 epochs): the learning curves showed training accuracy still rising at ~21% - the network
  was **undertrained**, and without class weights it mostly predicted the large classes (fish, coral), giving
  ~0.44 accuracy but only ~0.09 macro F1. Balanced class weights, longer training and a learning-rate schedule
  fix both problems. (This is a good example of reading learning curves to diagnose underfitting.)
- **Q: What does ReduceLROnPlateau do?** When the training loss stops improving for 3 epochs, it halves the
  learning rate, letting the optimiser take smaller, more careful steps as it approaches a minimum.
- **Q: Why only 3-fold CV?** Each CNN training run takes minutes; 3 folds triple the cost, 5 would be 5x.
  Hold-out validation was used to compare varieties, CV only to confirm the final choice.
- **Q: Why is the CNN not far better than the classical models?** Training from scratch needs a lot of data;
  ~5.5k images (17 for the rarest species) and a small CPU-sized network limit it. Transfer learning from a
  pretrained network would help but is outside the module's scope.
- **Q: What does BatchNormalization do?** It standardizes each layer's activations within a batch, which
  stabilises and speeds up training and has a mild regularising effect.
