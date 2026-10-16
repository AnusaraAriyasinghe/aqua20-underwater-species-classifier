# Module: `notebooks/preprocessing/IT25101088_Color_Correction.ipynb`

## Role in the pipeline
Corrects underwater-specific color cast and low contrast using CLAHE before the image is
normalized and fed into a CNN — the technique this project adds beyond what a "clean" dataset
(like the sample group's Cats-vs-Dogs project) would need.

## Flow
```
dataset['train'][0]['image'] (PIL image)
        |
        v
 to_numpy(..., size=(224,224))  -> RGB array
        |
        v
 apply_clahe(image_array)
   RGB -> LAB color space
   split into L (lightness), a, b (color) channels
   CLAHE applied to L only (local, contrast-limited histogram equalization)
   merge channels back -> LAB -> RGB
        |
        v
 Before/after image comparison (EDA plot #1)
 Before/after pixel-intensity histogram (EDA plot #2)
```

## Line-by-line walkthrough

### Cell 3 — Import OpenCV
```python
import cv2
```
OpenCV supplies both the LAB color-space conversion and the CLAHE implementation — reimplementing
adaptive histogram equalization by hand would be unnecessary and error-prone when a
well-tested library function exists.

### Cell 5 — The CLAHE function
```python
def apply_clahe(image_array):
    lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
```
- `cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)` converts the image from RGB into the LAB color
  space. LAB separates **L** (lightness/brightness, roughly what a grayscale version would look
  like) from **a** (green-red) and **b** (blue-yellow) — the two color-only channels.
- `cv2.split(lab)` splits the 3-channel LAB array into three separate 2D arrays: `l`, `a`, `b`.
- `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))` configures the CLAHE algorithm:
  - `tileGridSize=(8, 8)` divides the image into an 8x8 grid of local tiles and equalizes each
    tile's histogram independently — this is the "adaptive"/local part, as opposed to global
    histogram equalization which treats the whole image as one region and would over-brighten
    areas that are already well-lit while under-correcting genuinely dark/hazy regions.
  - `clipLimit=2.0` caps how much any single intensity bin can be amplified within a tile,
    preventing the algorithm from over-amplifying noise in near-uniform (very dark or very
    bright) tiles — a known failure mode of plain adaptive histogram equalization without a
    clip limit.
- `clahe.apply(l)` runs the algorithm on the lightness channel only.
- `cv2.merge((l_eq, a, b))` recombines the equalized lightness channel with the **original,
  untouched** color channels — this is deliberate: we want to fix contrast, not shift hue/color,
  since color patterns can be part of what distinguishes marine species.
- `cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)` converts back to RGB for display/further processing.

### Cell 6 — Before/after image plot
Straightforward `imshow` side-by-side of `example` (original) and `corrected` (post-CLAHE),
saved to `color_correction_before_after.png`.

### Cell 7 — Pixel intensity histograms
```python
plt.hist(example.ravel(), bins=50)
...
plt.hist(corrected.ravel(), bins=50)
```
- `.ravel()` flattens the 3D `(224, 224, 3)` array into a single 1D list of all pixel values
  across all three RGB channels, so the histogram represents the overall intensity spread of
  the entire image.
- Comparing the two histograms is the quantitative evidence for whether CLAHE actually improved
  contrast: a narrow, bunched original histogram vs. a wider, more spread-out corrected one
  indicates successful contrast recovery.

## Why this technique / method
- **Why CLAHE and not plain (global) histogram equalization?** Underwater images have
  non-uniform lighting — parts of the frame may be well-lit while others are in murky shadow.
  Global equalization treats the whole image as one histogram, which tends to over-brighten
  already-bright regions while barely helping genuinely dark ones. CLAHE's per-tile, clip-limited
  approach targets local contrast without this side effect.
- **Why work in LAB space instead of applying CLAHE to each RGB channel directly?** Applying
  histogram equalization independently to R, G, and B channels can shift the image's overall
  color balance in unpredictable ways (each channel gets stretched differently). Isolating
  brightness (L) from color (a, b) lets us fix contrast while leaving color information intact —
  important since color patterns can be a real classification signal for species.
- **Why this technique specifically for AQUA20 vs. a "clean" dataset?** The AQUA20 dataset card
  explicitly documents "reduced visibility, inadequate lighting, and partial occlusion" as core
  challenges — a straightforward RGB decode (sufficient for a clean dataset like Cats-vs-Dogs)
  does not address this domain-specific problem.

## Viva prep questions
1. **Q: Why LAB color space instead of RGB or HSV?**
   A: LAB cleanly separates lightness from color in a way that's designed to match human
   perceptual difference; equalizing only the L channel avoids distorting hue/saturation
   information the way independently equalizing R, G, B channels would. (HSV's V channel is
   an alternative that also isolates brightness, but LAB's L channel is the more standard choice
   for CLAHE-based contrast correction in computer vision literature.)
2. **Q: What does `clipLimit` control, and what happens if you set it very high?**
   A: It caps the maximum contrast amplification allowed in any local tile's histogram, to
   prevent over-amplifying noise in near-flat regions. A very high value approaches unlimited
   adaptive histogram equalization, which can introduce visible noise/artifacts especially in
   dark, low-detail underwater regions.
3. **Q: What does `tileGridSize=(8, 8)` mean, and how would a smaller/larger grid change the
   result?**
   A: The image is split into an 8x8 grid (64 tiles), each equalized independently. A smaller
   grid (e.g. 2x2) behaves closer to global equalization; a larger grid (e.g. 16x16) makes
   corrections more local but risks blocky artifacts at tile boundaries if the clip limit is
   too permissive.
4. **Q: How do you know CLAHE actually helped, beyond visually looking at the image?**
   A: The pixel-intensity histogram comparison, backed by a numeric measure: for the sample image,
   the overall pixel standard deviation increases from about 55.9 (original) to about 67.8 (after
   CLAHE), while the mean barely shifts (~118.1 -> ~122.4). A higher standard deviation with a
   near-constant mean is exactly what "more contrast, same overall exposure" looks like
   numerically — confirming the visual spread seen in the histogram plot.
5. **Q: Why is color correction applied to *all* splits (train/val/test) in the group pipeline,
   unlike augmentation which is training-only?**
   A: Color correction fixes a genuine defect in how the image was captured (poor
   lighting/visibility) — it should be applied consistently everywhere the model will see data,
   including at inference/test time. Augmentation, by contrast, artificially creates variety
   only to help the model generalize during training and would make validation/test evaluation
   inconsistent with real deployment conditions if applied there too.
