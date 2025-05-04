import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ─── Adjust these to your setup ────────────────────────────────────────────────
img_path  = "amd_2037257_1.jpg"
# This is the folder where you ran MedSAM_Inference with -o out_masks
mask_dir  = "/Users/ij/Desktop/PYTHON/kites/MedSAM/out_masks"
# Change the filename if yours has a different suffix
mask_file = "seg_amd_2037257_1.jpg"
# ───────────────────────────────────────────────────────────────────────────────

mask_path = os.path.join(mask_dir, mask_file)

# Load image and mask
img  = np.array(Image.open(img_path).convert("RGB"))
mask = np.array(Image.open(mask_path).convert("L"))

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: input + box
ax1.imshow(img)
# draw the box you used: x0=372, y0=162, x1=383, y1=173
rect = patches.Rectangle((372, 162),
                         383 - 372,
                         173 - 162,
                         linewidth=2,
                         edgecolor='blue',
                         facecolor='none')
ax1.add_patch(rect)
ax1.set_title("Input Image and Bounding Box")
ax1.axis('off')

# Right: overlay mask
ax2.imshow(img)
ax2.imshow(mask > 0, alpha=0.4)  # mask as translucent yellow
ax2.set_title("MedSAM Segmentation")
ax2.axis('off')

plt.tight_layout()
plt.show()
