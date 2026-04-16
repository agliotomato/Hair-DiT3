"""
nanobanana 생성 이미지와 오리지널 matte의 헤어 영역 차이 확인.
braid_2562 (smile, sad) vs 오리지널 matte.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATASET_ROOT = "dataset3/braid"
NANO_ROOT = "dataset/nanobanana"
OUT_DIR = "results/matte_diff"

import os
os.makedirs(OUT_DIR, exist_ok=True)

def load_gray(path):
    return np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0

def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))

def overlay_matte(img_rgb, matte, color=(255, 0, 0), alpha=0.5):
    """matte 영역을 color로 반투명 오버레이."""
    overlay = img_rgb.copy().astype(np.float32)
    mask = matte > 0.5
    for c, v in enumerate(color):
        overlay[..., c] = np.where(mask, overlay[..., c] * (1 - alpha) + v * alpha, overlay[..., c])
    return overlay.astype(np.uint8)

# --- 파일 로드 ---
orig = load_rgb(f"{DATASET_ROOT}/img/test/braid_2562.png")
matte = load_gray(f"{DATASET_ROOT}/matte/test/braid_2562.png")
smile = load_rgb(f"{NANO_ROOT}/braid_2562_smile.png")
sad   = load_rgb(f"{NANO_ROOT}/braid_2562_sad.png")

# 이미지 크기 맞추기 (nanobanana 이미지가 다를 수 있음)
H, W = orig.shape[:2]
def resize(img): return np.array(Image.fromarray(img).resize((W, H), Image.LANCZOS))
smile = resize(smile)
sad   = resize(sad)

# --- 헤어 영역 pixel 비교 ---
hair_mask = matte > 0.5
orig_hair  = orig[hair_mask]
smile_hair = smile[hair_mask]
sad_hair   = sad[hair_mask]

diff_smile = np.abs(orig_hair.astype(float) - smile_hair.astype(float))
diff_sad   = np.abs(orig_hair.astype(float) - sad_hair.astype(float))

print("=== 오리지널 matte 영역 내 pixel 차이 ===")
print(f"  smile - orig : mean={diff_smile.mean():.2f}, max={diff_smile.max():.0f}")
print(f"  sad   - orig : mean={diff_sad.mean():.2f},   max={diff_sad.max():.0f}")
print(f"  matte 커버 비율: {hair_mask.mean()*100:.1f}% of image")

# --- 시각화 ---
fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 4, figure=fig)

titles = ["Original", "Smile", "Sad"]
images = [orig, smile, sad]

# Row 0: 원본 이미지 + matte 오버레이
for i, (img, title) in enumerate(zip(images, titles)):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(overlay_matte(img, matte))
    ax.set_title(f"{title}\n(red = original matte)")
    ax.axis("off")

# Row 0, col 3: matte 자체
ax = fig.add_subplot(gs[0, 3])
ax.imshow(matte, cmap="gray")
ax.set_title("Original Matte")
ax.axis("off")

# Row 1: 헤어 영역만 crop diff map
def make_diff_map(img_a, img_b, mask):
    diff = np.zeros((*mask.shape, 3), dtype=np.uint8)
    d = np.abs(img_a.astype(float) - img_b.astype(float)).mean(axis=-1)
    d_norm = np.clip(d / 100.0, 0, 1)
    diff[mask] = plt.cm.hot(d_norm[mask])[:, :3] * 255
    return diff

diff_map_smile = make_diff_map(orig, smile, hair_mask)
diff_map_sad   = make_diff_map(orig, sad,   hair_mask)

ax = fig.add_subplot(gs[1, 0])
ax.imshow(diff_map_smile)
ax.set_title(f"Diff: Smile vs Orig\n(matte 영역, mean={diff_smile.mean():.1f})")
ax.axis("off")

ax = fig.add_subplot(gs[1, 1])
ax.imshow(diff_map_sad)
ax.set_title(f"Diff: Sad vs Orig\n(matte 영역, mean={diff_sad.mean():.1f})")
ax.axis("off")

# Row 1, col 2-3: 헤어 영역 crop
def crop_hair(img, mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    pad = 20
    r0, r1 = max(0, r0-pad), min(img.shape[0], r1+pad)
    c0, c1 = max(0, c0-pad), min(img.shape[1], c1+pad)
    return img[r0:r1, c0:c1], mask[r0:r1, c0:c1]

orig_crop,  mask_crop  = crop_hair(orig,  hair_mask)
smile_crop, _          = crop_hair(smile, hair_mask)
sad_crop,   _          = crop_hair(sad,   hair_mask)

ax = fig.add_subplot(gs[1, 2])
ax.imshow(np.concatenate([orig_crop, smile_crop], axis=1))
ax.set_title("Hair crop: Orig | Smile")
ax.axis("off")

ax = fig.add_subplot(gs[1, 3])
ax.imshow(np.concatenate([orig_crop, sad_crop], axis=1))
ax.set_title("Hair crop: Orig | Sad")
ax.axis("off")

plt.suptitle("braid_2562: nanobanana vs original matte region", fontsize=13)
plt.tight_layout()
out_path = f"{OUT_DIR}/braid_2562_matte_diff.png"
plt.savefig(out_path, dpi=150)
print(f"\n저장: {out_path}")
plt.show()
