"""
BiSeNet (face_segment16.pth, Stable-Hair2) + 기존 matte로
3구역 semantic map 생성.

  헤어   → 검정 (0)    ← 기존 matte 우선
  얼굴   → 회색 (128)  ← BiSeNet face labels
  배경   → 흰색 (255)

Usage:
    python scripts/make_semantic_map.py --ids braid_2534 braid_2562 braid_2537
    python scripts/make_semantic_map.py  # 전체 처리
"""
import argparse
import sys, os

STABLE_HAIR2 = os.path.expanduser("~/Stable-Hair2")
sys.path.insert(0, STABLE_HAIR2)

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

from utils.bisenet import BiSeNet

# face_segment16.pth 라벨 (16-class BiSeNet)
# 0:bg 1:skin 2:l_brow 3:r_brow 4:l_eye 5:r_eye 6:eye_g
# 7:l_ear 8:r_ear 9:ear_r 10:hair 11:nose 12:mouth 13:u_lip
# 14:l_lip 15:neck ...
HAIR_LABEL  = 10
FACE_LABELS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14}

CKPT_PATH = os.path.join(STABLE_HAIR2, "models/face_segment16.pth")

TRANSFORM = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def load_model(device):
    print(f"BiSeNet 로드: {CKPT_PATH}")
    model = BiSeNet(n_classes=16).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device), strict=False)
    model.eval()
    print("로드 완료.")
    return model


def predict(img_pil, model, device, size=512):
    inp = TRANSFORM(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp)[0]              # [1, 16, H, W]
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()   # [H, W]
    pred_pil = Image.fromarray(pred.astype(np.uint8)).resize((size, size), Image.NEAREST)
    return np.array(pred_pil)


def make_semantic_map(img_pil, matte_pil, model, device, size=512):
    pred = predict(img_pil, model, device, size)

    hair_mask = np.array(matte_pil.convert("L").resize((size, size))) / 255.0 > 0.5
    face_mask = np.isin(pred, list(FACE_LABELS))

    sem = np.full((size, size), 255, dtype=np.uint8)   # 배경: 흰색
    sem[face_mask] = 128                                # 얼굴: 회색
    sem[hair_mask] = 0                                  # 헤어: 검정 (matte 우선)

    return Image.fromarray(sem, mode="L")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir",   default="dataset3/braid/img/test")
    parser.add_argument("--matte_dir", default="dataset3/braid/matte/test")
    parser.add_argument("--out_dir",   default="dataset/semantic")
    parser.add_argument("--ids",       nargs="+", default=None,
                        help="처리할 ID (예: braid_2534 braid_2562). 미지정 시 전체.")
    parser.add_argument("--size",      type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    img_dir   = Path(args.img_dir)
    matte_dir = Path(args.matte_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [img_dir / f"{i}.png" for i in args.ids] if args.ids \
              else sorted(img_dir.glob("*.png"))

    model = load_model(device)

    for img_path in targets:
        stem       = img_path.stem
        matte_path = matte_dir / f"{stem}.png"
        out_path   = out_dir / f"{stem}.png"

        if not img_path.exists():
            print(f"[skip] 이미지 없음: {img_path}"); continue
        if not matte_path.exists():
            print(f"[skip] matte 없음: {matte_path}"); continue

        img_pil   = Image.open(img_path).convert("RGB")
        matte_pil = Image.open(matte_path).convert("L")

        sem_map = make_semantic_map(img_pil, matte_pil, model, device, args.size)
        sem_map.save(out_path)
        print(f"  저장: {out_path}")

    print("완료.")


if __name__ == "__main__":
    main()
