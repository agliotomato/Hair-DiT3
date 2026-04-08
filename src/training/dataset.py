"""HairS2IDataset — (background, sketch, matte, target) 4-tuple 반환"""
import os
from pathlib import Path
from typing import Tuple, Optional, List

import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np
from PIL import Image

try:
    import kornia.filters as KF
    import kornia.morphology as KM
    HAS_KORNIA = True
except ImportError:
    HAS_KORNIA = False

# ThicknessJitter와 MatteBoundaryPerturbation은 이제 .augmentation 모듈로 통합되어 관리됨.


from .augmentation import build_augmentation_pipeline, soft_composite

class HairS2IDataset(Dataset):
    """
    헤어 Sketch-to-Image 학습 데이터셋.

    디렉토리 구조:
        data_dir/
          background/  ← 원본 프레임 (얼굴 + 기존 헤어) RGB
          sketch/      ← 목표 헤어 스케치 (컬러 RGB 지원) RGB
          matte/       ← 헤어 매트 (헤어=255, 배경=0) L
          target/      ← 목표 완성 이미지 (배경 + 목표 헤어) RGB

    모든 하위 디렉토리 내 파일은 같은 순서(stem)로 대응되어야 함.

    반환:
        background: [3, H, W] float32 in [-1, 1]
        sketch:     [3, H, W] float32 in [-1, 1] (컬러 스케치)
        matte:      [1, H, W] float32 in [0, 1]
        target:     [3, H, W] float32 in [-1, 1]
    """

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(
        self,
        data_dir: str,
        image_size: int = 512,
        augment: bool = True,
    ):
        self.data_dir   = Path(data_dir)
        self.image_size = image_size
        self.augment    = augment

        # 파일 목록 수집 (background 기준)
        bg_dir = self.data_dir / "background"
        self.stems: List[str] = sorted([
            p.stem for p in bg_dir.iterdir()
            if p.suffix.lower() in self.VALID_EXTENSIONS
        ])

        if len(self.stems) == 0:
            raise ValueError(f"데이터가 없음: {bg_dir}")

        if self.augment:
            self.aug_pipeline = build_augmentation_pipeline(phase="train")
        else:
            self.aug_pipeline = None

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]

        # 파일 로드
        background = self._load_image(self.data_dir / "background", stem, "RGB")
        sketch     = self._load_image(self.data_dir / "sketch",     stem, "RGB")  # RGB로 변경
        matte      = self._load_image(self.data_dir / "matte",      stem, "L")
        target     = self._load_image(self.data_dir / "target",     stem, "RGB")

        # 크기 통일
        size = (self.image_size, self.image_size)
        background = background.resize(size, Image.LANCZOS)
        sketch     = sketch.resize(size, Image.LANCZOS)
        matte      = matte.resize(size, Image.LANCZOS)
        target     = target.resize(size, Image.LANCZOS)

        # 수평 flip augmentation (sketch-matte 일관성 유지)
        if self.augment and torch.rand(1).item() > 0.5:
            background = TF.hflip(background)
            sketch     = TF.hflip(sketch)
            matte      = TF.hflip(matte)
            target     = TF.hflip(target)

        # Tensor 변환
        bg_tensor  = self._to_tensor_rgb(background)   # [3, H, W] in [-1, 1]
        sk_tensor  = self._to_tensor_rgb(sketch)       # [3, H, W] in [-1, 1] (RGB & 정규화)
        mt_tensor  = self._to_tensor_l(matte)           # [1, H, W] in [0, 1]
        tgt_tensor = self._to_tensor_rgb(target)        # [3, H, W] in [-1, 1]

        # 증강 파이프라인 적용
        if self.aug_pipeline is not None:
            sample = {
                "background": bg_tensor,
                "sketch":     sk_tensor,
                "matte":      mt_tensor,
                "target":     tgt_tensor,
            }
            sample = self.aug_pipeline(sample)
            bg_tensor  = sample["background"]
            sk_tensor  = sample["sketch"]
            mt_tensor  = sample["matte"]
            tgt_tensor = sample["target"]

        return {
            "background": bg_tensor,
            "sketch":     sk_tensor,
            "matte":      mt_tensor,
            "target":     tgt_tensor,
        }

    def _load_image(self, directory: Path, stem: str, mode: str) -> Image.Image:
        for ext in self.VALID_EXTENSIONS:
            path = directory / f"{stem}{ext}"
            if path.exists():
                return Image.open(path).convert(mode)
        raise FileNotFoundError(f"{directory}/{stem}.* 파일 없음")

    @staticmethod
    def _to_tensor_rgb(img: Image.Image) -> torch.Tensor:
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    @staticmethod
    def _to_tensor_l(img: Image.Image) -> torch.Tensor:
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)


class HairRegionDataset(Dataset):
    """
    hair-dit 방식 데이터셋 — dataset3/ 디렉토리 구조 지원.

    디렉토리 구조:
        dataset_root/{style}/img/{subset}/*.png
        dataset_root/{style}/sketch/{subset}/*.png
        dataset_root/{style}/matte/{subset}/*.png

    split 예시: "unbraid_train", "unbraid_test", "braid_train", "braid_test"

    target_split=None (권장): same-domain 재구성 학습
      Phase 1: split="unbraid_train" (3000장 재구성)
      Phase 2: split="braid_train"   (1000장 fine-tuning, Phase 1 체크포인트 로드)

    반환 (EHC 모델 입력 형식):
        background: [3, H, W] float32 in [-1, 1]  — input img * (1 - matte)
        sketch:     [3, H, W] float32 in [-1, 1]  — 컬러 스케치 (VAE 인코딩 대상)
        matte:      [1, H, W] float32 in [0, 1]   — 소프트 헤어 마스크
        target:     [3, H, W] float32 in [-1, 1]  — target img * target matte
        prompt:     str ("")
    """

    VALID_SPLITS = ("unbraid_train", "unbraid_test", "braid_train", "braid_test")

    def __init__(
        self,
        split: str,
        dataset_root: str,
        image_size: int = 512,
        augment: bool = True,
        target_split: Optional[str] = None,
    ):
        if split not in self.VALID_SPLITS:
            raise ValueError(f"split은 {self.VALID_SPLITS} 중 하나여야 함, 받은 값: '{split}'")

        style, subset = split.rsplit("_", 1)
        root = Path(dataset_root) / style

        self.img_dir    = root / "img"    / subset
        self.sketch_dir = root / "sketch" / subset
        self.matte_dir  = root / "matte"  / subset

        for d in [self.img_dir, self.sketch_dir, self.matte_dir]:
            if not d.exists():
                raise FileNotFoundError(f"디렉토리 없음: {d}")

        # cross-domain: target은 별도 스타일에서 로드
        if target_split is not None:
            if target_split not in self.VALID_SPLITS:
                raise ValueError(f"target_split은 {self.VALID_SPLITS} 중 하나여야 함, 받은 값: '{target_split}'")
            tgt_style, tgt_subset = target_split.rsplit("_", 1)
            tgt_root = Path(dataset_root) / tgt_style
            self.tgt_img_dir   = tgt_root / "img"   / tgt_subset
            self.tgt_matte_dir = tgt_root / "matte" / tgt_subset
            for d in [self.tgt_img_dir, self.tgt_matte_dir]:
                if not d.exists():
                    raise FileNotFoundError(f"디렉토리 없음 (target_split): {d}")
        else:
            self.tgt_img_dir   = self.img_dir
            self.tgt_matte_dir = self.matte_dir

        self.stems = sorted(p.stem for p in self.img_dir.glob("*.png"))
        if not self.stems:
            raise ValueError(f"PNG 파일 없음: {self.img_dir}")

        self.image_size = image_size
        self.aug_pipeline = build_augmentation_pipeline() if augment else None

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]
        size = (self.image_size, self.image_size)

        # 입력 도메인 (sketch/matte/background)
        img    = Image.open(self.img_dir    / f"{stem}.png").convert("RGB").resize(size, Image.LANCZOS)
        sketch = Image.open(self.sketch_dir / f"{stem}.png").convert("RGB").resize(size, Image.LANCZOS)
        matte  = Image.open(self.matte_dir  / f"{stem}.png").convert("L").resize(size, Image.LANCZOS)

        # 타겟 도메인 (cross-domain이면 별도 폴더)
        tgt_img   = Image.open(self.tgt_img_dir   / f"{stem}.png").convert("RGB").resize(size, Image.LANCZOS)
        tgt_matte = Image.open(self.tgt_matte_dir / f"{stem}.png").convert("L").resize(size, Image.LANCZOS)

        img_t      = self._to_tensor_rgb(img)       # [3, H, W] in [-1, 1]
        sketch_t   = self._to_tensor_rgb(sketch)    # [3, H, W] in [-1, 1]
        matte_t    = self._to_tensor_l(matte)       # [1, H, W] in [0, 1]
        tgt_img_t  = self._to_tensor_rgb(tgt_img)   # [3, H, W] in [-1, 1]
        tgt_matte_t = self._to_tensor_l(tgt_matte)  # [1, H, W] in [0, 1]

        img_01     = (img_t + 1.0) / 2.0
        tgt_img_01 = (tgt_img_t + 1.0) / 2.0

        background_t = soft_composite(img_01, 1.0 - matte_t) * 2.0 - 1.0   # input img * (1-matte)
        target_t     = soft_composite(tgt_img_01, tgt_matte_t) * 2.0 - 1.0  # target img * target matte

        sample = {
            "background": background_t,
            "sketch":     sketch_t,
            "matte":      matte_t,
            "target":     target_t,
        }

        if self.aug_pipeline is not None:
            sample = self.aug_pipeline(sample)

        # Remove internal prompt key if exists
        sample.pop("prompt", None)

        return sample

    @staticmethod
    def _to_tensor_rgb(img: Image.Image) -> torch.Tensor:
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    @staticmethod
    def _to_tensor_l(img: Image.Image) -> torch.Tensor:
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
