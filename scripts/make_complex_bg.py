"""
인물 없는 복잡한 배경 이미지 생성기.

실내 인물 사진 배경을 시뮬레이션:
  - 상단 70%: 따뜻한 벽 그라디언트
  - 하단 30%: 어두운 바닥 그라디언트
  - 보케 효과: 랜덤 소프트 원형 오브젝트 (아웃포커스 느낌)
  - 그레인: 미세 노이즈 오버레이
  - 밝은 창문 영역 (우측 상단)

출력: dataset3/backgrounds/indoor_scene.png (512×512)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path

SIZE = 512
SEED = 42
rng  = np.random.default_rng(SEED)


def make_wall_gradient(size: int) -> np.ndarray:
    """상단 70% 벽: 따뜻한 베이지/황토색 수직 그라디언트."""
    top_color    = np.array([210, 190, 165], dtype=np.float32)  # 밝은 베이지
    bottom_color = np.array([160, 138, 112], dtype=np.float32)  # 어두운 황토
    floor_line   = int(size * 0.70)

    canvas = np.zeros((size, size, 3), dtype=np.float32)

    # 벽 (0 ~ floor_line)
    for y in range(floor_line):
        t = y / floor_line
        canvas[y] = (1 - t) * top_color + t * bottom_color

    # 바닥 (floor_line ~ size): 더 어두운 갈색
    floor_top    = np.array([120, 95, 68], dtype=np.float32)
    floor_bottom = np.array([60,  45, 30], dtype=np.float32)
    for y in range(floor_line, size):
        t = (y - floor_line) / (size - floor_line)
        canvas[y] = (1 - t) * floor_top + t * floor_bottom

    return canvas


def add_window(canvas: np.ndarray, size: int) -> np.ndarray:
    """우측 상단에 밝은 창문 영역 추가."""
    result = canvas.copy()
    cx, cy = int(size * 0.80), int(size * 0.20)
    radius = int(size * 0.18)

    yy, xx = np.ogrid[:size, :size]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
    # 소프트 원형 광원
    glow = np.clip(1.0 - dist / radius, 0.0, 1.0) ** 1.5
    window_color = np.array([255, 245, 220], dtype=np.float32)  # 따뜻한 흰빛
    result += glow[..., None] * (window_color - result) * 0.65
    return result


def add_bokeh(canvas: np.ndarray, size: int, n: int = 18) -> np.ndarray:
    """아웃포커스 보케 원 추가 (배경 오브젝트 시뮬레이션)."""
    result = canvas.copy()
    floor_line = int(size * 0.70)

    for _ in range(n):
        cx   = rng.integers(0, size)
        cy   = rng.integers(0, floor_line)
        r    = rng.integers(20, 70)
        hue  = rng.uniform(0, 1)
        # HSV → RGB 느낌: 따뜻한 색조 위주
        base = np.array([
            180 + rng.uniform(-30, 60),
            160 + rng.uniform(-40, 50),
            120 + rng.uniform(-40, 40),
        ], dtype=np.float32)
        alpha = rng.uniform(0.08, 0.22)

        yy, xx = np.ogrid[:size, :size]
        dist   = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
        mask   = np.clip(1.0 - dist / r, 0.0, 1.0) ** 2
        result += mask[..., None] * (base - result) * alpha

    return result


def add_grain(canvas: np.ndarray, strength: float = 8.0) -> np.ndarray:
    """미세 노이즈 그레인."""
    noise  = rng.normal(0, strength, canvas.shape).astype(np.float32)
    return canvas + noise


def main():
    out_path = Path("dataset3/backgrounds/indoor_scene.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("배경 이미지 생성 중...")

    canvas = make_wall_gradient(SIZE)
    canvas = add_window(canvas, SIZE)
    canvas = add_bokeh(canvas, SIZE)
    canvas = add_grain(canvas)
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    img = Image.fromarray(canvas)
    # 최종 소프트 블러로 자연스럽게
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    img.save(out_path)

    print(f"저장 완료: {out_path}  ({img.size[0]}×{img.size[1]})")


if __name__ == "__main__":
    main()
