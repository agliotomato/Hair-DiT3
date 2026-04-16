"""
Sketch 형식 변환 유틸리티

colored sketch (RGB) → grayscale B/W (SketchHairSalon 입력용)

사용법:
  # 단일 파일
  python scripts/convert_sketch.py --input sketch.png --output sketch_bw.png

  # 디렉토리 일괄 변환
  python scripts/convert_sketch.py \\
    --input_dir  dataset3/braid/sketch/test \\
    --output_dir results/sketch_bw/braid/test
"""
import argparse
from pathlib import Path
from PIL import Image


def colored_to_bw(img: Image.Image, threshold: int = 128) -> Image.Image:
    """
    RGB colored sketch → binary B/W sketch.
    검정(0) 배경에 흰색(255) stroke 형태로 변환.
    """
    gray = img.convert("L")
    bw = gray.point(lambda x: 255 if x > threshold else 0, "L")
    return bw


def parse_args():
    parser = argparse.ArgumentParser(description="Colored sketch → B/W 변환")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input",     type=str, help="단일 입력 파일")
    group.add_argument("--input_dir", type=str, help="입력 디렉토리")
    parser.add_argument("--output",     type=str, default=None, help="단일 출력 파일")
    parser.add_argument("--output_dir", type=str, default=None, help="출력 디렉토리")
    parser.add_argument("--threshold", type=int, default=128, help="이진화 임계값 (기본 128)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.input:
        out = args.output or Path(args.input).with_stem(Path(args.input).stem + "_bw")
        img = Image.open(args.input).convert("RGB")
        bw  = colored_to_bw(img, args.threshold)
        bw.save(str(out))
        print(f"저장: {out}")
    else:
        in_dir  = Path(args.input_dir)
        out_dir = Path(args.output_dir) if args.output_dir else in_dir.parent / (in_dir.name + "_bw")
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(in_dir.glob("*.png")) + sorted(in_dir.glob("*.jpg"))
        for f in files:
            img = Image.open(f).convert("RGB")
            bw  = colored_to_bw(img, args.threshold)
            bw.save(str(out_dir / (f.stem + ".png")))
        print(f"{len(files)}장 변환 완료 → {out_dir}")


if __name__ == "__main__":
    main()
