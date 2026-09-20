"""
transparent 폴더의 알파(PNG) 이미지 위에 src/edit_qr.png(위치 지정 패턴 이미지)를
알파 채널까지 고려해 합성하는 스크립트 (Porter-Duff 'over' 합성).

일반 오버레이(overlay_qr.py)와 달리 배경(base)도 알파를 갖고 있으므로,
두 레이어 모두 투명한 영역은 결과도 투명하게 유지된다.

사용법:
    python composite_transparent.py --base_dir ./transparent --overlay ./src/edit_qr.png --output_dir ./transparent_comp --size 1000

필요 라이브러리:
    pip install opencv-python-headless numpy
"""

import cv2
import numpy as np
import os
import glob
import argparse


def load_rgba(path, size):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"이미지를 읽지 못했습니다: {path}")
    if im.shape[0] != size or im.shape[1] != size:
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)

    if im.ndim == 2:  # 그레이스케일 -> 완전 불투명 BGRA
        bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        alpha = np.full((size, size), 255, np.uint8)
    elif im.shape[2] == 3:  # 알파 없음 -> 완전 불투명
        bgr = im
        alpha = np.full((size, size), 255, np.uint8)
    else:
        bgr = im[:, :, :3]
        alpha = im[:, :, 3]

    bgr = bgr.astype(np.float32)
    a = (alpha.astype(np.float32) / 255.0)[:, :, None]
    return bgr, a


def alpha_over(fg_bgr, fg_a, bg_bgr, bg_a):
    """Porter-Duff 'over': fg를 bg 위에 얹는다."""
    out_a = fg_a + bg_a * (1.0 - fg_a)
    safe_a = np.where(out_a <= 1e-6, 1.0, out_a)
    out_rgb = (fg_bgr * fg_a + bg_bgr * bg_a * (1.0 - fg_a)) / safe_a
    out_rgb = np.where(out_a <= 1e-6, 0.0, out_rgb)
    return out_rgb, out_a


def composite(base_path, out_path, ov_bgr, ov_a, size):
    base_bgr, base_a = load_rgba(base_path, size)

    out_rgb, out_a = alpha_over(ov_bgr, ov_a, base_bgr, base_a)

    out_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8)
    out_a8 = np.clip(out_a * 255.0, 0, 255).astype(np.uint8)
    b, g, r = cv2.split(out_rgb)
    out = cv2.merge([b, g, r, out_a8[:, :, 0]])

    cv2.imwrite(out_path, out)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="./transparent", help="배경이 될 투명 PNG 폴더")
    ap.add_argument("--overlay", default="./src/edit_qr.png", help="위에 올릴 위치 지정 패턴 이미지")
    ap.add_argument("--output_dir", default="./transparent_comp", help="합성 결과 저장 폴더")
    ap.add_argument("--size", type=int, default=1000, help="결과 이미지 한 변 길이(px)")
    ap.add_argument("--ext", default="png", help="입력 확장자")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ov_bgr, ov_a = load_rgba(args.overlay, args.size)

    files = sorted(glob.glob(os.path.join(args.base_dir, f"*.{args.ext}")))
    print(f"총 {len(files)}개 파일 합성 시작")

    fail = []
    for f in files:
        name = os.path.basename(f)
        out_name = name.replace("_alpha_", "_comp_", 1) if "_alpha_" in name else f"comp_{name}"
        out_path = os.path.join(args.output_dir, out_name)
        if composite(f, out_path, ov_bgr, ov_a, args.size):
            print(f"[OK] {out_name}")
        else:
            fail.append(f)

    print(f"\n완료: {len(files) - len(fail)}개 성공 / {len(fail)}개 실패")
    for f in fail:
        print(" -", f)


if __name__ == "__main__":
    main()
