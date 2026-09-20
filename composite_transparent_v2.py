"""
transparent 폴더의 알파(PNG) 이미지 위에 src/edit_qr.png(위치 지정 패턴 이미지)를
합성하되, 두 레이어의 바깥 여백을 같은 비율로 잘라내 "함께" 확대해서 얹는 v2 스크립트.

composite_transparent.py 와의 차이:
- overlay(edit_qr.png)의 파인더 패턴은 base(transparent/*.png)의 QR 자체 파인더
  패턴과 같은 위치에 정렬되도록 설계되어 있으므로(둘 다 전체 캔버스 기준 약 7%
  여백), 정렬을 유지하려면 두 레이어를 반드시 "동일한 비율"로 잘라내야 한다.
- crop_frac: 각 이미지의 전체 캔버스 기준으로 가장자리에서 잘라낼 비율(0~0.5 미만).
  base는 1000x1000 캔버스 기준으로, overlay는 원본(4167x4167) 캔버스 기준으로
  같은 비율만큼 잘라낸 뒤 각각 1000x1000으로 리사이즈 -> 두 레이어가 정렬을
  유지한 채 함께 확대(zoom)된다.
- 합성 방식은 동일하게 Porter-Duff 'over' (알파 반영).

사용법:
    python composite_transparent_v2.py --base_dir ./transparent --overlay ./src/edit_qr.png ^
        --output_dir ./transparent_comp_v2 --size 1000 --crop_frac 0.053

필요 라이브러리:
    pip install opencv-python-headless numpy
"""

import cv2
import numpy as np
import os
import glob
import argparse


def load_rgba_raw(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"이미지를 읽지 못했습니다: {path}")
    if im.ndim == 2:
        bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        alpha = np.full(im.shape[:2], 255, np.uint8)
    elif im.shape[2] == 3:
        bgr = im
        alpha = np.full(im.shape[:2], 255, np.uint8)
    else:
        bgr = im[:, :, :3]
        alpha = im[:, :, 3]
    return bgr, alpha


def crop_border_frac(bgr, alpha, crop_frac):
    """캔버스 가장자리에서 crop_frac 비율만큼 균일하게 잘라낸다(컨텐츠 위치와 무관)."""
    h, w = alpha.shape
    cy = int(round(h * crop_frac))
    cx = int(round(w * crop_frac))
    if cy <= 0 and cx <= 0:
        return bgr, alpha
    return bgr[cy:h - cy, cx:w - cx], alpha[cy:h - cy, cx:w - cx]


def prep_overlay(path, size, crop_frac):
    bgr, alpha = load_rgba_raw(path)
    bgr, alpha = crop_border_frac(bgr, alpha, crop_frac)

    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (size, size), interpolation=cv2.INTER_AREA)

    bgr = bgr.astype(np.float32)
    a = (alpha.astype(np.float32) / 255.0)[:, :, None]
    return bgr, a


def load_base_rgba(path, size, crop_frac):
    bgr, alpha = load_rgba_raw(path)
    bgr, alpha = crop_border_frac(bgr, alpha, crop_frac)
    if bgr.shape[0] != size or bgr.shape[1] != size:
        bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, (size, size), interpolation=cv2.INTER_AREA)
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


def composite(base_path, out_path, ov_bgr, ov_a, size, crop_frac):
    base_bgr, base_a = load_base_rgba(base_path, size, crop_frac)

    out_rgb, out_a = alpha_over(ov_bgr, ov_a, base_bgr, base_a)

    out_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8)
    out_a8 = np.clip(out_a * 255.0, 0, 255).astype(np.uint8)
    b, g, r = cv2.split(out_rgb)
    out = cv2.merge([b, g, r, out_a8[:, :, 0]])

    cv2.imwrite(out_path, out)
    return True


def make_output_filename(original_name):
    """qr_alpha_000.png -> qr_compV2_000.png 형태로 변환."""
    root, ext = os.path.splitext(original_name)
    if "_alpha_" in root:
        new_root = root.replace("_alpha_", "_compV2_", 1)
    elif "_" in root:
        prefix, rest = root.split("_", 1)
        new_root = f"{prefix}_compV2_{rest}"
    else:
        new_root = f"compV2_{root}"
    return new_root + ".png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="./transparent", help="배경이 될 투명 PNG 폴더")
    ap.add_argument("--overlay", default="./src/edit_qr.png", help="위에 올릴 위치 지정 패턴 이미지")
    ap.add_argument("--output_dir", default="./transparent_comp_v2", help="합성 결과 저장 폴더")
    ap.add_argument("--size", type=int, default=1000, help="결과 이미지 한 변 길이(px)")
    ap.add_argument("--crop_frac", type=float, default=0.053,
                     help="두 레이어(overlay, base) 각각의 전체 캔버스 기준 가장자리 크롭 비율. "
                          "클수록 정렬을 유지한 채 QR+파인더 패턴이 함께 확대됨")
    ap.add_argument("--ext", default="png", help="입력 확장자")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ov_bgr, ov_a = prep_overlay(args.overlay, args.size, args.crop_frac)

    files = sorted(glob.glob(os.path.join(args.base_dir, f"*.{args.ext}")))
    print(f"총 {len(files)}개 파일 합성 시작")

    fail = []
    for f in files:
        name = os.path.basename(f)
        out_name = make_output_filename(name)
        out_path = os.path.join(args.output_dir, out_name)
        if composite(f, out_path, ov_bgr, ov_a, args.size, args.crop_frac):
            print(f"[OK] {out_name}")
        else:
            fail.append(f)

    print(f"\n완료: {len(files) - len(fail)}개 성공 / {len(fail)}개 실패")
    for f in fail:
        print(" -", f)


if __name__ == "__main__":
    main()
