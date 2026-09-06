"""
cleaned 폴더의 이미지 위에 src/edit_qr.png 를 합성하는 스크립트

- 모든 결과물은 1000x1000px 로 통일한다.
- edit_qr.png 에 알파 채널이 있으면 알파 기준으로 자연스럽게 올린다.

사용법:
    python overlay_qr.py --base_dir ./cleaned --overlay ./src/edit_qr.png --output_dir ./composited --size 1000

필요 라이브러리:
    pip install opencv-python-headless numpy
"""

import cv2
import numpy as np
import os
import glob
import argparse


def load_overlay(path, size):
    ov = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if ov is None:
        raise FileNotFoundError(f"오버레이 이미지를 읽지 못했습니다: {path}")
    ov = cv2.resize(ov, (size, size), interpolation=cv2.INTER_AREA)

    if ov.ndim == 2:  # 그레이스케일
        ov = cv2.cvtColor(ov, cv2.COLOR_GRAY2BGR)

    if ov.shape[2] == 4:
        bgr = ov[:, :, :3].astype(np.float32)
        alpha = (ov[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
    else:
        bgr = ov.astype(np.float32)
        alpha = np.ones((size, size, 1), np.float32)
    return bgr, alpha


def composite(base_path, out_path, ov_bgr, ov_alpha, size):
    base = cv2.imread(base_path)
    if base is None:
        print(f"[읽기 실패] {base_path}")
        return False
    if base.shape[0] != size or base.shape[1] != size:
        base = cv2.resize(base, (size, size), interpolation=cv2.INTER_AREA)

    out = base.astype(np.float32) * (1.0 - ov_alpha) + ov_bgr * ov_alpha
    out = np.clip(out, 0, 255).astype(np.uint8)

    ext = os.path.splitext(out_path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(out_path, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        cv2.imwrite(out_path, out)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="./cleaned", help="배경이 될 이미지 폴더")
    ap.add_argument("--overlay", default="./src/edit_qr.png", help="위에 올릴 이미지")
    ap.add_argument("--output_dir", default="./composited", help="합성 결과 저장 폴더")
    ap.add_argument("--size", type=int, default=1000, help="결과 이미지 한 변 길이(px)")
    ap.add_argument("--ext", default="jpg", help="입력 확장자")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ov_bgr, ov_alpha = load_overlay(args.overlay, args.size)

    files = sorted(glob.glob(os.path.join(args.base_dir, f"*.{args.ext}")))
    print(f"총 {len(files)}개 파일 합성 시작")

    fail = []
    for f in files:
        name = os.path.basename(f)
        out_name = name.replace("_clean_", "_final_", 1) if "_clean_" in name else f"final_{name}"
        out_path = os.path.join(args.output_dir, out_name)
        if composite(f, out_path, ov_bgr, ov_alpha, args.size):
            print(f"[OK] {out_name}")
        else:
            fail.append(f)

    print(f"\n완료: {len(files) - len(fail)}개 성공 / {len(fail)}개 실패")
    for f in fail:
        print(" -", f)


if __name__ == "__main__":
    main()
