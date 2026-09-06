"""
배경 클리닝된 jpg 이미지의 흰 배경을 투명으로 바꿔 png로 저장하는 스크립트

밝기(흰색일수록 큰 값)를 그대로 뒤집어 알파값으로 사용한다.
- 완전한 흰색(255) -> 알파 0 (완전 투명)
- 완전한 검정(0)   -> 알파 255 (완전 불투명)
- 중간 회색(안티에일리어싱 경계) -> 중간 알파값 (부드러운 반투명 경계)
이 방식은 이진(흰색=투명/그외=불투명) 방식과 달리 선 가장자리가
딱딱하게 잘리지 않고 자연스럽게 페이드아웃되는 장점이 있다.

사용법:
    python make_transparent.py --input_dir ./cleaned --output_dir ./transparent

필요 라이브러리:
    pip install opencv-python-headless numpy
"""

import cv2
import numpy as np
import os
import glob
import argparse


def make_output_filename(original_name):
    """qr_clean_000.jpg -> qr_alpha_000.png 형태로 변환."""
    root, ext = os.path.splitext(original_name)
    if "_clean_" in root:
        new_root = root.replace("_clean_", "_alpha_", 1)
    elif "_" in root:
        prefix, rest = root.split("_", 1)
        new_root = f"{prefix}_alpha_{rest}"
    else:
        new_root = f"alpha_{root}"
    return new_root + ".png"


def make_transparent(img_path, out_path, white_floor=255):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[읽기 실패] {img_path}")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 흰색(255)에 가까울수록 투명(알파 0), 검정(0)에 가까울수록 불투명(알파 255).
    # white_floor를 255보다 낮게 주면 그 값 이상은 전부 완전 투명으로 강제할 수 있다.
    # int32로 계산해야 capped*255 (최대 65025)에서 오버플로우가 나지 않는다.
    capped = np.minimum(gray, white_floor).astype(np.int32)
    alpha = (255 - (capped * 255 // white_floor)).astype(np.uint8)

    b, g, r = cv2.split(img)
    out = cv2.merge([b, g, r, alpha])

    cv2.imwrite(out_path, out)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="배경 클리닝된 이미지 폴더")
    ap.add_argument("--output_dir", required=True, help="투명 배경 png 저장 폴더")
    ap.add_argument("--white_floor", type=int, default=255,
                     help="이 밝기 이상은 전부 완전 투명(알파 0) 처리 (기본 255)")
    ap.add_argument("--ext", default="jpg", help="입력 확장자 (jpg, png 등)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, f"*.{args.ext}")))

    print(f"총 {len(files)}개 파일 처리 시작")
    fail_list = []
    for f in files:
        out_name = make_output_filename(os.path.basename(f))
        out_path = os.path.join(args.output_dir, out_name)
        ok = make_transparent(f, out_path, white_floor=args.white_floor)
        if not ok:
            fail_list.append(f)
        else:
            print(f"[OK] {out_name}")

    print(f"\n완료: {len(files)-len(fail_list)}개 성공 / {len(fail_list)}개 실패")
    if fail_list:
        print("실패한 파일:")
        for f in fail_list:
            print(" -", f)


if __name__ == "__main__":
    main()
