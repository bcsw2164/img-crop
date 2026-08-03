"""
크롭된 이미지의 배경(연한 회색/흰색 얼룩)을 완전한 순백색으로 정리하는 스크립트

배경 밝기가 이미지마다/영역마다 균일하지 않을 수 있으므로,
단순 고정 임계값 대신 "국소 배경 추정 후 정규화 -> 임계값 적용" 방식을 사용한다.

사용법:
    python clean_background.py --input_dir ./cropped --output_dir ./cleaned --threshold 230

필요 라이브러리:
    pip install opencv-python-headless numpy
"""

import cv2
import numpy as np
import os
import glob
import argparse


def estimate_background(gray, dilate_kernel=25, blur_kernel=21):
    """얇은 어두운 선(연필 스트로크)을 지우고 국소 배경 밝기 지도를 추정한다."""
    if dilate_kernel % 2 == 0:
        dilate_kernel += 1
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
    bg = cv2.dilate(gray, kernel)          # 밝은 쪽으로 팽창 -> 얇은 어두운 선 제거
    bg = cv2.medianBlur(bg, blur_kernel)   # 잔여 얼룩 평탄화
    return bg


def clean_image(img_path, out_path, threshold=230, dilate_kernel=25, blur_kernel=21,
                 min_gray_floor=200):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[읽기 실패] {img_path}")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = estimate_background(gray, dilate_kernel=dilate_kernel, blur_kernel=blur_kernel)

    # 국소 배경 대비 상대 밝기로 정규화 (배경 편차 제거 -> 배경은 ~255로 수렴)
    norm = cv2.divide(gray, bg, scale=255)

    # 배경 추정이 어긋나더라도(예: 커널보다 큰 검은 채움 영역) 원본이 이미 어두운
    # 픽셀은 절대 흰색으로 바뀌지 않도록 원본 밝기 기준 안전장치를 추가로 건다.
    mask = (norm >= threshold) & (gray >= min_gray_floor)
    out = img.copy()
    out[mask] = (255, 255, 255)

    cv2.imwrite(out_path, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True


def make_output_filename(original_name):
    """
    qr_crop_000.jpg -> qr_clean_000.jpg 형태로 변환.
    파일명에 'crop'이 있으면 'clean'으로 치환하고,
    '_'만 있으면 첫 번째 '_' 뒤에 'clean_'을 삽입하고,
    '_'가 없으면 앞에 'clean_'을 붙인다.
    """
    root, ext = os.path.splitext(original_name)
    if "_crop_" in root:
        new_root = root.replace("_crop_", "_clean_", 1)
    elif "_" in root:
        prefix, rest = root.split("_", 1)
        new_root = f"{prefix}_clean_{rest}"
    else:
        new_root = f"clean_{root}"
    return new_root + ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="크롭된 이미지 폴더")
    ap.add_argument("--output_dir", required=True, help="배경 정리 결과 저장 폴더")
    ap.add_argument("--threshold", type=int, default=230,
                     help="정규화된 밝기 기준 순백색 처리 임계값 (0~255)")
    ap.add_argument("--dilate_kernel", type=int, default=25,
                     help="배경 추정용 dilate 커널 크기(px). 선 굵기보다 커야 함")
    ap.add_argument("--blur_kernel", type=int, default=21,
                     help="배경 추정용 median blur 커널 크기(px, 홀수)")
    ap.add_argument("--min_gray_floor", type=int, default=200,
                     help="이 값보다 어두운 원본 픽셀은 절대 흰색으로 바꾸지 않음(안전장치)")
    ap.add_argument("--ext", default="jpg", help="입력 확장자 (jpg, png 등)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, f"*.{args.ext}")))

    print(f"총 {len(files)}개 파일 처리 시작")
    fail_list = []
    for f in files:
        out_name = make_output_filename(os.path.basename(f))
        out_path = os.path.join(args.output_dir, out_name)
        ok = clean_image(f, out_path, threshold=args.threshold,
                          dilate_kernel=args.dilate_kernel, blur_kernel=args.blur_kernel,
                          min_gray_floor=args.min_gray_floor)
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
