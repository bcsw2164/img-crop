"""
네 모서리 정렬마커를 인식해 동일한 크기로 자동 크롭하는 스크립트

사용법:
    python crop_by_markers.py --input_dir ./scans --output_dir ./cropped --size 1000

필요 라이브러리:
    pip install opencv-python-headless numpy
"""

import cv2
import numpy as np
import os
import glob
import argparse


def detect_markers(gray, corner_frac=0.30, min_area=1500):
    """이미지의 네 모서리에서 정렬마커의 중심 좌표를 찾는다."""
    h, w = gray.shape
    _, binimg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    mx, my = int(w * corner_frac), int(h * corner_frac)
    corner_regions = {
        "TL": (0, my, 0, mx),
        "TR": (0, my, w - mx, w),
        "BL": (h - my, h, 0, mx),
        "BR": (h - my, h, w - mx, w),
    }

    centers = {}
    for name, (y0, y1, x0, x1) in corner_regions.items():
        roi = binimg[y0:y1, x0:x1]
        contours, _ = cv2.findContours(roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = bw / float(bh) if bh > 0 else 0
            # 마커는 거의 정사각형 bbox를 가짐 + 라벨스티커/로고보다 작음(60% 이하)
            if 0.7 < aspect < 1.4 and bw < mx * 0.6 and bh < my * 0.6:
                if best is None or area > best[0]:
                    best = (area, x0 + x + bw / 2.0, y0 + y + bh / 2.0)

        if best is None:
            return None  # 검출 실패
        centers[name] = (best[1], best[2])

    return centers


def warp_crop(img_path, out_path, out_size=1000, margin=0, corner_frac=0.30):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[읽기 실패] {img_path}")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    centers = detect_markers(gray, corner_frac=corner_frac)

    if centers is None:
        print(f"[마커 검출 실패] {img_path}  <- 수동 확인 필요")
        return False

    # 마커 4개 중심점 -> 목표 정사각형 좌표로 매핑
    src = np.float32([centers["TL"], centers["TR"], centers["BR"], centers["BL"]])
    dst = np.float32([
        [margin, margin],
        [out_size - margin, margin],
        [out_size - margin, out_size - margin],
        [margin, out_size - margin],
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        img, M, (out_size, out_size),
        flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255)
    )
    cv2.imwrite(out_path, warped, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True


def make_output_filename(original_name):
    """
    qr_001.jpg -> qr_crop_001.jpg 형태로 변환.
    파일명에 '_'가 있으면 첫 번째 '_' 뒤에 'crop_'을 삽입하고,
    '_'가 없으면 앞에 'crop_'을 붙인다.
    """
    root, ext = os.path.splitext(original_name)
    if "_" in root:
        prefix, rest = root.split("_", 1)
        new_root = f"{prefix}_crop_{rest}"
    else:
        new_root = f"crop_{root}"
    return new_root + ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="원본 스캔 이미지 폴더")
    ap.add_argument("--output_dir", required=True, help="크롭 결과 저장 폴더")
    ap.add_argument("--size", type=int, default=1000, help="결과 정사각형 한 변 픽셀 크기")
    ap.add_argument("--margin", type=int, default=0, help="마커 중심 기준 안쪽 여백(px)")
    ap.add_argument("--corner_frac", type=float, default=0.30,
                     help="마커를 찾을 코너 검색영역 비율(이미지 가로/세로의 %%)")
    ap.add_argument("--ext", default="jpg", help="입력 확장자 (jpg, png 등)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, f"*.{args.ext}")))

    print(f"총 {len(files)}개 파일 처리 시작")
    fail_list = []
    for f in files:
        out_name = make_output_filename(os.path.basename(f))
        out_path = os.path.join(args.output_dir, out_name)
        ok = warp_crop(f, out_path, out_size=args.size, margin=args.margin,
                        corner_frac=args.corner_frac)
        if not ok:
            fail_list.append(f)
        else:
            print(f"[OK] {os.path.basename(f)} -> {out_name}")

    print(f"\n완료: {len(files)-len(fail_list)}개 성공 / {len(fail_list)}개 실패")
    if fail_list:
        print("실패한 파일 (수동 확인 필요):")
        for f in fail_list:
            print(" -", f)


if __name__ == "__main__":
    main()
