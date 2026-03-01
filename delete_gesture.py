import os
import sys
import shutil
import numpy as np
from gestures import GESTURE_MAP

CSV_PATH = "data/gesture_train.csv"

gesture_names = list(GESTURE_MAP.keys())

if len(sys.argv) < 2 or sys.argv[1] not in GESTURE_MAP:
    print(f"Usage: python delete_gesture.py <gesture_name>")
    print(f"  가능한 제스처: {', '.join(gesture_names)}")
    sys.exit(1)

gesture_name = sys.argv[1]
label = GESTURE_MAP[gesture_name]

# CSV에서 해당 라벨 데이터 삭제
if os.path.exists(CSV_PATH):
    data = np.genfromtxt(CSV_PATH, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    before = data.shape[0]
    data = data[data[:, -1] != label]
    after = data.shape[0]
    removed = before - after

    if data.shape[0] > 0:
        np.savetxt(CSV_PATH, data, delimiter=",")
    else:
        os.remove(CSV_PATH)

    print(f"CSV: {removed}개 삭제 ({before} -> {after})")
else:
    print("CSV 파일이 없습니다.")

# 이미지 폴더 삭제
img_dir = f"data/images/{gesture_name}"
if os.path.exists(img_dir):
    count = len(os.listdir(img_dir))
    shutil.rmtree(img_dir)
    print(f"이미지: {img_dir}/ 삭제 ({count}개)")
else:
    print(f"이미지 폴더 없음: {img_dir}/")
