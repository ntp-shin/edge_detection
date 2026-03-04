"""
process_data.py
- Chia dữ liệu YOLO segment thành tập train / valid
- Augment tập train để tăng cường dữ liệu
"""

import os
import shutil
import random
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml

# ========================== CẤU HÌNH ==========================
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

# Thư mục nguồn
SRC_IMAGES_DIR = DATA_DIR / "images"
SRC_LABELS_DIR = DATA_DIR / "labels"

# Thư mục đầu ra (YOLO format)
OUTPUT_DIR = DATA_DIR / "processed"
TRAIN_IMAGES_DIR = OUTPUT_DIR / "images" / "train"
VALID_IMAGES_DIR = OUTPUT_DIR / "images" / "valid"
TRAIN_LABELS_DIR = OUTPUT_DIR / "labels" / "train"
VALID_LABELS_DIR = OUTPUT_DIR / "labels" / "valid"

# Tỷ lệ chia
VALID_RATIO = 0.2  # 20% cho validation
RANDOM_SEED = 42

# Số lần augment mỗi ảnh train
NUM_AUGMENTATIONS = 3


# ========================== TIỆN ÍCH ==========================

def parse_yolo_segment_label(label_path):
    """
    Đọc file label YOLO segment.
    Mỗi dòng: class_id x1 y1 x2 y2 ... xn yn
    Trả về list of (class_id, polygon_points)
    polygon_points là list of (x, y) normalized.
    """
    annotations = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # ít nhất class + 3 điểm (6 tọa độ)
                continue
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            annotations.append((class_id, points))
    return annotations


def write_yolo_segment_label(label_path, annotations):
    """
    Ghi file label YOLO segment.
    annotations: list of (class_id, polygon_points)
    """
    with open(label_path, "w") as f:
        for class_id, points in annotations:
            coords_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
            f.write(f"{class_id} {coords_str}\n")


def denormalize_polygon(points, img_w, img_h):
    """Chuyển polygon từ normalized (0-1) sang pixel coordinates."""
    return [(x * img_w, y * img_h) for x, y in points]


def normalize_polygon(points, img_w, img_h):
    """Chuyển polygon từ pixel coordinates sang normalized (0-1)."""
    return [(x / img_w, y / img_h) for x, y in points]


def clip_polygon(points):
    """Clip polygon coordinates về [0, 1]."""
    return [(max(0.0, min(1.0, x)), max(0.0, min(1.0, y))) for x, y in points]


# ========================== CHIA DỮ LIỆU ==========================

def collect_data_pairs():
    """
    Thu thập các cặp (image_path, label_path) hợp lệ.
    Chỉ giữ lại những ảnh có label tương ứng.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    pairs = []

    for img_file in sorted(SRC_IMAGES_DIR.iterdir()):
        if img_file.suffix.lower() not in image_extensions:
            continue
        label_file = SRC_LABELS_DIR / (img_file.stem + ".txt")
        if label_file.exists():
            pairs.append((img_file, label_file))
        else:
            print(f"[WARN] Không tìm thấy label cho: {img_file.name}")

    return pairs


def split_train_valid(pairs):
    """Chia dữ liệu thành train / valid."""
    train_pairs, valid_pairs = train_test_split(
        pairs, test_size=VALID_RATIO, random_state=RANDOM_SEED, shuffle=True
    )
    print(f"Tổng: {len(pairs)} | Train: {len(train_pairs)} | Valid: {len(valid_pairs)}")
    return train_pairs, valid_pairs


def copy_pairs(pairs, images_dir, labels_dir):
    """Copy ảnh và label vào thư mục đích."""
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path in pairs:
        shutil.copy2(img_path, images_dir / img_path.name)
        shutil.copy2(lbl_path, labels_dir / lbl_path.name)


# ========================== AUGMENTATION ==========================

def create_augmentation_pipeline():
    """
    Tạo pipeline augmentation phù hợp cho YOLO segment.
    Chỉ dùng các phép biến đổi hình học + màu sắc cơ bản.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
            scale=(0.95, 1.05),
            p=0.5,
        ),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
            A.CLAHE(clip_limit=2.0, p=1.0),
        ], p=0.7),
        A.ImageCompression(quality_range=(75, 95), p=0.2),
    ],
        # Dùng keypoint transform cho polygon
        # Albumentations không hỗ trợ trực tiếp polygon ngoài bbox/keypoint
        # Ta sẽ xử lý polygon thủ công qua keypoints
        keypoint_params=A.KeypointParams(
            format="xy",
            remove_invisible=False,
        ),
    )


def augment_image_with_polygons(image, annotations, transform):
    """
    Augment ảnh cùng với polygon labels.

    Args:
        image: numpy array (BGR)
        annotations: list of (class_id, normalized_points)
        transform: albumentations Compose

    Returns:
        aug_image, aug_annotations hoặc None nếu thất bại
    """
    img_h, img_w = image.shape[:2]

    # Chuẩn bị keypoints và mapping về annotation
    all_keypoints = []
    keypoint_mapping = []  # (annotation_idx, point_idx)

    for ann_idx, (class_id, points) in enumerate(annotations):
        for pt_idx, (nx, ny) in enumerate(points):
            px = nx * img_w
            py = ny * img_h
            # Clip để đảm bảo nằm trong ảnh
            px = max(0, min(img_w - 1, px))
            py = max(0, min(img_h - 1, py))
            all_keypoints.append((px, py))
            keypoint_mapping.append((ann_idx, pt_idx))

    if len(all_keypoints) == 0:
        return None

    try:
        result = transform(image=image, keypoints=all_keypoints)
    except Exception as e:
        print(f"[WARN] Augmentation failed: {e}")
        return None

    aug_image = result["image"]
    aug_keypoints = result["keypoints"]
    aug_h, aug_w = aug_image.shape[:2]

    # Rebuild annotations từ augmented keypoints
    # Tạo dict tạm
    rebuilt = {}
    for kp_idx, (ann_idx, pt_idx) in enumerate(keypoint_mapping):
        if kp_idx >= len(aug_keypoints):
            continue
        kpx, kpy = aug_keypoints[kp_idx]
        # Normalize
        nx = kpx / aug_w
        ny = kpy / aug_h
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        if ann_idx not in rebuilt:
            rebuilt[ann_idx] = []
        rebuilt[ann_idx].append((pt_idx, (nx, ny)))

    # Tạo annotations mới
    aug_annotations = []
    for ann_idx, (class_id, original_points) in enumerate(annotations):
        if ann_idx not in rebuilt:
            continue
        # Sắp xếp theo pt_idx gốc
        pts = sorted(rebuilt[ann_idx], key=lambda x: x[0])
        new_points = [p for _, p in pts]
        # Chỉ giữ nếu còn ≥ 3 điểm
        if len(new_points) >= 3:
            aug_annotations.append((class_id, new_points))

    if len(aug_annotations) == 0:
        return None

    return aug_image, aug_annotations


def augment_training_data(train_pairs):
    """
    Augment toàn bộ tập train.
    Mỗi ảnh gốc → NUM_AUGMENTATIONS ảnh augmented.
    """
    transform = create_augmentation_pipeline()
    aug_count = 0
    fail_count = 0

    for img_path, lbl_path in train_pairs:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[WARN] Không đọc được ảnh: {img_path.name}")
            continue

        annotations = parse_yolo_segment_label(lbl_path)
        if len(annotations) == 0:
            continue

        stem = img_path.stem
        suffix = img_path.suffix

        for i in range(NUM_AUGMENTATIONS):
            result = augment_image_with_polygons(image, annotations, transform)
            if result is None:
                fail_count += 1
                continue

            aug_image, aug_annotations = result
            aug_name = f"{stem}_aug{i}"

            # Lưu ảnh augmented
            aug_img_path = TRAIN_IMAGES_DIR / f"{aug_name}{suffix}"
            cv2.imwrite(str(aug_img_path), aug_image)

            # Lưu label augmented
            aug_lbl_path = TRAIN_LABELS_DIR / f"{aug_name}.txt"
            write_yolo_segment_label(aug_lbl_path, aug_annotations)

            aug_count += 1

    print(f"Augmentation hoàn tất: {aug_count} ảnh tạo thêm | {fail_count} lỗi")
    return aug_count


# ========================== TẠO data.yaml ==========================

def create_data_yaml(num_classes, class_names):
    """Tạo file data.yaml cho YOLO."""
    data = {
        "path": str(OUTPUT_DIR.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    print(f"Đã tạo: {yaml_path}")


# ========================== MAIN ==========================

def main():
    print("=" * 60)
    print("PROCESS DATA - YOLO SEGMENT")
    print("=" * 60)

    # 1. Thu thập dữ liệu
    print("\n[1/5] Thu thập dữ liệu...")
    pairs = collect_data_pairs()
    if len(pairs) == 0:
        print("[ERROR] Không tìm thấy cặp image-label nào!")
        return
    print(f"Tìm thấy {len(pairs)} cặp image-label hợp lệ")

    # 2. Chia train / valid
    print("\n[2/5] Chia train / valid...")
    train_pairs, valid_pairs = split_train_valid(pairs)

    # 3. Xóa output cũ và copy dữ liệu
    print("\n[3/5] Copy dữ liệu vào thư mục train/valid...")
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    copy_pairs(train_pairs, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)
    copy_pairs(valid_pairs, VALID_IMAGES_DIR, VALID_LABELS_DIR)
    print(f"  Train: {len(train_pairs)} ảnh → {TRAIN_IMAGES_DIR}")
    print(f"  Valid: {len(valid_pairs)} ảnh → {VALID_IMAGES_DIR}")

    # 4. Augment tập train
    print(f"\n[4/5] Augment tập train (x{NUM_AUGMENTATIONS} mỗi ảnh)...")
    # Tạo lại train_pairs dựa trên file đã copy (để đường dẫn đúng)
    train_pairs_copied = [
        (TRAIN_IMAGES_DIR / img.name, TRAIN_LABELS_DIR / lbl.name)
        for img, lbl in train_pairs
    ]
    aug_count = augment_training_data(train_pairs_copied)

    # 5. Tạo data.yaml
    print("\n[5/5] Tạo data.yaml...")
    # Đọc class names từ data.yaml gốc
    src_yaml = DATA_DIR / "data.yaml"
    if src_yaml.exists():
        with open(src_yaml, "r") as f:
            src_data = yaml.safe_load(f)
        class_names = list(src_data.get("names", {}).values())
    else:
        # Fallback: đếm số class từ labels
        all_classes = set()
        for _, lbl_path in pairs:
            anns = parse_yolo_segment_label(lbl_path)
            for cid, _ in anns:
                all_classes.add(cid)
        class_names = [f"class_{i}" for i in range(max(all_classes) + 1)]

    create_data_yaml(len(class_names), class_names)

    # Thống kê cuối
    total_train = len(list(TRAIN_IMAGES_DIR.glob("*")))
    total_valid = len(list(VALID_IMAGES_DIR.glob("*")))
    print("\n" + "=" * 60)
    print("KẾT QUẢ:")
    print(f"  Train: {total_train} ảnh (gốc: {len(train_pairs)}, augmented: {aug_count})")
    print(f"  Valid: {total_valid} ảnh")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Config: {OUTPUT_DIR / 'data.yaml'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
