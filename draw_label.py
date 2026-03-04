from ultralytics import YOLO
import numpy as np
import cv2
import os
from config import get_config

def annotate_object_counts(image, counts, font_scale=1, thickness=2, margin=10, config=get_config()):
    """
    Ghi thống kê số lượng object theo từng class lên góc phải của ảnh.
    
    Tham số:
      image: Ảnh gốc dạng numpy array.
      counts: dict chứa thông tin số lượng object theo từng class, ví dụ: {0: 2, 1: 5}
      colors: list hoặc mảng numpy chứa màu cho từng class, ví dụ: 
              [ (B,G,R) cho class 0, (B,G,R) cho class 1, ... ]
      font_scale: kích thước font chữ.
      thickness: độ dày của nét chữ.
      margin: khoảng cách biên từ cạnh ảnh.
      
    Trả về:
      image đã được ghi thông tin thống kê.
    """
    # Chọn font chữ của OpenCV
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Lấy kích thước ảnh
    h, w = image.shape[:2]
    
    # Danh sách tên của các class (có thể thay đổi theo yêu cầu)
    class_names = config['class_names']
    colors = config['colors']
    
    # Sắp xếp các class theo id để in theo thứ tự tăng dần
    sorted_counts = sorted(counts.items(), key=lambda x: x[0])
    
    # Tính chiều cao của một dòng text (dùng text "Class 0: 0" làm mẫu)
    (text_width, text_height), baseline = cv2.getTextSize("Class 0: 0", font, font_scale, thickness)
    
    # Tọa độ y bắt đầu (dòng đầu tiên)
    y0 = margin + text_height
    
    for i, (cls, cnt) in enumerate(sorted_counts):
        # Kiểm tra chỉ số class có nằm trong danh sách tên không
        class_name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
        line = f"Class {class_name}: {cnt}"
        
        # Tính kích thước của dòng text
        (line_width, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        # Tọa độ x căn chỉnh sang phải với margin
        x = w - margin - line_width
        # Tọa độ y cho dòng hiện tại
        y = y0 + i * (text_height + baseline + 5)  # 5 pixel giữa các dòng

        # Lấy màu tương ứng với class từ danh sách colors
        # Giả sử colors là list hoặc mảng numpy có cấu trúc như: [(B, G, R), ...]
        color = colors[cls] if cls < len(colors) else (255, 255, 255)
        # Nếu colors là numpy array, chuyển về tuple các số nguyên
        if isinstance(color, np.ndarray):
            color = tuple(int(c) for c in color.tolist())
        else:
            color = tuple(color)

        # Vẽ text với viền đen để nổi bật
        cv2.putText(image, line, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        # Vẽ text với màu của class
        cv2.putText(image, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    return image

def draw_class_confident(image, pts, label_text, color, font_scale=0.75, thickness=1, margin=10, config=get_config()):
    if not config['draw_class_conf']:
        return image

    # Get the bounding rectangle of the polygon
    x, y, w_rect, h_rect = cv2.boundingRect(pts)

    # We will draw the text at the top-right corner of the bounding rectangle
    position = (x + w_rect, y)          # Top-right corner
    if position[0] / image.shape[1] > 0.8:
        position = (x - 2 * margin, y)      # Top-left corner
    font = cv2.FONT_HERSHEY_SIMPLEX


    cv2.putText(image, label_text, position, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, label_text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return image


def draw_image_segment(image, model_path=None, label_path=None, config=get_config()):
    """
    Vẽ annotation lên image (video) theo phong cách chung.
    
    Nếu label_path được cung cấp và tồn tại, hàm sẽ đọc file nhãn theo định dạng YOLO segmentation:
      - Mỗi dòng: class_id x1 y1 x2 y2 ... (tọa độ chuẩn hóa)
    Nếu không có label_path, hàm sẽ chạy inference với model tại model_path để lấy detections.
    
    Sau đó, hàm vẽ bounding box (hoặc segmentation) và hiển thị tên đối tượng kèm độ chính xác
    (nếu có) tại góc trên bên phải của đối tượng.
    
    Tham số:
      - image: image từ video (numpy array)
      - model_path: đường dẫn đến model YOLO (sẽ dùng để chạy inference nếu không có file nhãn)
      - label_path: đường dẫn tới file nhãn (nếu có)
      - config: cấu hình chứa 'colors', 'conf', 'device', 'draw_conf', 'class_names',...
      
    Trả về:
      - image đã được vẽ annotation.
    """
    # Check if image is a path to an image
    if type(image) is str: 
        image = cv2.imread(image)
    colors = config['colors']
    h, w = image.shape[:2]      # The height and width of the image
    annotations = []            # The list of annotations. Ex: [(pts, class_id, conf), ...]
    class_counts = {}           # The count of each class in the image

    if label_path is not None and os.path.exists(label_path):
        # Use the ground truth label
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                
                # Check if it's bounding box format (4 values) or segmentation format (8+ values)
                if len(coords) == 4:
                    # Bounding box format: x_center, y_center, width, height (normalized)
                    x_center, y_center, bbox_width, bbox_height = coords
                    # Convert to corner coordinates
                    x1 = (x_center - bbox_width / 2) * w
                    y1 = (y_center - bbox_height / 2) * h
                    x2 = (x_center + bbox_width / 2) * w
                    y2 = (y_center + bbox_height / 2) * h
                    # Create polygon points for the bounding box
                    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                else:
                    # Segmentation format: x1, y1, x2, y2, ..., xn, yn
                    pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
                    # Normalize the points if they are not
                    if pts.max() <= 1:
                        pts[:, 0] *= w
                        pts[:, 1] *= h
                    pts = pts.astype(np.int32)
                
                annotations.append((pts, class_id, None))
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
    else:
        # Run inference with YOLO model
        model = YOLO(model_path)
        results = model(
                            image,
                            conf=config['conf'], 
                            device=config['device'], 
                            agnostic_nms=config['agnostic_nms'], 
                            iou=config['iou']
                        )
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            return image

        predictions = results[0].boxes.xyxy.cpu().numpy()               # [x1, y1, x2, y2]
        pred_classes = results[0].boxes.cls.cpu().numpy().astype(int)   # class_id
        confidences = results[0].boxes.conf.cpu().numpy()               # confidence of each detection
        detections = []

        # Check if segmentation points are available or not
        if hasattr(results[0], 'masks') and results[0].masks is not None and len(results[0].masks) > 0:
            segs = results[0].masks.xy  # Danh sách các mảng numpy, mỗi mảng có shape (N, 2)
            for i, (box, cls, conf) in enumerate(zip(predictions, pred_classes, confidences)):
                seg_points = segs[i]
                # Gộp bbox, class_id, confidence và segmentation points
                detection = list(box) + [cls, conf] + seg_points.flatten().tolist()
                detections.append(detection)
        else:
            for box, cls, conf in zip(predictions, pred_classes, confidences):
                # Dùng bounding box, thêm class_id và confidence
                detection = list(box) + [cls, conf]
                detections.append(detection)
        
        for det in detections:
            if len(det) >= 6:
                if len(det) > 6:
                    # Segmentation points are available. Ex format: [x1, y1, x2, y2, class_id, conf, p1x, p1y, ..., pNx, pNy]
                    class_id = det[4]
                    conf_val = det[5]
                    seg_points = np.array(det[6:], dtype=np.float32).reshape(-1, 2)
                    if seg_points.max() <= 1:
                        seg_points[:, 0] *= w
                        seg_points[:, 1] *= h
                    pts = seg_points.astype(np.int32)
                else:
                    # Only box and class_id (no segmentation)
                    x1, y1, x2, y2, class_id, conf_val = det
                    pts = np.array([[int(x1), int(y1)],
                                    [int(x2), int(y1)],
                                    [int(x2), int(y2)],
                                    [int(x1), int(y2)]], dtype=np.int32)
            else:
                x1, y1, x2, y2, class_id = det
                pts = np.array([[int(x1), int(y1)],
                                [int(x2), int(y1)],
                                [int(x2), int(y2)],
                                [int(x1), int(y2)]], dtype=np.int32)
                conf_val = None
            annotations.append((pts, int(class_id), conf_val))
            class_counts[int(class_id)] = class_counts.get(int(class_id), 0) + 1

    # Filter annotations by selected classes if specified
    selected_classes = config.get('selected_classes', None)
    if selected_classes is not None:
        # Filter annotations to only include selected classes
        filtered_annotations = []
        filtered_counts = {}
        for pts, class_id, conf_val in annotations:
            if class_id in selected_classes:
                filtered_annotations.append((pts, class_id, conf_val))
                filtered_counts[class_id] = filtered_counts.get(class_id, 0) + 1
        annotations = filtered_annotations
        class_counts = filtered_counts

    # Draw annotations on the image
    class_names = config['class_names']
    for pts, class_id, conf_val in annotations:
        # Draw polygon
        color = tuple(int(c) for c in colors[class_id].tolist())

        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
        label_text = f"{class_names[class_id]}"
        if conf_val is not None:
            label_text += f": {conf_val:.2f}"

        # Draw class name and confidence
        image = draw_class_confident(image, pts, label_text, color, font_scale=0.75, thickness=1, margin=10, config=config)

    # Annotate object counts
    if config['annotate_object_counts']:
        image = annotate_object_counts(image, class_counts)

    return image

if __name__ == '__main__':
    print("=" * 60)
    print("Starting draw_label.py script...")
    print("=" * 60)
    
    images_dir = 'data/raw_images'
    labels_dir = 'data/labels'
    output_dir = 'data/images'
    os.makedirs(output_dir, exist_ok=True)

    model_path = '/mnt/data/ntpshin/roadAI/RDD2022andYOLO/pre-trained/yolo_07_03_11n_allsize_def_aug_e160.pt'
    use_model = False
    use_label = True

    print(f"📁 Images directory: {images_dir}")
    print(f"📁 Labels directory: {labels_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Use model: {use_model}")
    print(f"🏷️  Use label: {use_label}")
    print("-" * 60)

    config_defect = get_config()
    print("✅ Loaded defect config")
    

    image_names = None  
    # Uncomment below to use specific images
    # image_names = [
    #     'MLDAcut_2025-06-06_08h_51m_58s_frame-005000_001000.jpg',
    #     'MLDAcut_2025-06-06_08h_51m_58s_frame-005130_001026.jpg',
    # ]
    
    if image_names is not None:
        image_names = image_names
    else:
        # Get all image files from directory
        print("🔍 Scanning images directory...")
        image_names = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        print(f"   Found {len(image_names)} image files")
        
        # If use_label is True, filter only images that have corresponding labels
        if use_label:
            print("🔍 Filtering images with labels...")
            image_names_with_labels = []
            for img_name in image_names:
                label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
                label_path = os.path.join(labels_dir, label_name)
                if os.path.exists(label_path):
                    image_names_with_labels.append(img_name)
            image_names = image_names_with_labels
            print(f"   Found {len(image_names)} images with labels")
    
    print(f"\n📊 Total images to process: {len(image_names)}")
    print("-" * 60)
    
    # Draw images with labels
    processed_count = 0
    for idx, img_name in enumerate(image_names, 1):
        if img_name.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            print(f"\n[{idx}/{len(image_names)}] Processing: {img_name}")
            img_path = os.path.join(images_dir, img_name)
            label_path = os.path.join(labels_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
            base_name = os.path.splitext(img_name)[0]

            if use_model:
                print(f"  🔍 Running model prediction...")
                image_model = cv2.imread(img_path)
                if image_model is None:
                    print(f"  ❌ Cannot read image: {img_path}")
                    continue
                if image_model.shape[:2] != (1080, 1920):
                    image_model = cv2.resize(image_model, (1920, 1080), interpolation=cv2.INTER_AREA)
                # If the image resolution is not 1920x1080, resize it to 1920x1080
                output_path = os.path.join(output_dir, f"{base_name}_pred.jpg")
                image_model = draw_image_segment(image_model, model_path=model_path, label_path=None, config=config_defect)
                cv2.imwrite(output_path, image_model)
                print(f"  ✅ Saved prediction: {output_path}")

            if not os.path.exists(label_path) and use_label:
                print(f"  ⚠️  Label not found: {label_path}")
                continue

            if use_label:
                print(f"  🏷️  Drawing ground truth labels...")
                image_label = cv2.imread(img_path)
                if image_label is None:
                    print(f"  ❌ Cannot read image: {img_path}")
                    continue
                if image_label.shape[:2] != (1080, 1920):
                    image_label = cv2.resize(image_label, (1920, 1080), interpolation=cv2.INTER_AREA)
                output_path = os.path.join(output_dir, f"{base_name}.jpg")
                image_label = draw_image_segment(image_label, model_path=None, label_path=label_path, config=config_defect)
                cv2.imwrite(output_path, image_label)
                print(f"  ✅ Saved ground truth: {output_path}")
            
            processed_count += 1

    print("\n" + "=" * 60)
    print(f"🎉 Finished! Processed {processed_count} images")
    print(f"📁 Output saved to: {output_dir}")
    print("=" * 60)

    '''
        0. o_ga
        1. nut_ngang
        2. mang_nut
        3. bong_troc
        4. luns
        5. nut_doc
        6. vungx_nuoc
        7. bong_lop
        8. nap cong
    '''


