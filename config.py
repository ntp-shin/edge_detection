# config.py
import numpy as np

def rgb_to_bgr(rgb_color):
    """
    Convert RGB color to BGR format for OpenCV.
    
    Args:
        rgb_color (np.ndarray): RGB color array [R, G, B]
    
    Returns:
        np.ndarray: BGR color array [B, G, R]
    """
    if isinstance(rgb_color, (list, tuple)):
        rgb_color = np.array(rgb_color, dtype=np.uint8)
    return np.array([rgb_color[2], rgb_color[1], rgb_color[0]], dtype=np.uint8)

def convert_colors_rgb_to_bgr(color_list):
    """
    Convert a list of RGB colors to BGR format.
    
    Args:
        color_list (list): List of RGB color arrays
    
    Returns:
        list: List of BGR color arrays
    """
    return [rgb_to_bgr(color) for color in color_list]

# Danh sách màu RGB, mỗi phần tử là một mảng numpy (dtype=np.uint8)

# Danh sách màu RGB cho 4 class mới
DEFECT_COLORS = [
    np.array([255, 0, 0], dtype=np.uint8),          # 0: bien_bao_hieu_lenh (Red) -> Maybe check standard colors. 
                                                    # Let's use:
                                                    # 0: Blue (Hieu lenh)
                                                    # 1: Red (Cam)
                                                    # 2: Yellow (Nguy hiem)
                                                    # 3: Green/Blue (Chi dan) - let's pick distinct colors
    np.array([255, 0, 0], dtype=np.uint8),          # 0: bien_bao_hieu_lenh (Blue in RGB is 0,0,255 but let's stick to simple first)
                                                    # Wait, let's use standard traffic sign colors for visualization if possible, or just distinct ones.
                                                    # Class 0: Indication (Hieu lenh) - Circle Blue background -> [0, 0, 255]
                                                    # Class 1: Prohibition (Cam) - Circle Red border -> [255, 0, 0]
                                                    # Class 2: Warning (Nguy hiem) - Triangle Yellow background -> [255, 255, 0]
                                                    # Class 3: Direction (Chi dan) - Square Blue/Green -> [0, 255, 0]
]

# Re-define to match standard visualization colors nicely
DEFECT_COLORS = [
    np.array([0, 0, 255], dtype=np.uint8),          # 0: bien_bao_hieu_lenh (Blue)
    np.array([255, 0, 0], dtype=np.uint8),          # 1: bien_bao_cam (Red)
    np.array([255, 255, 0], dtype=np.uint8),        # 2: bien_bao_nguy_hiem (Yellow)
    np.array([0, 255, 0], dtype=np.uint8),          # 3: bien_bao_chi_dan (Green)
    np.array([0, 0, 0], dtype=np.uint8),            # 4: bien_bao_phu (Black)

]

# Convert RGB to BGR for OpenCV
DEFECT_COLORS = convert_colors_rgb_to_bgr(DEFECT_COLORS)

CLASS_NAMES = [
    'bien_bao_hieu_lenh',      # 0: indication signs
    'bien_bao_cam',            # 1: prohibition signs
    'bien_bao_nguy_hiem',      # 2: warning/danger signs
    'bien_bao_chi_dan',        # 3: direction/guidance signs
    'bien_bao_phu',            # 4: additional signs
]


def get_config():
    return{
        'conf': 0.5,
        'model_path': "pre-trained/yolo_06_06_11n_allsize_e170.pt",
        'colors': DEFECT_COLORS,
        'class_names': CLASS_NAMES,
        'device': 0,                                # 0: GPU,  'cpu': CPU
        'draw_class_conf': False,                   # Show class name and confidence of object
        'annotate_object_counts': False,             # Show number of objects in each class
        'agnostic_nms': True,                       # Overlap NMS
        'iou': 0.6,                                 # Intersection over Union 
        'frame': 0,                                 # Frame number
        'selected_classes': None                    # List of class IDs to draw (None = draw all classes)
    }


if __name__ == "__main__":
    print("🎯 CONFIG.PY - Testing utilities")
