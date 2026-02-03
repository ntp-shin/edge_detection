import cv2
import os
import numpy as np
from pathlib import Path

def draw_rectangle_on_images(input_folder, output_folder=None):
    """
    Đọc ảnh từ folder và vẽ hình chữ nhật xanh lá ở góc trên bên phải
    
    Args:
        input_folder: Đường dẫn đến folder chứa ảnh
        output_folder: Đường dẫn đến folder lưu ảnh đã vẽ (nếu None sẽ tạo folder mới)
    """
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"Folder không tồn tại: {input_folder}")
        return
    
    # Tạo folder output nếu không được chỉ định
    if output_folder is None:
        output_folder = input_path.parent / f"{input_path.name}_with_rectangle"
    
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    # Các định dạng ảnh được hỗ trợ
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Kích thước hình chữ nhật
    rect_width = 200
    rect_height = 100
    
    # Màu xanh lá (BGR format)
    green_color = (0, 255, 0)
    
    # Độ dày viền
    thickness = 2
    
    processed_count = 0
    
    for image_file in input_path.iterdir():
        if image_file.suffix.lower() in supported_formats:
            try:
                # Đọc ảnh
                img = cv2.imread(str(image_file))
                if img is None:
                    print(f"Không thể đọc ảnh: {image_file.name}")
                    continue
                
                # Lấy kích thước ảnh
                height, width = img.shape[:2]
                
                # Tính toán vị trí góc trên bên phải (cách góc 20px)
                margin = 20
                top_left_x = width - rect_width - margin
                top_left_y = margin
                bottom_right_x = width - margin
                bottom_right_y = margin + rect_height
                
                # Kiểm tra xem hình chữ nhật có vừa trong ảnh không
                if top_left_x < 0 or top_left_y < 0:
                    print(f"Ảnh {image_file.name} quá nhỏ để vẽ hình chữ nhật")
                    continue
                
                # Vẽ hình chữ nhật
                cv2.rectangle(img, 
                            (top_left_x, top_left_y), 
                            (bottom_right_x, bottom_right_y), 
                            green_color, 
                            thickness)
                
                # Lưu ảnh
                output_file = output_path / image_file.name
                cv2.imwrite(str(output_file), img)
                
                processed_count += 1
                print(f"Đã xử lý: {image_file.name}")
                
            except Exception as e:
                print(f"Lỗi khi xử lý {image_file.name}: {str(e)}")
    
    print(f"\nHoàn thành! Đã xử lý {processed_count} ảnh.")
    print(f"Ảnh đã được lưu tại: {output_path}")

def preview_single_image(image_path):
    """
    Xem trước một ảnh với hình chữ nhật đã vẽ
    
    Args:
        image_path: Đường dẫn đến ảnh
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return
    
    # Lấy kích thước ảnh
    height, width = img.shape[:2]
    
    # Kích thước hình chữ nhật
    rect_width = 200
    rect_height = 100
    
    # Màu xanh lá (BGR format)
    green_color = (0, 255, 0)
    
    # Độ dày viền
    thickness = 2
    
    # Tính toán vị trí góc trên bên phải (cách góc 20px)
    margin = 20
    top_left_x = width - rect_width - margin
    top_left_y = margin
    bottom_right_x = width - margin
    bottom_right_y = margin + rect_height
    
    # Vẽ hình chữ nhật
    cv2.rectangle(img, 
                (top_left_x, top_left_y), 
                (bottom_right_x, bottom_right_y), 
                green_color, 
                thickness)
    
    # Hiển thị ảnh
    cv2.imshow('Image with Rectangle', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Đường dẫn đến folder chứa ảnh
    input_folder = "/mnt/tphat/dataset/Marcro Test_1"
    
    # Xử lý tất cả ảnh trong folder
    draw_rectangle_on_images(input_folder)
    
    # Uncomment dòng dưới để xem trước một ảnh cụ thể
    # preview_single_image("/path/to/your/image.jpg")
