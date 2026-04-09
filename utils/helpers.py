# 通用辅助函数

import os

import numpy as np
from PIL import Image as PILImage


def get_plant_color(plant_id):
    """获取植株颜色"""
    np.random.seed(plant_id)
    return (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255), 120)


def validate_image_path(image_path):
    """验证图片路径是否有效"""
    if not image_path:
        return False
    if not os.path.exists(image_path):
        return False
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    ext = os.path.splitext(image_path.lower())[1]
    return ext in valid_extensions


def load_image(image_path):
    """加载图片"""
    if not validate_image_path(image_path):
        return None
    try:
        return PILImage.open(image_path).convert("RGBA")
    except Exception as error:
        print(f"加载图片失败: {error}")
        return None


def get_image_size(image_path):
    """获取图片尺寸"""
    image = load_image(image_path)
    if image:
        return image.width, image.height
    return 0, 0


def calculate_signed_polygon_area(polygon):
    """计算多边形有符号面积。"""
    if not polygon or len(polygon) < 3:
        return 0.0
    area = 0.0
    point_count = len(polygon)
    for index in range(point_count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % point_count]
        area += float(x1) * float(y2) - float(x2) * float(y1)
    return area / 2.0


def calculate_polygon_area(polygon):
    """计算多边形绝对面积。"""
    return abs(calculate_signed_polygon_area(polygon))


def format_image_progress(current, total):
    """格式化图片进度"""
    return f"{current + 1}/{total}"
