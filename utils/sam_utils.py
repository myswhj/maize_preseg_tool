# SAM工具函数
import cv2
import numpy as np

def mask_to_polygons(mask, epsilon_ratio=0.005, pixel_interval=5):
    """
    将掩码转换为多边形，每5个像素生成一个点
    
    Args:
        mask: 二值掩码
        epsilon_ratio: 多边形近似参数
        pixel_interval: 点间隔，默认5像素
        
    Returns:
        多边形列表
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    
    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        if area <= 50:  # 降低面积阈值，允许更小的区域
            continue
        
        # 多边形近似
        epsilon = max(1.0, epsilon_ratio * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 每5个像素取一个点
        sampled_points = []
        for i, point in enumerate(approx):
            if i % pixel_interval == 0:
                x, y = float(point[0][0]), float(point[0][1])
                sampled_points.append((x, y))
        
        # 确保多边形闭合
        if sampled_points:
            if sampled_points[0] != sampled_points[-1]:
                sampled_points.append(sampled_points[0])
            polygons.append(sampled_points)
    
    return polygons

def process_sam_polygons(polygons):
    """
    处理SAM生成的多边形，确保与手动标注格式一致
    
    Args:
        polygons: SAM生成的多边形列表
        
    Returns:
        处理后的多边形列表
    """
    processed_polygons = []
    
    for polygon in polygons:
        # 确保多边形至少有3个点
        if len(polygon) < 3:
            continue
        
        # 确保多边形闭合
        if polygon[0] != polygon[-1]:
            polygon.append(polygon[0])
        
        processed_polygons.append(polygon)
    
    return processed_polygons