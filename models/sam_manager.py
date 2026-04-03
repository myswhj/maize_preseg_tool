# SAM模型管理模块
import os
import torch
from segment_anything import SamPredictor, sam_model_registry

class SamManager:
    def __init__(self):
        self.model = None
        self.predictor = None
        self.model_path = None
        self.model_type = None
    
    def load_model(self, model_path, model_type="vit_b"):
        """加载SAM模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        self.model_path = model_path
        self.model_type = model_type
        
        # 加载模型
        self.model = sam_model_registry[model_type](checkpoint=model_path)
        self.model.to(device="cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = SamPredictor(self.model)
        
        return True
    
    def get_predictor(self):
        """获取SAM预测器"""
        if not self.predictor:
            raise RuntimeError("模型未加载")
        return self.predictor
    
    def has_model_loaded(self):
        """检查模型是否已加载"""
        return self.model is not None