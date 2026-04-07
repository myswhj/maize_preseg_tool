# SAM 模型管理模块
import os

import torch
from segment_anything import SamPredictor, sam_model_registry


class SamManager:
    def __init__(self):
        self.model = None
        self.predictor = None
        self.model_path = None
        self.model_type = None
        self.device = self._resolve_device()

    @staticmethod
    def _resolve_device(device=None):
        if device is None:
            return "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(device, torch.device):
            device = str(device)
        device = str(device)
        if device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return device

    def build_model(self, model_path, model_type="vit_b", device=None):
        if model_type not in sam_model_registry:
            raise ValueError(f"Unsupported SAM model type: {model_type}")

        resolved_device = self._resolve_device(device)
        # Bypass segment_anything's internal torch.load(checkpoint) so checkpoints
        # saved from CUDA devices can still be loaded on CPU-only machines.
        model = sam_model_registry[model_type](checkpoint=None)
        state_dict = torch.load(model_path, map_location=torch.device(resolved_device))
        model.load_state_dict(state_dict)
        model.to(device=resolved_device)
        model.eval()
        self.device = resolved_device
        return model

    def load_model(self, model_path, model_type="vit_b", device=None):
        """加载 SAM 模型。"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self.model_path = model_path
        self.model_type = model_type
        self.model = self.build_model(model_path=model_path, model_type=model_type, device=device)
        self.predictor = SamPredictor(self.model)
        return True

    def get_predictor(self):
        """获取 SAM 预测器。"""
        if not self.predictor:
            raise RuntimeError("模型未加载")
        return self.predictor

    def has_model_loaded(self):
        """检查模型是否已加载。"""
        return self.model is not None
