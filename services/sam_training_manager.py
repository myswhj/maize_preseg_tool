from dataclasses import dataclass
from pathlib import Path
import os

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.annotation_schema import compute_annotation_hash
from utils.helpers import calculate_polygon_area


REPO_ROOT = Path(__file__).resolve().parent.parent


class Config:
    SAM_MODEL_TYPE = "vit_b"
    SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
    INPUT_SIZE = 1024
    BATCH_SIZE = 1
    LEARNING_RATE = 3e-5
    EPOCHS = 30
    NUM_WORKERS = 0 if os.name == "nt" else min(2, os.cpu_count() or 1)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    MASK_DILATE_KERNEL = 5
    MASK_DILATE_ITER = 2
    POS_WEIGHT = 250
    GRADIENT_ACCUMULATION_STEPS = 6
    DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "sam_training"
    BEST_MODEL_FILENAME = "sam_stem_tassel_best.pth"
    LATEST_MODEL_FILENAME = "sam_stem_tassel_latest.pth"
    VALIDATION_DIRNAME = "validation_preview"


@dataclass
class TrainingPaths:
    output_root: Path
    run_dir: Path
    best_model_path: Path
    latest_model_path: Path
    validation_output_dir: Path
    checkpoint_path: Path | None = None


class SingleStemTasselDataset(Dataset):
    def __init__(self, coco_container, image_paths):
        self.samples = self._prepare_single_instance_samples(coco_container, image_paths)
        self.input_size = Config.INPUT_SIZE

    @staticmethod
    def _read_image_rgb(image_path):
        image_path = str(image_path)
        if not image_path or not os.path.exists(image_path):
            return None

        image = cv2.imread(image_path)
        if image is not None:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            buffer = np.fromfile(image_path, dtype=np.uint8)
            if buffer.size > 0:
                decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                if decoded is not None:
                    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        except Exception:
            pass

        try:
            with Image.open(image_path) as pil_image:
                return np.array(pil_image.convert("RGB"))
        except Exception:
            return None

    @staticmethod
    def _build_mask_from_polygons(polygons, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in polygons or []:
            points = np.array(polygon, dtype=np.int32)
            if len(points) < 3:
                continue
            fill_value = 255 if calculate_polygon_area(points.tolist()) < 0 else 0
            cv2.fillPoly(mask, [points], fill_value)
        return mask

    def _prepare_single_instance_samples(self, coco_container, image_paths):
        samples = []
        for image_path in image_paths or []:
            annotation = coco_container.get(image_path)
            if not annotation:
                continue
            if not annotation.get("image_state", {}).get("annotation_completed", False):
                continue

            orig_img = self._read_image_rgb(image_path)
            if orig_img is None:
                print(f"Skip unreadable training image: {image_path}")
                continue
            orig_h, orig_w = orig_img.shape[:2]

            for plant in annotation.get("plants", []):
                polygons = plant.get("polygons") or []
                if not polygons:
                    continue

                all_points = []
                for polygon in polygons:
                    if len(polygon) >= 3:
                        all_points.extend(polygon)
                if len(all_points) < 3:
                    continue

                all_points_np = np.array(all_points, dtype=np.float32)
                x_min, y_min = np.min(all_points_np, axis=0)
                x_max, y_max = np.max(all_points_np, axis=0)

                box_w = x_max - x_min
                box_h = y_max - y_min
                x_min = max(0, x_min - box_w * 0.1)
                y_min = max(0, y_min - box_h * 0.1)
                x_max = min(orig_w, x_max + box_w * 0.1)
                y_max = min(orig_h, y_max + box_h * 0.1)

                samples.append(
                    {
                        "image_path": image_path,
                        "orig_size": (orig_h, orig_w),
                        "plant_polygons": polygons,
                        "bbox": [x_min, y_min, x_max, y_max],
                    }
                )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        orig_h, orig_w = sample["orig_size"]
        plant_polygons = sample["plant_polygons"]
        bbox = sample["bbox"]

        image = self._read_image_rgb(image_path)
        if image is None:
            raise ValueError(f"Unable to read training image: {image_path}")

        mask = self._build_mask_from_polygons(plant_polygons, orig_h, orig_w)

        kernel = np.ones((Config.MASK_DILATE_KERNEL, Config.MASK_DILATE_KERNEL), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=Config.MASK_DILATE_ITER)

        scale = self.input_size / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)

        image_resized = cv2.resize(image, (new_w, new_h))
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        bbox_resized = np.array(bbox, dtype=np.float32) * scale

        image_padded = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        image_padded[:new_h, :new_w, :] = image_resized

        mask_padded = np.zeros((self.input_size, self.input_size), dtype=np.uint8)
        mask_padded[:new_h, :new_w] = mask_resized

        point_prompt = bbox_resized.astype(np.float32)

        if np.random.random() > 0.5:
            image_padded = np.fliplr(image_padded).copy()
            mask_padded = np.fliplr(mask_padded).copy()
            x1, x2 = float(point_prompt[0]), float(point_prompt[2])
            point_prompt[0] = self.input_size - x2
            point_prompt[2] = self.input_size - x1

        image_tensor = torch.from_numpy(image_padded).permute(2, 0, 1).float() / 255.0
        for channel, mean, std in zip(image_tensor, Config.MEAN, Config.STD):
            channel.sub_(mean).div_(std)

        mask_tensor = torch.from_numpy(mask_padded).float() / 255.0
        point_tensor = torch.from_numpy(point_prompt).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "point": point_tensor,
            "orig_size": (orig_h, orig_w),
        }


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        return 1 - (2.0 * intersection + self.smooth) / (union + self.smooth)


class SamTrainingManagerV2:
    def __init__(self, sam_manager=None):
        self.device = Config.DEVICE
        self.sam_manager = sam_manager
        self.model = None
        self.dice_loss = DiceLoss()
        self.best_val_loss = float("inf")
        self.best_val_iou = float("-inf")
        self.last_run_info = {}

    def _resolve_checkpoint_path(self, checkpoint_path=None):
        candidates = []
        if checkpoint_path:
            candidates.append(Path(checkpoint_path))

        configured = Path(Config.SAM_CHECKPOINT)
        candidates.extend(
            [
                configured,
                REPO_ROOT / configured,
                REPO_ROOT / "checkpoints" / configured.name,
            ]
        )

        for candidate in candidates:
            if candidate and candidate.exists():
                return candidate.resolve()
        return None

    def _create_training_paths(self, output_dir=None):
        output_root = Path(output_dir) if output_dir else Config.DEFAULT_OUTPUT_ROOT
        output_root = output_root.expanduser().resolve()
        from datetime import datetime

        run_dir = output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        validation_output_dir = run_dir / Config.VALIDATION_DIRNAME
        validation_output_dir.mkdir(parents=True, exist_ok=True)
        return TrainingPaths(
            output_root=output_root,
            run_dir=run_dir,
            best_model_path=run_dir / Config.BEST_MODEL_FILENAME,
            latest_model_path=run_dir / Config.LATEST_MODEL_FILENAME,
            validation_output_dir=validation_output_dir,
        )

    def _load_sam_model(self, checkpoint_path=None):
        if self.sam_manager and self.sam_manager.has_model_loaded():
            self.model = self.sam_manager.model
            self.model.to(self.device)
        else:
            resolved_checkpoint = self._resolve_checkpoint_path(checkpoint_path)
            if resolved_checkpoint is None:
                raise FileNotFoundError(
                    f"Unable to locate SAM checkpoint: {checkpoint_path or Config.SAM_CHECKPOINT}"
                )
            if self.sam_manager:
                self.model = self.sam_manager.build_model(
                    model_path=str(resolved_checkpoint),
                    model_type=Config.SAM_MODEL_TYPE,
                    device=self.device,
                )
            else:
                self.model = sam_model_registry[Config.SAM_MODEL_TYPE](checkpoint=None)
                state_dict = torch.load(resolved_checkpoint, map_location=torch.device(self.device))
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()

        for param in self.model.image_encoder.parameters():
            param.requires_grad = False
        for param in self.model.image_encoder.blocks[-2:].parameters():
            param.requires_grad = True

    def _compute_iou(self, pred_mask, true_mask):
        pred = (pred_mask > 0.5).float()
        true = (true_mask > 0.5).float()
        intersection = (pred * true).sum()
        union = pred.sum() + true.sum() - intersection
        if union == 0:
            return torch.tensor(1.0, device=self.device)
        return intersection / union

    def _compute_loss(self, batch):
        images = batch["image"].to(self.device)
        masks = batch["mask"].to(self.device)
        boxes = batch["point"].to(self.device)

        batch_size = images.shape[0]
        total_loss = 0.0
        total_iou = 0.0

        with torch.no_grad():
            image_embeddings = self.model.image_encoder(images)

        for index in range(batch_size):
            bbox = boxes[index].unsqueeze(0)
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=bbox,
                masks=None,
            )

            low_res_masks, _ = self.model.mask_decoder(
                image_embeddings=image_embeddings[index : index + 1],
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            pred_masks = torch.nn.functional.interpolate(
                low_res_masks,
                size=(Config.INPUT_SIZE, Config.INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
            )

            mask_gt = masks[index : index + 1].unsqueeze(1)
            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                pred_masks,
                mask_gt,
                pos_weight=torch.tensor([Config.POS_WEIGHT], device=self.device),
            )
            dice_loss = self.dice_loss(pred_masks, mask_gt)
            loss = 0.9 * dice_loss + 0.1 * bce_loss
            total_loss += loss

            with torch.no_grad():
                total_iou += self._compute_iou(torch.sigmoid(pred_masks[0, 0]), mask_gt[0, 0])

        return total_loss / batch_size, total_iou / batch_size

    @staticmethod
    def _split_train_val_samples(samples):
        unique_image_paths = list({sample["image_path"] for sample in samples})
        np.random.shuffle(unique_image_paths)

        if len(unique_image_paths) <= 1:
            train_image_paths = set(unique_image_paths)
            val_image_paths = set(unique_image_paths)
        else:
            val_image_count = max(1, len(unique_image_paths) // 5)
            val_image_count = min(val_image_count, len(unique_image_paths) - 1)
            val_image_paths = set(unique_image_paths[:val_image_count])
            train_image_paths = set(unique_image_paths[val_image_count:])

        train_samples = [sample for sample in samples if sample["image_path"] in train_image_paths]
        val_samples = [sample for sample in samples if sample["image_path"] in val_image_paths]

        if not val_samples:
            val_samples = list(train_samples)
            val_image_paths = set(train_image_paths)
        return train_samples, val_samples, train_image_paths, val_image_paths

    @staticmethod
    def _build_snapshot_hashes(coco_container, image_paths):
        snapshot_hashes = {}
        for image_path in image_paths or []:
            annotation = coco_container.get(image_path)
            if not annotation:
                continue
            image_state = annotation.get("image_state", {})
            if not image_state.get("annotation_completed", False):
                continue
            annotation_hash = annotation.get("annotation_hash")
            if not annotation_hash:
                annotation_hash = compute_annotation_hash(annotation.get("plants", []), image_state)
            snapshot_hashes[image_path] = annotation_hash
        return snapshot_hashes

    @staticmethod
    def _resolve_orig_size(orig_sizes, index):
        default_size = (Config.INPUT_SIZE, Config.INPUT_SIZE)
        try:
            if isinstance(orig_sizes, torch.Tensor):
                size = orig_sizes[index].tolist()
                return int(size[0]), int(size[1])
            if isinstance(orig_sizes, tuple) and len(orig_sizes) == 2:
                first, second = orig_sizes
                return int(first[index]), int(second[index])
            if isinstance(orig_sizes, list):
                item = orig_sizes[index]
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    return int(item[0]), int(item[1])
        except Exception:
            return default_size
        return default_size

    def _sync_best_model_to_runtime(self, best_model_path):
        if self.sam_manager:
            model_type = self.sam_manager.model_type or Config.SAM_MODEL_TYPE
            self.model = self.sam_manager.build_model(
                model_path=str(best_model_path),
                model_type=model_type,
                device=self.device,
            )
            self.sam_manager.model = self.model
            self.sam_manager.predictor = SamPredictor(self.model)
            self.sam_manager.model_path = str(best_model_path)
            self.sam_manager.model_type = model_type
            return

        state_dict = torch.load(best_model_path, map_location=torch.device(self.device))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def train(self, coco_container, image_paths, output_dir=None, checkpoint_path=None):
        self.best_val_loss = float("inf")
        self.best_val_iou = float("-inf")
        self.last_run_info = {}

        training_paths = self._create_training_paths(output_dir=output_dir)
        self._load_sam_model(checkpoint_path=checkpoint_path)

        full_dataset = SingleStemTasselDataset(coco_container, image_paths)
        if len(full_dataset) == 0:
            raise ValueError("No valid completed annotations available for SAM training.")

        train_samples, val_samples, train_image_paths, val_image_paths = self._split_train_val_samples(
            full_dataset.samples
        )
        if not train_samples:
            raise ValueError("Training split is empty after filtering completed images.")

        full_dataset.samples = train_samples
        train_dataset = full_dataset
        val_dataset = SingleStemTasselDataset(coco_container, image_paths)
        val_dataset.samples = val_samples

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=Config.LEARNING_RATE, weight_decay=5e-6)
        scheduler = StepLR(optimizer, step_size=8, gamma=0.5)

        snapshot_hashes = self._build_snapshot_hashes(coco_container, image_paths)
        self.last_run_info = {
            "run_dir": str(training_paths.run_dir),
            "best_model_path": str(training_paths.best_model_path),
            "latest_model_path": str(training_paths.latest_model_path),
            "validation_output_dir": str(training_paths.validation_output_dir),
            "snapshot_hashes": snapshot_hashes,
            "train_image_count": len(train_image_paths),
            "val_image_count": len(val_image_paths),
            "sample_count": len(full_dataset.samples) + len(val_dataset.samples),
        }

        print(f"Training samples: {len(train_dataset)}, validation samples: {len(val_dataset)}")
        print(f"Epochs: {Config.EPOCHS}, device: {self.device}")

        for epoch in range(Config.EPOCHS):
            self.model.train()
            train_loss_sum = 0.0
            train_iou_sum = 0.0
            optimizer.zero_grad()
            accumulation_steps = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS} [Train]")
            for batch in pbar:
                loss, iou = self._compute_loss(batch)
                normalized_loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
                normalized_loss.backward()
                accumulation_steps += 1

                if accumulation_steps >= Config.GRADIENT_ACCUMULATION_STEPS:
                    optimizer.step()
                    optimizer.zero_grad()
                    accumulation_steps = 0

                train_loss_sum += loss.item()
                train_iou_sum += iou.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{iou.item():.4f}"})

            if accumulation_steps:
                optimizer.step()
                optimizer.zero_grad()

            avg_train_loss = train_loss_sum / len(train_loader)
            avg_train_iou = train_iou_sum / len(train_loader)

            self.model.eval()
            val_loss_sum = 0.0
            val_iou_sum = 0.0

            with torch.no_grad():
                pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS} [Val]")
                for batch in pbar:
                    loss, iou = self._compute_loss(batch)
                    val_loss_sum += loss.item()
                    val_iou_sum += iou.item()
                    pbar.set_postfix({"val_loss": f"{loss.item():.4f}", "val_iou": f"{iou.item():.4f}"})

            avg_val_loss = val_loss_sum / len(val_loader)
            avg_val_iou = val_iou_sum / len(val_loader)
            scheduler.step()

            print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
            print(f"  train loss: {avg_train_loss:.4f}, train iou: {avg_train_iou:.4f}")
            print(f"  val loss:   {avg_val_loss:.4f}, val iou:   {avg_val_iou:.4f}")
            print(f"  lr:         {optimizer.param_groups[0]['lr']:.6f}")

            if avg_val_iou >= self.best_val_iou:
                self.best_val_iou = avg_val_iou
                self.best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), training_paths.best_model_path)
                print(f"  saved best model: {training_paths.best_model_path}")

            torch.save(self.model.state_dict(), training_paths.latest_model_path)

        if not training_paths.best_model_path.exists():
            torch.save(self.model.state_dict(), training_paths.best_model_path)

        self._predict_val_set(
            val_loader=val_loader,
            best_model_path=training_paths.best_model_path,
            output_dir=training_paths.validation_output_dir,
        )
        self._sync_best_model_to_runtime(training_paths.best_model_path)
        self.last_run_info.update(
            {
                "best_model_path": str(training_paths.best_model_path),
                "latest_model_path": str(training_paths.latest_model_path),
                "validation_output_dir": str(training_paths.validation_output_dir),
                "best_val_iou": self.best_val_iou,
                "best_val_loss": self.best_val_loss,
            }
        )
        return str(training_paths.best_model_path)

    def _predict_val_set(self, val_loader, best_model_path, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.model.load_state_dict(torch.load(best_model_path, map_location=torch.device(self.device)))
        self.model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                boxes = batch["point"].to(self.device)
                orig_sizes = batch["orig_size"]

                image_embeddings = self.model.image_encoder(images)
                for index in range(images.shape[0]):
                    bbox = boxes[index].unsqueeze(0)
                    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                        points=None,
                        boxes=bbox,
                        masks=None,
                    )
                    low_res_masks, _ = self.model.mask_decoder(
                        image_embeddings=image_embeddings[index : index + 1],
                        image_pe=self.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    pred_masks = torch.nn.functional.interpolate(
                        low_res_masks,
                        size=(Config.INPUT_SIZE, Config.INPUT_SIZE),
                        mode="bilinear",
                        align_corners=False,
                    )

                    pred_mask = (torch.sigmoid(pred_masks[0, 0]) > 0.5).cpu().numpy()
                    true_mask = masks[index].cpu().numpy()

                    image = batch["image"][index].cpu().numpy()
                    for channel_index, (mean, std) in enumerate(zip(Config.MEAN, Config.STD)):
                        image[channel_index] = image[channel_index] * std + mean
                    image = np.clip(image * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0)

                    orig_h, orig_w = self._resolve_orig_size(orig_sizes, index)
                    image = cv2.resize(image, (orig_w, orig_h))
                    pred_mask = cv2.resize(pred_mask.astype(np.float32), (orig_w, orig_h)) > 0.5
                    true_mask = cv2.resize(true_mask.astype(np.float32), (orig_w, orig_h)) > 0.5

                    base_filename = f"val_{batch_idx}_{index}"
                    Image.fromarray(image).save(output_dir / f"{base_filename}_orig.jpg", quality=85, optimize=True)
                    Image.fromarray((pred_mask * 255).astype(np.uint8)).save(
                        output_dir / f"{base_filename}_pred.png",
                        optimize=True,
                        compress_level=9,
                    )
                    Image.fromarray((true_mask * 255).astype(np.uint8)).save(
                        output_dir / f"{base_filename}_true.png",
                        optimize=True,
                        compress_level=9,
                    )

                    overlay = image.copy()
                    overlay[pred_mask] = (0, 255, 0)
                    overlay[true_mask] = (255, 0, 0)
                    Image.fromarray(overlay).save(
                        output_dir / f"{base_filename}_overlay.jpg",
                        quality=85,
                        optimize=True,
                    )

    def start_training(self, coco_container, image_paths, output_dir=None, checkpoint_path=None):
        try:
            best_model_path = self.train(
                coco_container,
                image_paths,
                output_dir=output_dir,
                checkpoint_path=checkpoint_path,
            )
            return True, f"Training finished, best model saved to {best_model_path}"
        except Exception as error:
            import traceback

            traceback.print_exc()
            return False, str(error)
