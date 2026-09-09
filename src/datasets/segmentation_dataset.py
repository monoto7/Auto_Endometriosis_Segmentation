from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

#Making a higher level class to reduce code duplication
class SegmentationDataset(Dataset):
    """
    Dataset for standardized binary segmentation folders:

      split/images/*.jpg
      split/masks/*.png

    Returns:
      image_tensor: float32, shape [3, H, W], normalized to ImageNet stats
      mask_tensor: float32, shape [1, H, W], values {0, 1}
      image_name
    """

    def __init__(
        self,
        images_dir,
        masks_dir,
        image_size=512,
        augment=False,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = int(image_size)
        self.augment = bool(augment)

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images folder not found: {self.images_dir}")

        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks folder not found: {self.masks_dir}")

        self.image_paths = []

        for extension in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]:
            self.image_paths.extend(sorted(self.images_dir.glob(extension)))

        if not self.image_paths:
            raise RuntimeError(f"No images found in: {self.images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def _find_mask_path(self, image_path: Path) -> Path:
        stem = image_path.stem

        for extension in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            candidate = self.masks_dir / f"{stem}{extension}"

            if candidate.exists():
                return candidate

        raise FileNotFoundError(f"Mask not found for image: {image_path.name}")

    def _augment(self, image, mask):
        if not self.augment:
            return image, mask

        if np.random.rand() < 0.5:
            image = np.ascontiguousarray(np.fliplr(image))
            mask = np.ascontiguousarray(np.fliplr(mask))

        if np.random.rand() < 0.5:
            image = np.ascontiguousarray(np.flipud(image))
            mask = np.ascontiguousarray(np.flipud(mask))

        return image, mask

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self._find_mask_path(image_path)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        original_size = image.size  # W, H

        image = np.array(image)
        mask = np.array(mask)

        mask = (mask > 0).astype(np.float32)

        image, mask = self._augment(image, mask)

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        image = image.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        image = (image - mean) / std

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_name": image_path.name,
            "original_width": original_size[0],
            "original_height": original_size[1],
        }

class BinarySegmentationDataset(SegmentationDataset):
    """
    Dataset for standardized binary segmentation folders:

      split/images/*.jpg
      split/masks/*.png

    Returns:
      image_tensor: float32, shape [3, H, W], normalized to ImageNet stats
      mask_tensor: float32, shape [1, H, W], values {0, 1}
      image_name
    """


    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self._find_mask_path(image_path)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        original_size = image.size  # W, H

        image = np.array(image)
        mask = np.array(mask)

        mask = (mask > 0).astype(np.float32)

        image, mask = self._augment(image, mask)

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        image = image.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        image = (image - mean) / std

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_name": image_path.name,
            "original_width": original_size[0],
            "original_height": original_size[1],
        }

class GrayscaleSegmentationDataset(SegmentationDataset):
    """
    Dataset for standardized binary segmentation folders:

      split/images/*.jpg
      split/masks/*.png

    Returns:
      image_tensor: float32, shape [3, H, W], normalized to ImageNet stats
      mask_tensor: float32, shape [1, H, W], values [0, 1]
      image_name
    """


    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self._find_mask_path(image_path)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        original_size = image.size  # W, H

        image = np.array(image)
        mask = np.array(mask)

        #TODO decide whether to scale values in mask between 0-1, likely needs to actually be integer values for segformer
        #Behaviour might have to be different for different models, needs to be looked into once segformer works as a proof of concept.
        mask = mask.astype(np.float32)/255.0
        
        image, mask = self._augment(image, mask)

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        image = image.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        image = (image - mean) / std

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_name": image_path.name,
            "original_width": original_size[0],
            "original_height": original_size[1],
        }