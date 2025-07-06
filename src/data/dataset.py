import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import albumentations as A
from pathlib import Path
from typing import Tuple, List, Optional
import random
from .image_preprocessor import SatelliteImagePreprocessor

class SatelliteDataset(Dataset):
    def __init__(self, 
                 lr_dir: str,
                 hr_dir: str,
                 patch_size: int = 128,
                 scale_factor: int = 4,
                 split: str = 'train',
                 augment: bool = True):
        """
        Dataset for satellite image super-resolution
        Args:
            lr_dir: Directory containing LR image pairs
            hr_dir: Directory containing HR images
            patch_size: Size of training patches
            scale_factor: Super-resolution scale factor
            split: 'train', 'val', or 'test'
            augment: Whether to apply augmentation
        """
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.split = split
        self.augment = augment and split == 'train'
        
        # Get image pairs
        self.lr_pairs = self._get_image_pairs()
        self.preprocessor = SatelliteImagePreprocessor()
        
        # Setup augmentations
        self.transform = self._setup_augmentations()
        
    def _get_image_pairs(self) -> List[Tuple[Path, Path, Path]]:
        """Get list of LR image pairs and corresponding HR image"""
        lr_pairs = []
        for lr1_path in sorted(self.lr_dir.glob('*_1.*')):
            # Find matching second LR image
            lr2_path = lr1_path.parent / lr1_path.name.replace('_1.', '_2.')
            if not lr2_path.exists():
                continue
                
            # Find corresponding HR image
            hr_name = lr1_path.name.replace('_1.', '.')
            hr_path = self.hr_dir / hr_name
            if not hr_path.exists():
                continue
                
            lr_pairs.append((lr1_path, lr2_path, hr_path))
            
        return lr_pairs
    
    def _setup_augmentations(self) -> A.Compose:
        """Setup augmentation pipeline"""
        if not self.augment:
            return A.Compose([
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
        return A.Compose([
            # Spatial augmentations
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            
            # Color augmentations
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5),
                A.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5)
            ], p=0.5),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(0, 25)),
                A.GaussianBlur(blur_limit=(3, 5)),
                A.MotionBlur(blur_limit=(3, 5))
            ], p=0.3),
            
            # Weather simulation
            A.OneOf([
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3),
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, 
                                num_flare_circles_lower=6, num_flare_circles_upper=10,
                                src_radius=400, src_color=(255, 255, 255))
            ], p=0.2),
            
            # Normalize
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def _load_image(self, path: Path) -> np.ndarray:
        """Load image from path"""
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
        
    def _random_crop(self, lr1: np.ndarray, lr2: np.ndarray, hr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Random crop LR and HR images"""
        if self.split != 'train':
            return lr1, lr2, hr
            
        h, w = lr1.shape[:2]
        lr_patch_size = self.patch_size // self.scale_factor
        
        # Random crop coordinates
        lr_x = random.randint(0, w - lr_patch_size)
        lr_y = random.randint(0, h - lr_patch_size)
        hr_x = lr_x * self.scale_factor
        hr_y = lr_y * self.scale_factor
        
        # Crop patches
        lr1_patch = lr1[lr_y:lr_y+lr_patch_size, lr_x:lr_x+lr_patch_size]
        lr2_patch = lr2[lr_y:lr_y+lr_patch_size, lr_x:lr_x+lr_patch_size]
        hr_patch = hr[hr_y:hr_y+self.patch_size, hr_x:hr_x+self.patch_size]
        
        return lr1_patch, lr2_patch, hr_patch
    
    def _augment(self, lr1: np.ndarray, lr2: np.ndarray, hr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply augmentations to image triplet"""
        if not self.augment:
            return self.transform(image=lr1)['image'], self.transform(image=lr2)['image'], \
                   self.transform(image=hr)['image']
        
        # Apply same spatial transforms to all images
        spatial_transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5)
        ])
        transformed = spatial_transform(image=lr1, additional_targets={'image1': 'image', 'image2': 'image'})
        lr1, lr2 = transformed['image'], transformed['image1']
        hr = transformed['image2']
        
        # Apply color and noise augmentations independently
        lr1 = self.transform(image=lr1)['image']
        lr2 = self.transform(image=lr2)['image']
        hr = self.transform(image=hr)['image']
        
        return lr1, lr2, hr
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a training sample"""
        lr1_path, lr2_path, hr_path = self.lr_pairs[idx]
        
        # Load images
        lr1 = self._load_image(lr1_path)
        lr2 = self._load_image(lr2_path)
        hr = self._load_image(hr_path)
        
        # Random crop
        lr1, lr2, hr = self._random_crop(lr1, lr2, hr)
        
        # Apply augmentations
        lr1, lr2, hr = self._augment(lr1, lr2, hr)
        
        # Convert to tensor
        lr1 = torch.from_numpy(lr1).permute(2, 0, 1).float()
        lr2 = torch.from_numpy(lr2).permute(2, 0, 1).float()
        hr = torch.from_numpy(hr).permute(2, 0, 1).float()
        
        # Stack LR images
        lr_pair = torch.stack([lr1, lr2], dim=0)
        
        return lr_pair, hr
    
    def __len__(self) -> int:
        return len(self.lr_pairs)
    
def create_dataloaders(config: dict):
    """
    Create train and validation dataloaders
    Args:
        config: Configuration dictionary containing dataset parameters
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = SatelliteDataset(
        lr_dir=config['train_lr_dir'],
        hr_dir=config['train_hr_dir'],
        patch_size=config['patch_size'],
        scale_factor=config['scale_factor'],
        split='train',
        augment=True
    )
    
    val_dataset = SatelliteDataset(
        lr_dir=config['val_lr_dir'],
        hr_dir=config['val_hr_dir'],
        patch_size=config['patch_size'],
        scale_factor=config['scale_factor'],
        split='val',
        augment=False
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader 