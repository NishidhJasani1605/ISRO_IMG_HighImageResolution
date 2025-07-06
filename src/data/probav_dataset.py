import torch
from torch.utils.data import Dataset
import numpy as np
import os
from pathlib import Path
import cv2
import pandas as pd
from typing import Tuple, List, Optional
from .image_preprocessor import SatelliteImagePreprocessor
from .enhanced_preprocessor import EnhancedSatellitePreprocessor

class ProbaVDataset(Dataset):
    def __init__(self, 
                 root_dir: str,
                 split: str = 'train',
                 band: str = 'RED',
                 patch_size: int = 128,
                 scale_factor: int = 3,
                 augment: bool = True):
        """
        Dataset for PROBA-V super-resolution
        Args:
            root_dir: Root directory containing the data
            split: 'train' or 'test'
            band: 'RED' or 'NIR'
            patch_size: Size of training patches
            scale_factor: Super-resolution scale factor
            augment: Whether to apply augmentation
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.band = band
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.augment = augment and split == 'train'
        
        # Load normalization parameters
        self.norm_params = pd.read_csv(self.root_dir / 'norm.csv')
        
        # Setup data paths
        self.data_dir = self.root_dir / split / band
        self.preprocessor = EnhancedSatellitePreprocessor()
        
        # Get all image sets
        self.image_sets = self._get_image_sets()
        print(f"Found {len(self.image_sets)} image sets in {self.data_dir}")
        
    def _get_image_sets(self) -> List[str]:
        """Get list of all image set directories"""
        image_sets = []
        for img_dir in self.data_dir.glob('imgset*'):
            if img_dir.is_dir():
                # Check if directory contains both LR and HR data
                if (img_dir / 'HR.png').exists() and list(img_dir.glob('LR*.png')):
                    image_sets.append(img_dir.name)
        return sorted(image_sets)
    
    def _load_png_data(self, png_path: Path) -> np.ndarray:
        """Load data from PNG file"""
        img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        return img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    def _normalize_data(self, data: np.ndarray, img_id: str) -> np.ndarray:
        """Normalize data using parameters from norm.csv"""
        params = self.norm_params[self.norm_params['img_id'] == img_id].iloc[0]
        data = (data - params['mean']) / params['std']
        return data
    
    def _get_item_from_set(self, img_set: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get LR and HR data for a specific image set"""
        img_dir = self.data_dir / img_set
        
        # Load HR data
        hr_file = img_dir / 'HR.png'
        hr_data = self._load_png_data(hr_file)
        
        # Load LR data (select two best quality images)
        lr_files = sorted(img_dir.glob('LR*.png'))
        lr_data = []
        quality_masks = []
        
        for lr_file in lr_files[:2]:  # Take first two LR images
            lr_img = self._load_png_data(lr_file)
            qm_file = img_dir / f'QM{lr_file.stem[2:]}.png'
            qm = self._load_png_data(qm_file) if qm_file.exists() else np.ones_like(lr_img)
            
            lr_data.append(lr_img)
            quality_masks.append(qm)
        
        # Normalize data
        lr1 = self._normalize_data(lr_data[0], img_set)
        lr2 = self._normalize_data(lr_data[1], img_set)
        hr_data = self._normalize_data(hr_data, img_set)
        
        # Register LR images using enhanced preprocessing
        registered_lr1, registered_lr2, quality_mask = self.preprocessor.enhance_image_pair(lr1, lr2)
        
        # Convert to torch tensors
        lr1_tensor = torch.from_numpy(registered_lr1).unsqueeze(0).float()
        lr2_tensor = torch.from_numpy(registered_lr2).unsqueeze(0).float()
        hr_tensor = torch.from_numpy(hr_data).unsqueeze(0).float()
        quality_mask_tensor = torch.from_numpy(quality_mask).unsqueeze(0).float()
        
        return (lr1_tensor, lr2_tensor, quality_mask_tensor), hr_tensor
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a training sample"""
        img_set = self.image_sets[idx]
        return self._get_item_from_set(img_set)
    
    def __len__(self) -> int:
        return len(self.image_sets)

def create_probav_dataloaders(config: dict):
    """
    Create train and validation dataloaders for PROBA-V data
    Args:
        config: Configuration dictionary
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = ProbaVDataset(
        root_dir=config['data']['root_dir'],
        split='train',
        band=config['data']['spectral_band'],
        patch_size=config['data']['patch_size'],
        scale_factor=config['data']['scale_factor'],
        augment=True
    )
    
    val_dataset = ProbaVDataset(
        root_dir=config['data']['root_dir'],
        split='test',
        band=config['data']['spectral_band'],
        patch_size=config['data']['patch_size'],
        scale_factor=config['data']['scale_factor'],
        augment=False
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader 