"""
GreenCNN: Carbon-efficient CNN architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class EfficientConvBlock(nn.Module):
    """Efficient convolution block with optional depth scaling"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1,
                 use_depthwise: bool = True, expansion_factor: float = 1.0):
        super().__init__()
        
        self.use_depthwise = use_depthwise
        expanded_channels = int(in_channels * expansion_factor)
        
        if use_depthwise:
            # Depthwise separable convolution for efficiency
            self.conv = nn.Sequential(
                # Pointwise expansion
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.ReLU6(inplace=True),
                
                # Depthwise convolution
                nn.Conv2d(expanded_channels, expanded_channels, kernel_size, 
                         stride=stride, padding=kernel_size//2, 
                         groups=expanded_channels, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.ReLU6(inplace=True),
                
                # Pointwise projection
                nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # Standard convolution
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, 
                         stride=stride, padding=kernel_size//2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        # Skip connection for residual learning
        self.skip_connection = None
        if in_channels != out_channels or stride != 1:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.conv(x)
        
        if self.skip_connection is not None:
            x = self.skip_connection(x)
        
        if x.shape == out.shape:
            out = out + x
        
        return F.relu(out)


class AdaptiveDepthModule(nn.Module):
    """Module that can dynamically adjust its depth based on energy constraints"""
    
    def __init__(self, channels: int, num_blocks: int = 3):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            EfficientConvBlock(channels, channels) for _ in range(num_blocks)
        ])
        
        self.active_blocks = num_blocks
        self.energy_threshold = 0.5  # Threshold for depth reduction
        
    def set_active_blocks(self, num_blocks: int):
        """Dynamically set number of active blocks"""
        self.active_blocks = min(max(1, num_blocks), len(self.blocks))
    
    def forward(self, x):
        for i in range(self.active_blocks):
            x = self.blocks[i](x)
        return x


class GreenCNN(nn.Module):
    """
    Carbon-efficient CNN with dynamic architecture adaptation
    """
    
    def __init__(self, num_classes: int = 10, 
                 input_channels: int = 3,
                 base_channels: int = 32,
                 carbon_aware: bool = True,
                 adaptive_depth: bool = True):
        super().__init__()
        
        self.num_classes = num_classes
        self.carbon_aware = carbon_aware
        self.adaptive_depth = adaptive_depth
        
        # Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Efficient feature blocks
        self.features = nn.ModuleList([
            # Block 1: 32x32 -> 16x16
            nn.Sequential(
                EfficientConvBlock(base_channels, base_channels * 2, stride=2),
                AdaptiveDepthModule(base_channels * 2, 2) if adaptive_depth 
                else EfficientConvBlock(base_channels * 2, base_channels * 2)
            ),
            
            # Block 2: 16x16 -> 8x8
            nn.Sequential(
                EfficientConvBlock(base_channels * 2, base_channels * 4, stride=2),
                AdaptiveDepthModule(base_channels * 4, 3) if adaptive_depth 
                else EfficientConvBlock(base_channels * 4, base_channels * 4)
            ),
            
            # Block 3: 8x8 -> 4x4
            nn.Sequential(
                EfficientConvBlock(base_channels * 4, base_channels * 8, stride=2),
                AdaptiveDepthModule(base_channels * 8, 2) if adaptive_depth 
                else EfficientConvBlock(base_channels * 8, base_channels * 8)
            )
        ])
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(base_channels * 8, num_classes)
        )
        
        # Energy monitoring
        self.energy_budget = 1.0  # Relative energy budget
        self.current_complexity = 1.0
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def adapt_to_energy_budget(self, energy_ratio: float):
        """Adapt model complexity based on energy budget"""
        if not self.adaptive_depth:
            return
        
        self.energy_budget = energy_ratio
        
        # Reduce depth if energy is constrained
        if energy_ratio < 0.7:
            depth_reduction = int((1 - energy_ratio) * 2)
            for feature_block in self.features:
                for module in feature_block:
                    if isinstance(module, AdaptiveDepthModule):
                        current_blocks = len(module.blocks)
                        new_blocks = max(1, current_blocks - depth_reduction)
                        module.set_active_blocks(new_blocks)
        
        # Calculate current complexity
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        base_params = 100000  # Approximate base parameter count
        self.current_complexity = total_params / base_params
    
    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # Feature extraction
        for feature_block in self.features:
            x = feature_block(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
    def get_model_stats(self) -> Dict:
        """Get current model statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Calculate FLOPs (approximate)
        flops = self.estimate_flops()
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'estimated_flops': flops,
            'energy_budget': self.energy_budget,
            'current_complexity': self.current_complexity,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
    
    def estimate_flops(self, input_size: Tuple[int, int, int] = (3, 32, 32)) -> int:
        """Estimate FLOPs for the model"""
        # Simplified FLOP estimation
        flops = 0
        
        # This is a rough estimation - in practice you'd use a library like ptflops
        # For now, return a placeholder based on parameters
        total_params = sum(p.numel() for p in self.parameters())
        flops = total_params * 2  # Rough approximation
        
        return flops


def create_green_cnn(dataset: str = "cifar10", **kwargs) -> GreenCNN:
    """Factory function to create GreenCNN for different datasets"""
    
    if dataset.lower() == "cifar10":
        return GreenCNN(num_classes=10, input_channels=3, **kwargs)
    elif dataset.lower() == "cifar100":
        return GreenCNN(num_classes=100, input_channels=3, **kwargs)
    elif dataset.lower() == "imagenet":
        return GreenCNN(num_classes=1000, input_channels=3, base_channels=64, **kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")