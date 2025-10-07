"""
Green Optimizer: Carbon-aware optimization strategies
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
import math
from .carbon_tracker import CarbonTracker


class GreenOptimizer:
    """
    Carbon-aware optimizer that adjusts training parameters based on energy efficiency
    """
    
    def __init__(self, 
                 model: nn.Module,
                 base_optimizer: str = "adam",
                 base_lr: float = 0.001,
                 carbon_tracker: Optional[CarbonTracker] = None,
                 efficiency_threshold: float = 0.1,
                 adaptation_rate: float = 0.1):
        
        self.model = model
        self.base_lr = base_lr
        self.carbon_tracker = carbon_tracker
        self.efficiency_threshold = efficiency_threshold
        self.adaptation_rate = adaptation_rate
        
        # Initialize base optimizer
        if base_optimizer.lower() == "adam":
            self.optimizer = optim.Adam(model.parameters(), lr=base_lr)
        elif base_optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9)
        elif base_optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(model.parameters(), lr=base_lr)
        else:
            raise ValueError(f"Unsupported optimizer: {base_optimizer}")
        
        # Green optimization state
        self.efficiency_history = []
        self.carbon_history = []
        self.lr_history = []
        self.current_lr = base_lr
        
        # Dynamic scaling factors
        self.lr_scale_factor = 1.0
        self.batch_scale_factor = 1.0
        
    def calculate_carbon_efficiency(self, accuracy: float, carbon_emissions: float) -> float:
        """Calculate carbon efficiency metric (accuracy per gram CO2)"""
        if carbon_emissions <= 0:
            return float('inf')
        return accuracy / (carbon_emissions * 1000)  # accuracy per gram CO2
    
    def adapt_learning_rate(self, current_efficiency: float, target_efficiency: float):
        """Adapt learning rate based on carbon efficiency"""
        if len(self.efficiency_history) < 2:
            return
        
        efficiency_trend = np.mean(self.efficiency_history[-3:]) - np.mean(self.efficiency_history[-6:-3]) if len(self.efficiency_history) >= 6 else 0
        
        if current_efficiency < target_efficiency:
            # Efficiency is low, reduce learning rate to be more conservative
            if efficiency_trend < 0:  # Efficiency is decreasing
                self.lr_scale_factor *= (1 - self.adaptation_rate)
            else:
                self.lr_scale_factor *= (1 - self.adaptation_rate * 0.5)
        else:
            # Efficiency is good, can be more aggressive
            if efficiency_trend > 0:  # Efficiency is increasing
                self.lr_scale_factor *= (1 + self.adaptation_rate * 0.5)
        
        # Clamp scaling factor
        self.lr_scale_factor = np.clip(self.lr_scale_factor, 0.1, 2.0)
        
        # Update learning rate
        new_lr = self.base_lr * self.lr_scale_factor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        self.current_lr = new_lr
    
    def step(self, accuracy: float, carbon_emissions: float):
        """Perform optimization step with carbon awareness"""
        # Calculate current efficiency
        current_efficiency = self.calculate_carbon_efficiency(accuracy, carbon_emissions)
        
        # Update history
        self.efficiency_history.append(current_efficiency)
        self.carbon_history.append(carbon_emissions)
        self.lr_history.append(self.current_lr)
        
        # Keep only recent history
        if len(self.efficiency_history) > 100:
            self.efficiency_history = self.efficiency_history[-100:]
            self.carbon_history = self.carbon_history[-100:]
            self.lr_history = self.lr_history[-100:]
        
        # Adapt learning rate based on efficiency
        if len(self.efficiency_history) > 1:
            target_efficiency = np.mean(self.efficiency_history) * 1.1  # Target 10% improvement
            self.adapt_learning_rate(current_efficiency, target_efficiency)
        
        # Perform actual optimization step
        self.optimizer.step()
    
    def zero_grad(self):
        """Zero gradients"""
        self.optimizer.zero_grad()
    
    def get_stats(self) -> Dict:
        """Get current optimization statistics"""
        if not self.efficiency_history:
            return {}
        
        return {
            'current_lr': self.current_lr,
            'lr_scale_factor': self.lr_scale_factor,
            'current_efficiency': self.efficiency_history[-1] if self.efficiency_history else 0,
            'avg_efficiency': np.mean(self.efficiency_history),
            'efficiency_trend': np.mean(self.efficiency_history[-5:]) - np.mean(self.efficiency_history[-10:-5]) if len(self.efficiency_history) >= 10 else 0,
            'total_carbon': sum(self.carbon_history),
            'avg_carbon_per_step': np.mean(self.carbon_history) if self.carbon_history else 0
        }


class EnergyAwareScheduler:
    """Learning rate scheduler that considers energy consumption"""
    
    def __init__(self, optimizer: GreenOptimizer, 
                 patience: int = 10,
                 factor: float = 0.5,
                 min_lr: float = 1e-6,
                 energy_weight: float = 0.3):
        
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.energy_weight = energy_weight
        
        self.best_efficiency = 0
        self.wait = 0
        
    def step(self, current_efficiency: float):
        """Step the scheduler based on carbon efficiency"""
        if current_efficiency > self.best_efficiency:
            self.best_efficiency = current_efficiency
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            # Reduce learning rate
            current_lr = self.optimizer.current_lr
            new_lr = max(current_lr * self.factor, self.min_lr)
            
            for param_group in self.optimizer.optimizer.param_groups:
                param_group['lr'] = new_lr
            
            self.optimizer.current_lr = new_lr
            self.wait = 0
            
            return True  # LR was reduced
        
        return False


class GradientClippingGreen:
    """Green gradient clipping that adapts based on energy consumption"""
    
    def __init__(self, max_norm: float = 1.0, 
                 adaptive: bool = True,
                 energy_factor: float = 0.1):
        
        self.base_max_norm = max_norm
        self.adaptive = adaptive
        self.energy_factor = energy_factor
        self.current_max_norm = max_norm
        
    def clip_gradients(self, model: nn.Module, carbon_emissions: float):
        """Clip gradients with energy-aware adaptation"""
        if self.adaptive:
            # Adapt clipping based on carbon emissions
            # Higher emissions -> more aggressive clipping
            emission_factor = 1 + (carbon_emissions * 1000 * self.energy_factor)  # Convert to grams
            self.current_max_norm = self.base_max_norm / emission_factor
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.current_max_norm)
        
        return self.current_max_norm