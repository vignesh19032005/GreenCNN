"""
GreenCNN: Carbon-Efficient Deep Vision Model
"""

from .model import GreenCNN, create_green_cnn
from .carbon_tracker import CarbonTracker, EpochCarbonTracker
from .green_optimizer import GreenOptimizer, EnergyAwareScheduler, GradientClippingGreen
from .trainer import GreenTrainer

__version__ = "0.1.0"
__author__ = "GreenCNN Team"

__all__ = [
    "GreenCNN",
    "create_green_cnn", 
    "CarbonTracker",
    "EpochCarbonTracker",
    "GreenOptimizer",
    "EnergyAwareScheduler", 
    "GradientClippingGreen",
    "GreenTrainer"
]