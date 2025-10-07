"""
Carbon Tracker for Real-time CO2 Emission Monitoring
"""

import time
import psutil
import platform
from typing import Dict, List, Optional
import json
from datetime import datetime
import threading
import numpy as np

try:
    import pynvml
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False


class CarbonTracker:
    """
    Real-time carbon footprint tracking for deep learning training
    """
    
    def __init__(self, 
                 project_name: str = "GreenCNN",
                 country_iso_code: str = "USA",
                 region: Optional[str] = None,
                 tracking_mode: str = "machine"):
        
        self.project_name = project_name
        self.country_iso_code = country_iso_code
        self.region = region
        self.tracking_mode = tracking_mode
        
        # Initialize tracking variables
        self.emissions_data = []
        self.energy_data = []
        self.performance_data = []
        self.start_time = None
        self.is_tracking = False
        
        # Hardware info
        self.cpu_count = psutil.cpu_count()
        self.memory_total = psutil.virtual_memory().total / (1024**3)  # GB
        
        # GPU info
        self.gpu_info = self._get_gpu_info()
        
        # Carbon intensity factors (kg CO2/kWh) by region
        self.carbon_intensity = {
            'USA': 0.4,
            'EU': 0.3,
            'CHINA': 0.6,
            'INDIA': 0.8,
            'GLOBAL': 0.5
        }
        
        # Initialize external tracker if available
        self.external_tracker = None
        if CODECARBON_AVAILABLE:
            try:
                self.external_tracker = EmissionsTracker(
                    project_name=project_name,
                    country_iso_code=country_iso_code,
                    region=region,
                    tracking_mode=tracking_mode
                )
            except Exception as e:
                print(f"Warning: Could not initialize CodeCarbon tracker: {e}")
        
        # Monitoring thread
        self.monitor_thread = None
        self.stop_monitoring = threading.Event()
    
    def _get_gpu_info(self) -> Dict:
        """Get GPU information"""
        gpu_info = {'count': 0, 'names': [], 'memory': []}
        
        if NVIDIA_AVAILABLE:
            try:
                pynvml.nvmlInit()
                gpu_count = pynvml.nvmlDeviceGetCount()
                gpu_info['count'] = gpu_count
                
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    gpu_info['names'].append(name)
                    gpu_info['memory'].append(memory.total / (1024**3))  # GB
                    
            except Exception as e:
                print(f"Warning: Could not get GPU info: {e}")
        
        return gpu_info
    
    def _monitor_resources(self):
        """Background monitoring of system resources"""
        while not self.stop_monitoring.is_set():
            try:
                # CPU and memory usage (non-blocking)
                cpu_percent = psutil.cpu_percent(interval=None)  # Non-blocking
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # GPU usage
                gpu_usage = []
                gpu_memory = []
                gpu_power = []
                
                if NVIDIA_AVAILABLE and self.gpu_info['count'] > 0:
                    try:
                        for i in range(self.gpu_info['count']):
                            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                            
                            # GPU utilization
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            gpu_usage.append(util.gpu)
                            
                            # GPU memory
                            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            gpu_memory.append(mem_info.used / mem_info.total * 100)
                            
                            # GPU power (if available)
                            try:
                                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Watts
                                gpu_power.append(power)
                            except:
                                gpu_power.append(0)
                                
                    except Exception as e:
                        print(f"Warning: GPU monitoring error: {e}")
                
                # Estimate power consumption
                estimated_power = self._estimate_power_consumption(
                    cpu_percent, memory_percent, gpu_power
                )
                
                # Calculate carbon emissions
                carbon_intensity = self.carbon_intensity.get(
                    self.country_iso_code, self.carbon_intensity['GLOBAL']
                )
                
                # Energy in kWh (power in watts * time in hours)
                energy_kwh = estimated_power * (5/3600)  # 5 second interval
                carbon_emissions = energy_kwh * carbon_intensity  # kg CO2
                
                # Store data
                timestamp = time.time()
                self.energy_data.append({
                    'timestamp': timestamp,
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'gpu_usage': gpu_usage,
                    'gpu_memory': gpu_memory,
                    'gpu_power': gpu_power,
                    'estimated_power_watts': estimated_power,
                    'energy_kwh': energy_kwh,
                    'carbon_emissions_kg': carbon_emissions
                })
                
            except Exception as e:
                print(f"Monitoring error: {e}")
            
            time.sleep(5)  # Monitor every 5 seconds instead of 1
    
    def _estimate_power_consumption(self, cpu_percent: float, 
                                  memory_percent: float, 
                                  gpu_power: List[float]) -> float:
        """Estimate total system power consumption"""
        
        # Base system power (idle)
        base_power = 50  # watts
        
        # CPU power estimation (rough approximation)
        cpu_tdp = 65  # Typical desktop CPU TDP
        cpu_power = (cpu_percent / 100) * cpu_tdp
        
        # Memory power estimation
        memory_power = (memory_percent / 100) * 10  # ~10W for memory
        
        # GPU power (actual measurements if available)
        total_gpu_power = sum(gpu_power) if gpu_power else 0
        
        # If no GPU power measurements, estimate based on usage
        if not gpu_power and self.gpu_info['count'] > 0:
            # Rough estimation for GPU power
            estimated_gpu_power = self.gpu_info['count'] * 150  # ~150W per GPU
            total_gpu_power = estimated_gpu_power * 0.5  # Assume 50% usage
        
        total_power = base_power + cpu_power + memory_power + total_gpu_power
        return total_power
    
    def start_tracking(self):
        """Start carbon tracking"""
        if self.is_tracking:
            return
        
        self.start_time = time.time()
        self.is_tracking = True
        self.stop_monitoring.clear()
        
        # Start external tracker
        if self.external_tracker:
            self.external_tracker.start()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        print(f"🌱 Carbon tracking started for {self.project_name}")
    
    def stop_tracking(self) -> Dict:
        """Stop tracking and return summary"""
        if not self.is_tracking:
            return {}
        
        self.is_tracking = False
        self.stop_monitoring.set()
        
        # Stop external tracker
        external_emissions = 0
        if self.external_tracker:
            try:
                external_emissions = self.external_tracker.stop()
            except Exception as e:
                print(f"Warning: External tracker error: {e}")
        
        # Wait for monitoring thread to finish
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        # Calculate summary
        summary = self._calculate_summary(external_emissions)
        
        print(f"🌱 Carbon tracking stopped. Total emissions: {summary['total_emissions_kg']:.6f} kg CO2")
        
        return summary
    
    def _calculate_summary(self, external_emissions: float = 0) -> Dict:
        """Calculate tracking summary"""
        if not self.energy_data:
            return {'total_emissions_kg': 0, 'total_energy_kwh': 0}
        
        total_energy = sum(d['energy_kwh'] for d in self.energy_data)
        total_emissions = sum(d['carbon_emissions_kg'] for d in self.energy_data)
        
        # Use external tracker result if available and higher
        if external_emissions > total_emissions:
            total_emissions = external_emissions
        
        duration = time.time() - self.start_time if self.start_time else 0
        
        avg_power = np.mean([d['estimated_power_watts'] for d in self.energy_data]) if self.energy_data else 0
        
        summary = {
            'total_emissions_kg': total_emissions,
            'total_energy_kwh': total_energy,
            'duration_seconds': duration,
            'avg_power_watts': avg_power,
            'carbon_intensity': self.carbon_intensity.get(
                self.country_iso_code, self.carbon_intensity['GLOBAL']
            ),
            'system_info': {
                'cpu_count': self.cpu_count,
                'memory_gb': self.memory_total,
                'gpu_count': self.gpu_info['count'],
                'gpu_names': self.gpu_info['names']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def get_current_stats(self) -> Dict:
        """Get current tracking statistics"""
        if not self.is_tracking or not self.energy_data:
            return {}
        
        recent_data = self.energy_data[-10:]  # Last 10 seconds
        
        current_stats = {
            'current_power_watts': recent_data[-1]['estimated_power_watts'],
            'avg_power_watts': np.mean([d['estimated_power_watts'] for d in recent_data]),
            'total_emissions_kg': sum(d['carbon_emissions_kg'] for d in self.energy_data),
            'total_energy_kwh': sum(d['energy_kwh'] for d in self.energy_data),
            'duration_seconds': time.time() - self.start_time,
            'emissions_rate_kg_per_hour': sum(d['carbon_emissions_kg'] for d in recent_data) * 360  # Extrapolate to hour
        }
        
        return current_stats
    
    def save_data(self, filepath: str):
        """Save tracking data to file"""
        data = {
            'project_name': self.project_name,
            'summary': self._calculate_summary(),
            'energy_data': self.energy_data,
            'system_info': {
                'cpu_count': self.cpu_count,
                'memory_gb': self.memory_total,
                'gpu_info': self.gpu_info,
                'platform': platform.platform()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Carbon tracking data saved to {filepath}")


class EpochCarbonTracker:
    """Track carbon emissions per training epoch"""
    
    def __init__(self, carbon_tracker: CarbonTracker):
        self.carbon_tracker = carbon_tracker
        self.epoch_data = []
        self.epoch_start_time = None
        self.epoch_start_emissions = 0
    
    def start_epoch(self, epoch: int):
        """Start tracking for an epoch"""
        self.epoch_start_time = time.time()
        current_stats = self.carbon_tracker.get_current_stats()
        self.epoch_start_emissions = current_stats.get('total_emissions_kg', 0)
        
    def end_epoch(self, epoch: int, metrics: Dict):
        """End epoch tracking and record metrics"""
        if self.epoch_start_time is None:
            return
        
        epoch_duration = time.time() - self.epoch_start_time
        current_stats = self.carbon_tracker.get_current_stats()
        epoch_emissions = current_stats.get('total_emissions_kg', 0) - self.epoch_start_emissions
        
        # Calculate carbon efficiency properly
        accuracy = metrics.get('accuracy', 0)
        emissions_grams = max(epoch_emissions * 1000, 0.001)  # Convert to grams, minimum 0.001g
        carbon_efficiency = accuracy / emissions_grams if emissions_grams > 0 else 0
        
        epoch_data = {
            'epoch': epoch,
            'duration_seconds': epoch_duration,
            'emissions_kg': epoch_emissions,
            'emissions_per_sample': epoch_emissions / metrics.get('samples', 1),
            'accuracy': accuracy,
            'loss': metrics.get('loss', 0),
            'carbon_efficiency': carbon_efficiency,  # accuracy per gram CO2
            'timestamp': datetime.now().isoformat()
        }
        
        self.epoch_data.append(epoch_data)
        
        print(f"Epoch {epoch}: {epoch_emissions*1000:.3f}g CO2, "
              f"Efficiency: {epoch_data['carbon_efficiency']:.2f} acc/g")
        
        return epoch_data
    
    def get_efficiency_trend(self) -> List[float]:
        """Get carbon efficiency trend over epochs"""
        return [d['carbon_efficiency'] for d in self.epoch_data]
    
    def get_best_efficiency_epoch(self) -> Dict:
        """Get epoch with best carbon efficiency"""
        if not self.epoch_data:
            return {}
        
        return max(self.epoch_data, key=lambda x: x['carbon_efficiency'])
