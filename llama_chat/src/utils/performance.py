from typing import Dict, Any, Optional
import time
import psutil
import torch
import threading
from collections import defaultdict
import numpy as np
from src.utils.logger import logger

class PerformanceMonitor:
    def __init__(self):
        self.operations = {}
        self.metrics = defaultdict(list)
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread = None
        self.sampling_interval = 1.0  # seconds
        
        # Initialize system info
        self.cpu_count = psutil.cpu_count()
        self.total_memory = psutil.virtual_memory().total
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory
        
        logger.info(f"Performance Monitor initialized with {self.cpu_count} CPUs"
                   f" and GPU: {self.gpu_available}")

    def start_operation(self, name: str):
        """Start timing an operation"""
        with self._lock:
            self.operations[name] = {
                'start_time': time.time(),
                'end_time': None,
                'duration': None
            }
        
        # Start resource monitoring if not already running
        self.start_monitoring()

    def end_operation(self, name: str):
        """End timing an operation"""
        with self._lock:
            if name in self.operations:
                self.operations[name]['end_time'] = time.time()
                self.operations[name]['duration'] = (
                    self.operations[name]['end_time'] - 
                    self.operations[name]['start_time']
                )

    def get_duration(self, name: str) -> Optional[float]:
        """Get duration of an operation"""
        with self._lock:
            if name in self.operations:
                if self.operations[name]['duration'] is not None:
                    return self.operations[name]['duration']
                elif self.operations[name]['start_time'] is not None:
                    # Operation still running
                    return time.time() - self.operations[name]['start_time']
        return None

    def log_metric(self, name: str, value: float):
        """Log a performance metric"""
        with self._lock:
            self.metrics[name].append({
                'timestamp': time.time(),
                'value': value
            })

    def start_monitoring(self):
        """Start continuous resource monitoring"""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_resources
            )
            self._monitor_thread.daemon = True
            self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop resource monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()

    def _monitor_resources(self):
        """Continuously monitor system resources"""
        while self._monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
                self.log_metric('cpu_usage', np.mean(cpu_percent))
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.log_metric('memory_usage', memory.percent)
                self.log_metric('memory_available', memory.available)
                
                # GPU usage if available
                if self.gpu_available:
                    gpu_memory = torch.cuda.memory_allocated(0)
                    gpu_memory_percent = (gpu_memory / self.gpu_memory) * 100
                    self.log_metric('gpu_memory_usage', gpu_memory_percent)
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                self.log_metric('disk_read_bytes', disk_io.read_bytes)
                self.log_metric('disk_write_bytes', disk_io.write_bytes)
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {str(e)}")
                time.sleep(self.sampling_interval)

    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        with self._lock:
            metrics_summary = {}
            
            # Summarize operations
            metrics_summary['operations'] = {
                name: {
                    'duration': self.get_duration(name),
                    'status': 'completed' if op['end_time'] else 'running'
                }
                for name, op in self.operations.items()
            }
            
            # Summarize metrics
            for metric_name, values in self.metrics.items():
                if values:
                    metric_values = [v['value'] for v in values]
                    metrics_summary[metric_name] = {
                        'min': float(np.min(metric_values)),
                        'max': float(np.max(metric_values)),
                        'mean': float(np.mean(metric_values)),
                        'std': float(np.std(metric_values)),
                        'last': float(values[-1]['value']),
                        'samples': len(values)
                    }
            
            # Add system info
            metrics_summary['system'] = {
                'cpu_count': self.cpu_count,
                'total_memory': self.total_memory,
                'gpu_available': self.gpu_available
            }
            
            if self.gpu_available:
                metrics_summary['system']['gpu_info'] = {
                    'name': self.gpu_name,
                    'total_memory': self.gpu_memory
                }
            
            return metrics_summary

    def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        usage = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent
        }
        
        if self.gpu_available:
            gpu_memory = torch.cuda.memory_allocated(0)
            usage['gpu_memory_percent'] = (gpu_memory / self.gpu_memory) * 100
            
        return usage

    def check_resources(self) -> bool:
        """Check if system has enough resources"""
        try:
            usage = self.get_resource_usage()
            
            # Define thresholds
            thresholds = {
                'cpu_percent': 90,
                'memory_percent': 90,
                'disk_percent': 95,
                'gpu_memory_percent': 90
            }
            
            # Check each resource
            for resource, value in usage.items():
                if value > thresholds.get(resource, 90):
                    logger.warning(f"High {resource}: {value}%")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Resource check error: {str(e)}")
            return True  # Continue on error

    def __enter__(self):
        """Context manager entry"""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_monitoring() 