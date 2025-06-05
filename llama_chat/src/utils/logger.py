import logging
import sys
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
from pathlib import Path
import threading
from typing import Optional

class CustomLogger:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.logger = logging.getLogger('LlamaChat')
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        self.logs_dir = Path('logs')
        self.logs_dir.mkdir(exist_ok=True)
        
        # Create handlers
        self._setup_handlers()
        
        # Track performance related logs
        self.performance_logs = []

    def _setup_handlers(self):
        """Setup file and console handlers with proper formatting"""
        # File handler with rotation
        log_file = self.logs_dir / f"llama_chat_{datetime.now():%Y%m%d}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create formatters and add it to the handlers
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s - %(message)s'
            ' [%(filename)s:%(lineno)d]'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s'
        )
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def debug(self, message: str, *args, **kwargs):
        """Log debug message with performance tracking"""
        self.logger.debug(message, *args, **kwargs)
        if 'performance' in kwargs:
            self._track_performance(message, 'debug', kwargs['performance'])

    def info(self, message: str, *args, **kwargs):
        """Log info message with performance tracking"""
        self.logger.info(message, *args, **kwargs)
        if 'performance' in kwargs:
            self._track_performance(message, 'info', kwargs['performance'])

    def warning(self, message: str, *args, **kwargs):
        """Log warning message with performance tracking"""
        self.logger.warning(message, *args, **kwargs)
        if 'performance' in kwargs:
            self._track_performance(message, 'warning', kwargs['performance'])

    def error(self, message: str, *args, **kwargs):
        """Log error message with performance tracking"""
        self.logger.error(message, *args, **kwargs)
        if 'performance' in kwargs:
            self._track_performance(message, 'error', kwargs['performance'])

    def critical(self, message: str, *args, **kwargs):
        """Log critical message with performance tracking"""
        self.logger.critical(message, *args, **kwargs)
        if 'performance' in kwargs:
            self._track_performance(message, 'critical', kwargs['performance'])

    def _track_performance(self, message: str, level: str, performance_data: dict):
        """Track performance-related log entries"""
        self.performance_logs.append({
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'performance': performance_data
        })

    def get_performance_logs(self):
        """Get all performance-related logs"""
        return self.performance_logs

    def clear_performance_logs(self):
        """Clear performance logs"""
        self.performance_logs = []

# Global logger instance
logger = CustomLogger() 