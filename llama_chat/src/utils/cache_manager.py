import os
import json
import hashlib
import time
from typing import Dict, Any, Optional
import logging
from src.utils.logger import logger
from src.utils.config import config

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, enable_caching=True):
        self.enable_caching = enable_caching
        self.cache_dir = "cache"
        
        if self.enable_caching:
            os.makedirs(os.path.join(self.cache_dir, "analysis"), exist_ok=True)
            os.makedirs(os.path.join(self.cache_dir, "chat"), exist_ok=True)
            os.makedirs(os.path.join(self.cache_dir, "overlay"), exist_ok=True)
            
        # In-memory LRU cache for frequently accessed items
        self.memory_cache = {}
        self.max_cache_size = 10
        
    def _get_cache_path(self, cache_type: str, key: str) -> str:
        """Get path for cache file"""
        return os.path.join(self.cache_dir, cache_type, f"{key}.json")
    
    def _generate_key(self, video_path: str, params: Dict = None) -> str:
        """Generate cache key from video path and parameters"""
        if params:
            param_str = json.dumps(params, sort_keys=True)
            return hashlib.md5((video_path + param_str).encode()).hexdigest()
        return hashlib.md5(video_path.encode()).hexdigest()
    
    def get_analysis_cache(self, video_path: str) -> Optional[Dict]:
        """Get cached video analysis"""
        if not self.enable_caching:
            return None
            
        key = self._generate_key(video_path)
        
        # Check memory cache first
        if key in self.memory_cache:
            logger.info(f"Memory cache hit for analysis: {key}")
            return self.memory_cache[key]
            
        cache_path = self._get_cache_path("analysis", key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    # Add to memory cache
                    self._update_memory_cache(key, data)
                    return data
            except Exception as e:
                logger.error(f"Error reading analysis cache: {e}")
        return None
        
    def set_analysis_cache(self, video_path: str, data: Dict) -> None:
        """Cache video analysis results"""
        if not self.enable_caching:
            return
            
        key = self._generate_key(video_path)
        cache_path = self._get_cache_path("analysis", key)
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            # Update memory cache
            self._update_memory_cache(key, data)
        except Exception as e:
            logger.error(f"Error writing analysis cache: {e}")
            
    def _update_memory_cache(self, key: str, data: Any) -> None:
        """Update in-memory LRU cache"""
        self.memory_cache[key] = data
        
        # If cache is too large, remove oldest item
        if len(self.memory_cache) > self.max_cache_size:
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]

    def get_cache(self, key: str) -> Optional[Dict]:
        """Get cached data if it exists and is valid"""
        try:
            cache_path = os.path.join(self.cache_dir, f"{key}.json")
            overlay_dir = os.path.join(self.cache_dir, key)
            
            if not os.path.exists(cache_path):
                return None
                
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
                
            # Check if cache is expired
            if time.time() - cache_data['timestamp'] > self.cache_duration:
                self._clean_cache(key)
                return None
                
            return cache_data['data']
            
        except Exception as e:
            logger.error(f"Cache read error: {str(e)}")
            return None

    def set_cache(self, key: str, data: Dict):
        """Save data to cache"""
        try:
            self._ensure_cache_size()
            
            cache_path = os.path.join(self.cache_dir, f"{key}.json")
            cache_data = {
                'timestamp': time.time(),
                'data': data
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
                
        except Exception as e:
            logger.error(f"Cache write error: {str(e)}")

    def get_overlay_cache(self, key: str) -> Optional[str]:
        """Get cached overlay file path"""
        try:
            cache_path = os.path.join(self.cache_dir, f"{key}_overlay")
            if os.path.exists(cache_path):
                return cache_path
            return None
        except Exception as e:
            logger.error(f"Overlay cache read error: {str(e)}")
            return None

    def set_overlay_cache(self, key: str, file_path: str):
        """Cache an overlay file"""
        try:
            self._ensure_cache_size()
            cache_path = os.path.join(self.cache_dir, f"{key}_overlay")
            os.replace(file_path, cache_path)
        except Exception as e:
            logger.error(f"Overlay cache write error: {str(e)}")

    def _clean_cache(self, key: str):
        """Remove cached data for a key"""
        try:
            cache_path = os.path.join(self.cache_dir, f"{key}.json")
            overlay_dir = os.path.join(self.cache_dir, key)
            
            if os.path.exists(cache_path):
                os.remove(cache_path)
            if os.path.exists(overlay_dir):
                for file in os.listdir(overlay_dir):
                    os.remove(os.path.join(overlay_dir, file))
                os.rmdir(overlay_dir)
                
        except Exception as e:
            logger.error(f"Cache cleanup error: {str(e)}")

    def _ensure_cache_size(self):
        """Ensure cache doesn't exceed max size"""
        try:
            total_size = 0
            cache_files = []
            
            for root, _, files in os.walk(self.cache_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    size = os.path.getsize(file_path)
                    total_size += size
                    cache_files.append((file_path, size, os.path.getmtime(file_path)))
            
            if total_size > self.max_cache_size:
                # Sort by last modified time (oldest first)
                cache_files.sort(key=lambda x: x[2])
                
                # Remove oldest files until under limit
                for file_path, size, _ in cache_files:
                    os.remove(file_path)
                    total_size -= size
                    if total_size <= self.max_cache_size:
                        break
                        
        except Exception as e:
            logger.error(f"Cache size management error: {str(e)}") 