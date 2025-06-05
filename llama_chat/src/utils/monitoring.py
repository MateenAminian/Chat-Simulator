import logging
import time
import os
import json
from typing import Dict, Any, Optional
import traceback
import uuid
from fastapi import Request, Response

# Replace with actual Sentry SDK if using Sentry
try:
    import sentry_sdk
    from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
    SENTRY_DSN = os.environ.get("SENTRY_DSN")
    if SENTRY_DSN:
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            traces_sample_rate=0.2,
            environment=os.environ.get("ENVIRONMENT", "development")
        )
        HAS_SENTRY = True
    else:
        HAS_SENTRY = False
except ImportError:
    HAS_SENTRY = False

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    def __init__(self):
        self.request_times = {}
        self.endpoint_stats = {}
        
    def record_request_start(self, request_id: str, path: str):
        self.request_times[request_id] = {
            "start": time.time(),
            "path": path
        }
        
    def record_request_end(self, request_id: str, status_code: int):
        if request_id in self.request_times:
            duration = time.time() - self.request_times[request_id]["start"]
            path = self.request_times[request_id]["path"]
            
            if path not in self.endpoint_stats:
                self.endpoint_stats[path] = {
                    "count": 0,
                    "total_time": 0,
                    "success": 0,
                    "error": 0,
                    "requests": []
                }
                
            self.endpoint_stats[path]["count"] += 1
            self.endpoint_stats[path]["total_time"] += duration
            
            if 200 <= status_code < 400:
                self.endpoint_stats[path]["success"] += 1
            else:
                self.endpoint_stats[path]["error"] += 1
                
            # Keep last 100 requests for analysis
            self.endpoint_stats[path]["requests"].append({
                "duration": duration,
                "status": status_code,
                "timestamp": time.time()
            })
            
            if len(self.endpoint_stats[path]["requests"]) > 100:
                self.endpoint_stats[path]["requests"].pop(0)
                
            # Clean up request times dictionary
            del self.request_times[request_id]
            
    def get_endpoint_performance(self, path: Optional[str] = None):
        if path:
            if path in self.endpoint_stats:
                stats = self.endpoint_stats[path]
                return {
                    "path": path,
                    "request_count": stats["count"],
                    "avg_duration": stats["total_time"] / stats["count"] if stats["count"] > 0 else 0,
                    "success_rate": stats["success"] / stats["count"] if stats["count"] > 0 else 0,
                    "error_rate": stats["error"] / stats["count"] if stats["count"] > 0 else 0
                }
            return None
        
        # Return all endpoints
        result = []
        for path, stats in self.endpoint_stats.items():
            result.append({
                "path": path,
                "request_count": stats["count"],
                "avg_duration": stats["total_time"] / stats["count"] if stats["count"] > 0 else 0,
                "success_rate": stats["success"] / stats["count"] if stats["count"] > 0 else 0,
                "error_rate": stats["error"] / stats["count"] if stats["count"] > 0 else 0
            })
        return result

# Global metrics tracker
metrics = PerformanceMetrics()

async def performance_middleware(request: Request, call_next):
    """Middleware to track request performance"""
    request_id = str(uuid.uuid4())
    path = request.url.path
    
    # Start timing
    metrics.record_request_start(request_id, path)
    
    # Process the request
    try:
        response = await call_next(request)
        metrics.record_request_end(request_id, response.status_code)
        return response
    except Exception as e:
        # Log the error
        logger.error(f"Request error: {str(e)}", exc_info=True)
        
        # Report to Sentry if available
        if HAS_SENTRY:
            sentry_sdk.capture_exception(e)
        
        # Record as error
        metrics.record_request_end(request_id, 500)
        
        # Re-raise for the global exception handler
        raise 