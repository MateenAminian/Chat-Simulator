import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from typing import Dict, Any, List
import logging
from pathlib import Path
import sys
from fastapi.middleware.cors import CORSMiddleware
import uuid
import time
import json
import shutil

from .core.video_processor import VideoProcessor
from .core.chat_generator import ChatGenerator
from .core.overlay_generator import OverlayGenerator

# Configure logging at application startup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI()

# Global state
class AppState:
    def __init__(self):
        self.current_video = None
        self.chat_messages = []
        self.overlay_progress = 0

state = AppState()

# Get absolute paths for all directories
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
TEMP_DIR = BASE_DIR / "temp"
UPLOADS_DIR = TEMP_DIR / "uploads"
OUTPUTS_DIR = TEMP_DIR / "outputs"

# Create all required directories
STATIC_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Mount static files with absolute path
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Add these constants at the top
SAFE_PROCESSING_LIMITS = {
    'messages_per_minute': {
        'min': 10,
        'max': 60,
        'default': 30
    },
    'num_chatters': {
        'min': 20,
        'max': 200,
        'default': 100
    },
    'emote_frequency': {
        'min': 0.1,
        'max': 0.8,
        'default': 0.5
    },
    'spam_level': {
        'min': 0.1,
        'max': 0.5,
        'default': 0.2
    }
}

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)
            
    async def send_progress(self, progress: float, step: str, websocket: WebSocket):
        data = {
            "type": "progress",
            "value": progress,
            "step": step
        }
        await websocket.send_text(json.dumps(data))

# Initialize the connection manager
manager = ConnectionManager()

# Add this after defining OUTPUTS_DIR
os.environ["OUTPUTS_DIR"] = str(OUTPUTS_DIR)

# Clear chat cache on startup
CACHE_CHAT_DIR = os.path.join(os.path.dirname(__file__), '..', 'cache', 'chat')
if os.path.exists(CACHE_CHAT_DIR):
    shutil.rmtree(CACHE_CHAT_DIR)
    os.makedirs(CACHE_CHAT_DIR, exist_ok=True)

@app.post("/preview")
async def preview_chat(
    video: UploadFile = File(...),
    messages_per_minute: int = Form(SAFE_PROCESSING_LIMITS['messages_per_minute']['default']),
    num_chatters: int = Form(SAFE_PROCESSING_LIMITS['num_chatters']['default']),
    emote_frequency: float = Form(SAFE_PROCESSING_LIMITS['emote_frequency']['default']),
    spam_level: float = Form(SAFE_PROCESSING_LIMITS['spam_level']['default'])
):
    try:
        logger.info(f"Received video: {video.filename}")
        
        # Save uploaded file
        input_path = f"temp/uploads/{video.filename}"
        with open(input_path, "wb") as buffer:
            content = await video.read()
            buffer.write(content)
        logger.info(f"Saved video to {input_path}")
        
        # Process video with immediate feedback
        logger.info("Starting video processing...")
        processor = VideoProcessor()
        analysis = await processor.analyze_video(input_path)
        logger.info("Video analysis complete")
        
        logger.info("Starting chat generation...")
        chat_generator = ChatGenerator()
        chat_data = await chat_generator.generate_chat(
            analysis,
            {
                'messages_per_minute': messages_per_minute,
                'num_chatters': num_chatters,
                'emote_frequency': emote_frequency,
                'spam_level': spam_level
            }
        )
        logger.info("Chat generation complete")
        
        # Clean up
        os.remove(input_path)
        logger.info("Cleaned up temporary files")
        
        return JSONResponse({
            'success': True,
            'preview': chat_data[:10],
            'total_messages': len(chat_data),
            'video_info': {
                'duration': analysis['duration'],
                'fps': analysis['fps']
            }
        })
        
    except Exception as e:
        logger.error(f"Preview generation failed: {str(e)}")
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post("/process")
async def process_video(
    video: UploadFile = File(...),
    messages_per_minute: int = Form(30),
    num_chatters: int = Form(50),
    emote_frequency: float = Form(0.5),
    position: str = Form("bottom-right")
):
    try:
        # Create uploads directory if it doesn't exist
        uploads_dir = os.path.join("src", "temp", "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Save uploaded file
        temp_path = os.path.join(uploads_dir, video.filename)
        with open(temp_path, "wb") as buffer:
            content = await video.read()
            buffer.write(content)
        
        # Process video
        processor = VideoProcessor()
        analysis = await processor.analyze_video(temp_path)
        
        # Generate chat
        chat_generator = ChatGenerator()
        chat_data = await chat_generator.generate_chat(
            analysis,
            {
                'messages_per_minute': messages_per_minute,
                'num_chatters': num_chatters,
                'emote_frequency': emote_frequency
            }
        )
        
        # Generate overlay
        overlay_generator = OverlayGenerator({
            'position': position
        })
        output_filename = await overlay_generator.generate_combined_video(temp_path, chat_data)
        
        # Clean up
        os.remove(temp_path)
        
        return {
            "success": True,
            "output_path": output_filename
        }
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated overlay video"""
    try:
        # Use the same output directory as the overlay generator
        file_path = os.path.join("src", "temp", "outputs", filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        return FileResponse(
            file_path,
            media_type="video/mp4",
            filename=filename
        )
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add root route to serve index.html
@app.get("/")
async def root():
    """Serve the main page"""
    return FileResponse("src/static/index.html")

@app.post("/generate_chat")
async def generate_chat(
    chatSpeed: int = Form(30),
    chatters: int = Form(50)
):
    """Generate chat messages based on video analysis"""
    try:
        if not state.current_video:
            raise HTTPException(status_code=400, detail="No video uploaded")
            
        # Validate and sanitize input
        msgs_per_minute = max(10, min(60, chatSpeed))
        num_chatters = max(20, min(150, chatters))
        
        logger.info(f"Generating chat with params: chatSpeed={chatSpeed}, chatters={chatters}")
        
        # Generate chat with simple parameters
        chat_params = {
            'messages_per_minute': msgs_per_minute,
            'num_chatters': num_chatters,
            'emote_frequency': 0.3  # Fixed reasonable default
        }
        
        video_processor = VideoProcessor()
        video_analysis = await video_processor.analyze_video(state.current_video)
        
        chat_generator = ChatGenerator()
        messages = await chat_generator.generate_chat(video_analysis, chat_params)
        
        # Store messages for overlay generation
        state.chat_messages = messages
        
        # Return a simple response 
        return {
            "message_count": len(messages),
            "messages": messages,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Failed to generate chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_overlay")
async def generate_overlay():
    """Generate video overlay with chat"""
    try:
        if not state.chat_messages:
            raise HTTPException(status_code=400, detail="No chat messages generated yet")
            
        if not state.current_video:
            raise HTTPException(status_code=400, detail="No video uploaded")
            
        logger.info("Starting overlay generation")
        
        # Generate overlay
        overlay_generator = OverlayGenerator({})
        filename = await overlay_generator.generate_overlay(
            state.current_video,
            state.chat_messages
        )
        
        logger.info(f"Overlay generated: {filename}")
        return {"filename": filename, "status": "success"}
        
    except Exception as e:
        logger.error(f"Failed to generate overlay: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/overlay_progress")
async def get_progress():
    """Get current overlay generation progress"""
    return JSONResponse({
        "progress": state.overlay_progress
    })

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file"""
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs(str(UPLOADS_DIR), exist_ok=True)
        
        # Generate a unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(str(UPLOADS_DIR), unique_filename)
        
        # Save the file
        with open(file_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)
            
        # Store the file path in state
        state.current_video = file_path
        state.chat_messages = []  # Reset chat messages
        
        logger.info(f"Video uploaded: {file_path}")
        return {"filename": unique_filename, "status": "success"}
        
    except Exception as e:
        logger.error(f"Failed to upload video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test_overlay")
async def test_overlay():
    """Generate a test overlay with sample messages"""
    try:
        logger.info("Starting test overlay generation")
        
        # Sample test messages
        test_messages = [
            {
                "timestamp": 0.0,
                "username": "Pogchamp123",
                "message": "Let's go! PogChamp",
                "color": "#FF0000"
            },
            {
                "timestamp": 1.0,
                "username": "KappaKing",
                "message": "This is amazing Kappa",
                "color": "#00FF00"
            },
            {
                "timestamp": 2.0,
                "username": "LULMaster",
                "message": "LULW LULW LULW",
                "color": "#0000FF"
            }
        ]
        
        # Use sample video
        sample_video = os.path.join("static", "sample", "test_video.mp4")
        if not os.path.exists(sample_video):
            # Create a simple test video if it doesn't exist
            logger.info("Creating test video...")
            import cv2
            import numpy as np
            
            os.makedirs(os.path.dirname(sample_video), exist_ok=True)
            out = cv2.VideoWriter(sample_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
            for _ in range(150):  # 5 seconds at 30fps
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[:] = (64, 64, 64)  # Gray background
                out.write(frame)
            out.release()
            
        logger.info(f"Using sample video: {sample_video}")
        
        # Generate overlay
        overlay_generator = OverlayGenerator({})
        filename = await overlay_generator.generate_overlay(
            sample_video,
            test_messages
        )
        
        logger.info(f"Test overlay generated: {filename}")
        return {"filename": filename, "status": "success"}
        
    except Exception as e:
        logger.error(f"Test overlay failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_combined")
async def generate_combined_video(
    position: str = Form("bottom-left")
):
    """Generate video with overlay directly on the video"""
    try:
        if not state.chat_messages:
            raise HTTPException(status_code=400, detail="No chat messages generated yet")
            
        if not state.current_video:
            raise HTTPException(status_code=400, detail="No video uploaded")
        
        # Validate position
        valid_positions = ['bottom-left', 'bottom-right', 'top-left', 'top-right']
        if position not in valid_positions:
            position = 'bottom-left'  # Default fallback
            
        logger.info(f"Starting combined video generation with position: {position}")
        
        # Generate combined video
        overlay_generator = OverlayGenerator({
            'position': position
        })
        
        # Add debug logging to verify the position is correctly set
        logger.info(f"Overlay generator position: {overlay_generator.position}")
        
        filename = await overlay_generator.generate_combined_video(
            state.current_video,
            state.chat_messages
        )
        
        # Rename the file to include position for better identification
        position_name = position.replace('-', '_')
        base_name, ext = os.path.splitext(filename)
        new_filename = f"{base_name}_{position_name}{ext}"
        
        old_path = os.path.join(overlay_generator.output_dir, filename)
        new_path = os.path.join(overlay_generator.output_dir, new_filename)
        
        # Rename file if it exists
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            logger.info(f"Renamed file to include position: {new_filename}")
            filename = new_filename
        
        logger.info(f"Combined video generated: {filename}")
        return {"filename": filename, "status": "success", "position": position}
        
    except Exception as e:
        logger.error(f"Failed to generate combined video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug")
async def debug_info():
    """Return debug information about the application state"""
    return {
        "state": {
            "has_video": state.current_video is not None,
            "video_path": state.current_video if state.current_video else None,
            "chat_message_count": len(state.chat_messages) if state.chat_messages else 0,
        },
        "system": {
            "python_version": sys.version,
            "directories": {
                "uploads_exists": os.path.exists(str(UPLOADS_DIR)),
                "outputs_exists": os.path.exists(str(OUTPUTS_DIR)),
                "static_exists": os.path.exists(str(STATIC_DIR)),
            }
        }
    }

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Global exception handler for the application"""
    error_id = uuid.uuid4().hex
    
    # Log the error with traceback
    logger.error(f"Error ID: {error_id}", exc_info=exc)
    
    # Create user-friendly response
    return JSONResponse(
        status_code=500,
        content={
            "error": "An unexpected error occurred",
            "error_id": error_id,
            "message": str(exc),
            "type": exc.__class__.__name__
        }
    )

@app.middleware("http")
async def request_validation_middleware(request, call_next):
    """Middleware to validate requests and handle common errors"""
    try:
        # Add request ID for tracking
        request_id = str(uuid.uuid4())
        logger.info(f"Request {request_id}: {request.method} {request.url.path}")
        
        # Rate limiting could go here
        
        # Process the request
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        
        logger.info(f"Request {request_id} completed in {duration:.2f}s: {response.status_code}")
        
        return response
    except Exception as e:
        logger.error(f"Request middleware error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(e)}
        )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Process any messages from client if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/files")
async def list_files():
    """List all available files for debugging"""
    static_outputs = []
    temp_outputs = []
    
    # Check static/outputs directory
    static_output_dir = os.path.join("static", "outputs")
    if os.path.exists(static_output_dir):
        static_outputs = os.listdir(static_output_dir)
    
    # Check temp/outputs directory
    if os.path.exists(OUTPUTS_DIR):
        temp_outputs = os.listdir(OUTPUTS_DIR)
    
    return {
        "static_outputs": static_outputs,
        "static_outputs_path": os.path.abspath(static_output_dir),
        "temp_outputs": temp_outputs,
        "temp_outputs_path": os.path.abspath(str(OUTPUTS_DIR))
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)