from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import numpy as np
import cv2
import logging
import time
import os
from typing import Dict, List, Any
from PIL import Image
import base64
from io import BytesIO
from moviepy.editor import VideoFileClip
import whisper
import asyncio
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Handles video and audio analysis for chat overlay generation.
    Extracts frames, transcribes audio, and provides scene/visual context.
    Optimized for stability and resource usage.
    """
    def __init__(self) -> None:
        """Initialize video processor, models, and signal handlers."""
        self.device = "cpu"
        logger.info(f"Using device: {self.device}")
        self.model = whisper.load_model("tiny", device=self.device)
        logger.info("VideoProcessor initialized with Whisper")
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.frame_sampling = {
            'short_form': {'interval': 5, 'early_frames': 3.0},
            'long_form': {'interval': 15, 'early_frames': 10.0}
        }
        self.batch_size = 2
        self.max_workers = 1
        try:
            from transformers import pipeline
            self.vision_model = pipeline(
                "image-text-to-text", 
                model="Salesforce/blip-image-captioning-base",
                device="cpu"
            )
        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")
            self.vision_model = None

    async def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze video content: extract frames, transcribe audio, and return analysis dict.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            video_type = 'short_form' if duration < 60 else 'long_form'
            sampling = self.frame_sampling[video_type]
            logger.info(f"Video type detected: {video_type}, using sampling: {sampling}")
            logger.info("Processing audio...")
            audio_analysis = await self._process_audio(video_path)
            logger.info("Processing frames...")
            frame_analysis = await self._process_frames_parallel(
                cap, 
                total_frames, 
                sampling['interval'],
                sampling['early_frames']
            )
            cap.release()
            return {
                'duration': duration,
                'fps': fps,
                'frame_analysis': frame_analysis,
                'transcription': audio_analysis
            }
        except Exception as e:
            logger.error(f"Video analysis failed: {str(e)}")
            raise

    async def _process_frames_parallel(self, cap, total_frames: int, interval: int, early_frames: float) -> List[Dict[str, Any]]:
        """
        Process video frames in parallel batches with reduced load for efficiency.
        Returns a list of frame analysis dicts.
        """
        frames = []
        frame_batches = []
        current_batch = []
        frames_to_process = []
        for i in range(0, total_frames, interval):
            if i < early_frames * cap.get(cv2.CAP_PROP_FPS):
                frames_to_process.append(i)
            else:
                frames_to_process.append(i)
        for frame_idx in frames_to_process:
            current_batch.append(frame_idx)
            if len(current_batch) >= self.batch_size:
                frame_batches.append(current_batch)
                current_batch = []
        if current_batch:
            frame_batches.append(current_batch)
        async def process_batch(batch):
            batch_frames = []
            for frame_idx in batch:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (320, 240))
                    timestamp = frame_idx / cap.get(cv2.CAP_PROP_FPS)
                    visual_analysis = self._analyze_frame(frame)
                    batch_frames.append({
                        'frame_idx': frame_idx,
                        'timestamp': timestamp,
                        'visual_analysis': visual_analysis
                    })
            return batch_frames
        semaphore = asyncio.Semaphore(self.max_workers)
        async def process_batch_with_semaphore(batch):
            async with semaphore:
                return await process_batch(batch)
        tasks = [process_batch_with_semaphore(batch) for batch in frame_batches]
        results = await asyncio.gather(*tasks)
        for batch_result in results:
            frames.extend(batch_result)
        return sorted(frames, key=lambda x: x['timestamp'])

    def _analyze_frame(self, frame: np.ndarray) -> str:
        """
        Analyze a single frame for basic scene context (brightness, etc).
        Returns a string description.
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (80, 60))
            brightness = np.mean(small)
            if brightness < 50:
                return "dark scene"
            elif brightness > 200:
                return "bright scene"
            else:
                return "normal scene"
        except Exception as e:
            logger.error(f"Frame analysis failed: {str(e)}")
            return "unknown scene"

    async def _process_audio(self, video_path: str) -> Dict[str, Any]:
        """
        Transcribe audio from video using Whisper.
        Returns a dict with 'text' and 'segments'.
        """
        try:
            result = self.model.transcribe(
                video_path,
                language="en",
                task="transcribe",
                fp16=False,
                verbose=False
            )
            return {
                'text': result['text'],
                'segments': result['segments']
            }
        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}")
            return {'text': '', 'segments': []}

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        logger.info("Received stop signal, stopping analysis...")
        sys.exit(0)

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """
        Convert a CV2 frame to a base64-encoded JPEG string for preview or debugging.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        max_size = 512
        ratio = max_size / max(image.size)
        new_size = tuple([int(x * ratio) for x in image.size])
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _process_audio_chunk(self, video_clip: VideoFileClip, start_time: float, end_time: float) -> List[Dict]:
        """Process a chunk of audio"""
        try:
            if video_clip.audio is None:
                return []

            audio_segment = video_clip.audio.subclip(start_time, end_time)
            frames = np.array(list(audio_segment.iter_frames()), dtype=np.float32)
            
            if frames.size == 0:
                return []

            result = self.model.transcribe(frames)
            
            audio_results = []
            for segment in result.get('segments', []):
                audio_results.append({
                    'timestamp': start_time + segment.get('start', 0),
                    'transcription': {
                        'text': segment.get('text', ''),
                        'segments': [segment]
                    }
                })
            
            return audio_results
            
        except Exception as e:
            logger.error(f"Audio chunk processing failed: {e}")
            return []

    def _extract_audio_segment(self, video_path: str, start_time: float, end_time: float) -> np.ndarray:
        """Extract audio segment from video"""
        try:
            video = VideoFileClip(video_path)
            if video.audio is None:
                return np.array([])

            audio_segment = video.audio.subclip(start_time, end_time)
            frames = list(audio_segment.iter_frames())
            
            # Convert to numpy array with correct dtype
            audio_data = np.array(frames, dtype=np.float32)
            
            # Close video to free resources
            video.close()
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Failed to extract audio: {str(e)}")
            return np.array([])

    def _transcribe_audio(self, audio_segment: np.ndarray) -> Dict:
        """Transcribe audio segment using Whisper"""
        try:
            # Convert to mono if needed
            if len(audio_segment.shape) > 1:
                audio_mono = audio_segment.mean(axis=1)
            else:
                audio_mono = audio_segment
            
            # Ensure correct dtype and normalize
            audio_mono = audio_mono.astype(np.float32)
            if np.any(audio_mono):  # Check if array is not all zeros
                audio_mono = audio_mono / np.max(np.abs(audio_mono))
            
            # Use whisper model directly
            result = self.model.transcribe(audio_mono)
            
            return {
                'text': result['text'],
                'segments': result.get('segments', [])
            }
            
        except Exception as e:
            logger.error(f"Audio transcription failed: {str(e)}")
            return {'text': '', 'segments': []} 