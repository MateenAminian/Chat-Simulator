import cv2
import logging
import time
import os
from typing import Dict, List, Optional
import numpy as np
import re
import random
import copy
import moviepy.editor as mpy

logger = logging.getLogger(__name__)

class OverlayGenerator:
    """
    Generates a Twitch-style chat overlay on video frames, stacking messages from the bottom up.
    Handles message formatting, positioning, and overlay rendering.
    """
    def __init__(self, style: Dict) -> None:
        """Initialize overlay generator with style and position settings."""
        self.style = style
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.45
        self.thickness = 1
        self.message_spacing = 25
        self.line_spacing = 15
        self.max_messages = 10
        self.output_dir = os.path.join("src", "temp", "outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        self.progress = 0
        self.position = style.get('position', 'bottom-left')
        self.position_settings = {
            'bottom-left': {'x_offset': 20, 'y_offset_from_bottom': 40, 'align': 'left'},
            'bottom-right': {'x_offset': -20, 'y_offset_from_bottom': 40, 'align': 'right'},
            'top-left': {'x_offset': 20, 'y_offset_from_top': 40, 'align': 'left'},
            'top-right': {'x_offset': -20, 'y_offset_from_top': 40, 'align': 'right'}
        }
        self.username_colors = [
            '#FF0000', '#0000FF', '#00FF00', '#FF7F50', '#9ACD32'
        ]
        
        # Adjust chat area dimensions based on video type
        self.short_form_settings = {
            'bottom_margin': 40,
            'max_width_ratio': 0.8,  # 80% of video width
            'chat_area_height': 200   # Fixed height for short videos
        }
        
        self.long_form_settings = {
            'bottom_margin': 60,
            'max_width_ratio': 0.4,  # 40% of video width
            'chat_area_height': 300   # More space for long videos
        }
        
    def clean_message(self, message: str) -> str:
        """Clean message text by removing timestamps and extra spaces."""
        message = re.sub(r'^\d+\.\d+s\s*', '', message)
        message = ' '.join(message.split())
        return message.strip()
        
    def wrap_text(self, text: str, max_width: int) -> List[str]:
        """Wrap text to fit within max_width pixels using OpenCV text size."""
        words = text.split()
        lines = []
        current_line = []
        current_width = 0
        
        for word in words:
            word_size = cv2.getTextSize(
                word + " ", 
                self.font, 
                self.font_scale, 
                self.thickness
            )[0][0]
            
            if current_width + word_size > max_width:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_width = word_size
                else:
                    lines.append(word)
                    current_line = []
                    current_width = 0
            else:
                current_line.append(word)
                current_width += word_size
        
        if current_line:
            lines.append(' '.join(current_line))
            
        return lines

    async def generate_overlay(self, video_path: str, messages: List[Dict]) -> str:
        """
        Generate a chat overlay video with messages stacked from the bottom up (Twitch style).
        Returns the output filename.
        """
        start_time = time.time()
        frame_count = 0
        
        logger.info(f"Received {len(messages)} messages to render")
        
        try:
            # Validate and clean timestamps
            for msg in messages:
                try:
                    if 'color' not in msg or not msg['color']:
                        msg['color'] = random.choice(self.username_colors)
                    
                    # Ensure we have a valid message
                    if not msg.get('message'):
                        continue
                        
                    # Use _timestamp for internal timing
                    if '_timestamp' in msg:
                        ts = msg['_timestamp']
                        if isinstance(ts, str):
                            ts = float(ts.rstrip('s'))
                        msg['_timestamp'] = float(ts)
                except (ValueError, AttributeError) as e:
                    logger.error(f"Invalid message format: {msg} - {str(e)}")
                    continue
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
                
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            output_filename = f"chat_overlay_{int(time.time())}.mp4"
            output_path = os.path.join(self.output_dir, output_filename)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            chat_area_height = self.message_spacing * self.max_messages
            chat_area_top = height - chat_area_height - 20
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                current_time = frame_count / fps
                # Get messages for this timestamp
                visible_messages = [
                    msg for msg in messages
                    if msg.get('_timestamp', 0) <= current_time
                ][-self.max_messages:]

                # --- Twitch-style: bottom-up stacking ---
                # Calculate message heights for all visible messages
                message_heights = []
                message_lines_list = []
                username_widths = []
                for msg in visible_messages:
                    username, message_text = self._format_message(msg)
                    username_text = f"{username}: "
                    username_width = cv2.getTextSize(
                        username_text, self.font, self.font_scale, self.thickness
                    )[0][0]
                    max_msg_width = width - username_width - 40
                    message_lines = self.wrap_text(message_text, max_msg_width)
                    message_height = max(1, len(message_lines)) * self.message_spacing
                    message_heights.append(message_height)
                    message_lines_list.append(message_lines)
                    username_widths.append(username_width)
                # Start y_pos at the bottom and move up for each message
                y_pos = height - 25
                # Draw messages from newest (bottom) to oldest (top)
                for idx in range(len(visible_messages)-1, -1, -1):
                    msg = visible_messages[idx]
                    message_lines = message_lines_list[idx]
                    username_width = username_widths[idx]
                    message_height = message_heights[idx]
                    y_pos -= message_height
                    if not msg.get('message'):
                        continue
                    username, message_text = self._format_message(msg)
                    username_text = f"{username}: "
                    # Draw username
                    cv2.putText(
                        frame,
                        username_text,
                        (20, y_pos + self.line_spacing),
                        self.font,
                        self.font_scale,
                        self._hex_to_bgr(msg['color']),
                        self.thickness
                    )
                    # Draw each line of the message
                    for i, line in enumerate(message_lines):
                        line_y = y_pos + ((i + 1) * self.line_spacing)
                        clean_line = ''.join(char for char in line if ord(char) < 128)
                        cv2.putText(
                            frame,
                            clean_line.strip(),
                            (20 + username_width, line_y),
                            self.font,
                            self.font_scale,
                            (255, 255, 255),
                            self.thickness
                        )
                
                out.write(frame)
                frame_count += 1
                self.progress = int((frame_count / total_frames) * 100)
                
                if frame_count % 30 == 0:
                    logger.info(f"Processed {frame_count}/{total_frames} frames ({self.progress}%)")
            
            out.release()
            
            logger.info(f"Overlay generation complete: {frame_count} frames")
            return output_filename
            
        except Exception as e:
            logger.error(f"Overlay generation failed: {str(e)}")
            raise
        finally:
            if 'cap' in locals():
                cap.release()
            if 'out' in locals():
                out.release()
                
    def _hex_to_bgr(self, hex_color: str) -> tuple:
        """Convert hex color to BGR format for OpenCV."""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (b, g, r)

    def _format_message(self, message: Dict) -> tuple:
        """Format message for display in the video, returning username and message separately."""
        username = message['username']
        message_text = message['message']
        return username, message_text

    async def generate_combined_video(self, video_path: str, messages: List[Dict]) -> str:
        """Generate video with chat overlay directly on the video and preserve original audio"""
        start_time = time.time()
        frame_count = 0
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps
            
            # Reduce resolution for processing
            target_width = min(width, 1280)  # Max width of 1280
            target_height = int(height * (target_width / width))
            
            logger.info(f"Video duration: {video_duration:.2f} seconds")
            logger.info(f"Processing at resolution: {target_width}x{target_height}")
            
            processed_messages = self._prepare_message_timestamps(messages, video_duration)
            output_filename = f"combined_video_{int(time.time())}_{self.position}.mp4"
            output_path = os.path.join(self.output_dir, output_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            logger.info(f"Saving combined video to: {output_path}")
            logger.info(f"Using position for overlay: {self.position}")
            
            # Use H.264 codec for better compression and compatibility
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            temp_video_path = output_path + ".noaudio.mp4"
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (target_width, target_height))
            
            # Process frames in smaller chunks to manage memory
            chunk_size = 30  # Process 1 second at a time (assuming 30fps)
            current_chunk = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame to target resolution
                frame = cv2.resize(frame, (target_width, target_height))
                
                # Add overlay
                frame = self._add_overlay_to_frame(frame, processed_messages, frame_count / fps)
                
                out.write(frame)
                frame_count += 1
                
                # Process in chunks and clear memory
                if frame_count % chunk_size == 0:
                    current_chunk += 1
                    logger.info(f"Processed chunk {current_chunk} ({frame_count}/{total_frames} frames)")
                    # Force garbage collection
                    import gc
                    gc.collect()
            
            # Release resources
            cap.release()
            out.release()
            
            # Add audio back to the video
            logger.info("Adding audio back to video...")
            self._add_audio_to_video(video_path, temp_video_path, output_path)
            
            # Clean up temporary file
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            
            processing_time = time.time() - start_time
            logger.info(f"Video processing completed in {processing_time:.2f} seconds")
            
            return output_filename
            
        except Exception as e:
            logger.error(f"Error generating combined video: {str(e)}")
            raise

    def _draw_messages_with_position(self, frame, messages, frame_width, frame_height, message_cache):
        """Draw chat messages with the correct position on the frame, Twitch/YouTube style (block stacking)."""
        position_config = self.position_settings.get(self.position, self.position_settings['bottom-left'])
        margin_x = 20  # left/right margin
        margin_y = 40  # bottom margin
        align = position_config['align']
        is_bottom = 'y_offset_from_bottom' in position_config

        if is_bottom:
            # Calculate total height of all visible messages (including all lines and paddings)
            total_height = 0
            message_heights = []
            for msg in messages:
                if msg['timestamp'] not in message_cache:
                    message_heights.append(0)
                    continue
                cache = message_cache[msg['timestamp']]
                h = cache['message_height'] + 10  # 10px padding
                message_heights.append(h)
                total_height += h
            # Start y so the last message is just above the bottom margin
            y_pos = frame_height - margin_y - total_height
            for idx, msg in enumerate(messages):
                if msg['timestamp'] not in message_cache:
                    continue
                cache = message_cache[msg['timestamp']]
                message_height = cache['message_height']
                if align == 'right':
                    x_pos = frame_width - cache['username_width'] - margin_x
                else:
                    x_pos = margin_x
                username_color = self._hex_to_bgr(cache['color'])
                cv2.putText(
                    frame, f"{cache['username']}: ", (x_pos, y_pos + self.line_spacing),
                    self.font, self.font_scale, username_color, self.thickness
                )
                for i, line in enumerate(cache['message_lines']):
                    cv2.putText(
                        frame, line,
                        (x_pos + cache['username_width'] if i == 0 else x_pos, y_pos + (i + 1) * self.line_spacing),
                        self.font, self.font_scale, (255, 255, 255), self.thickness
                    )
                y_pos += message_height + 10  # move down for next message
                # Stop if we reach the bottom margin (shouldn't happen, but safety)
                if y_pos > frame_height - margin_y:
                    break
        else:
            # For top positions, stack downward from the top margin
            y_pos = margin_y
            for msg in messages:
                if msg['timestamp'] not in message_cache:
                    continue
                cache = message_cache[msg['timestamp']]
                message_height = cache['message_height']
                if align == 'right':
                    x_pos = frame_width - cache['username_width'] - margin_x
                else:
                    x_pos = margin_x
                username_color = self._hex_to_bgr(cache['color'])
                cv2.putText(
                    frame, f"{cache['username']}: ", (x_pos, y_pos + self.line_spacing),
                    self.font, self.font_scale, username_color, self.thickness
                )
                for i, line in enumerate(cache['message_lines']):
                    cv2.putText(
                        frame, line,
                        (x_pos + cache['username_width'] if i == 0 else x_pos, y_pos + (i + 1) * self.line_spacing),
                        self.font, self.font_scale, (255, 255, 255), self.thickness
                    )
                y_pos += message_height + 10  # 10px padding
                if y_pos > frame_height - margin_y:
                    break

    def _draw_message_background(self, frame, x, y, width, height):
        """Draw message background - now completely transparent"""
        # No background implementation - simply don't draw anything
        pass

    def _draw_message(self, frame, msg, x, y, max_width):
        """Draw a single chat message with background"""
        username_text = f"{msg.get('username', '')}: "
        message_text = msg.get('message', '')
        
        # Calculate total width and height
        username_width = cv2.getTextSize(
            username_text, 
            self.font, 
            self.font_scale, 
            self.thickness
        )[0][0]
        
        message_lines = self.wrap_text(message_text, max_width - username_width)
        total_height = len(message_lines) * self.line_spacing
        total_width = max(
            [cv2.getTextSize(line, self.font, self.font_scale, self.thickness)[0][0] 
             for line in message_lines]
        ) + username_width
        
        # Draw background
        self._draw_message_background(frame, x, y, total_width, total_height)
        
        # Draw username and message
        # ... existing text drawing code ... 

    def _calculate_position(self, frame_width, frame_height, message_height, message_width):
        """Calculate the position of the chat overlay based on selected position"""
        # Debug logging to verify what position is being used
        logger.info(f"Using position: {self.position}")
        
        position_config = self.position_settings.get(self.position, self.position_settings['bottom-left'])
        logger.info(f"Position config: {position_config}")
        
        # Handle x-coordinate
        if position_config['align'] == 'right':
            x = frame_width - message_width - abs(position_config['x_offset'])
        else:
            x = position_config['x_offset']
        
        # Handle y-coordinate
        if 'y_offset_from_bottom' in position_config:
            y = frame_height - position_config['y_offset_from_bottom']
        else:
            y = position_config['y_offset_from_top'] + message_height
        
        logger.info(f"Calculated position: x={x}, y={y}")
        return x, y

    def _prepare_message_timestamps(self, messages: List[Dict], video_duration: float) -> List[Dict]:
        """Ensure messages have proper timestamps for gradual appearance"""
        # If the messages don't have timestamps, or they're all bunched together,
        # redistribute them across the video duration
        
        # Sort messages by timestamp if they exist
        if all('timestamp' in msg for msg in messages):
            # Check if timestamps are too bunched together
            timestamps = [msg['timestamp'] for msg in messages]
            time_span = max(timestamps) - min(timestamps)
            
            if time_span < (video_duration * 0.5):  # If messages use less than half the video
                logger.info(f"Redistributing timestamps across video duration")
                # Need to redistribute
                need_redistribution = True
            else:
                # Timestamps are already well distributed
                need_redistribution = False
        else:
            # No timestamps, need to create them
            need_redistribution = True
        
        # If we need to redistribute timestamps
        if need_redistribution:
            # Create a copy of the messages to avoid modifying the originals
            processed_messages = copy.deepcopy(messages)
            
            # Distribute messages across 85% of the video duration
            # (leaving some space at the beginning and end)
            effective_duration = video_duration * 0.85
            start_offset = video_duration * 0.05  # Start after 5% of the video
            
            # Calculate time between messages
            message_count = len(processed_messages)
            if message_count > 1:
                time_step = effective_duration / (message_count - 1)
            else:
                time_step = 0
            
            # Assign new timestamps
            for i, msg in enumerate(processed_messages):
                msg['timestamp'] = start_offset + (i * time_step)
                
            logger.info(f"Distributed {message_count} messages from {start_offset:.2f}s to {start_offset + effective_duration:.2f}s")
            return processed_messages
        else:
            # Return original messages if timestamps are already good
            return messages 

    def _add_audio_to_video(self, original_video: str, video_without_audio: str, output_path: str):
        """Add audio back to the video using ffmpeg directly"""
        try:
            import subprocess
            
            # Check if original video has audio
            probe_cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_type', '-of', 'default=noprint_wrappers=1:nokey=1',
                original_video
            ]
            
            has_audio = subprocess.check_output(probe_cmd).decode().strip() == 'audio'
            
            if has_audio:
                # Use ffmpeg to copy audio stream
                cmd = [
                    'ffmpeg', '-y',
                    '-i', video_without_audio,
                    '-i', original_video,
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-map', '0:v:0',
                    '-map', '1:a:0',
                    '-shortest',
                    output_path
                ]
                
                subprocess.run(cmd, check=True, capture_output=True)
            else:
                # If no audio, just rename the file
                os.rename(video_without_audio, output_path)
                
        except Exception as e:
            logger.error(f"Error adding audio to video: {str(e)}")
            # If audio addition fails, use the video without audio
            os.rename(video_without_audio, output_path) 

    def _add_overlay_to_frame(self, frame, messages: List[Dict], current_time: float) -> np.ndarray:
        """Add chat overlay to a single frame"""
        try:
            frame_height, frame_width = frame.shape[:2]
            
            # Get messages for this timestamp
            visible_messages = [
                msg for msg in messages
                if msg.get('timestamp', 0) <= current_time
            ][-self.max_messages:]
            
            if not visible_messages:
                return frame
            
            # Calculate chat area dimensions
            chat_area_width = int(frame_width * 0.4)  # 40% of frame width
            chat_area_height = self.message_spacing * self.max_messages
            
            # Get position settings
            position_config = self.position_settings.get(self.position, self.position_settings['bottom-left'])
            margin_x = 20
            margin_y = 40
            
            # Calculate starting position
            if 'y_offset_from_bottom' in position_config:
                y_pos = frame_height - margin_y - chat_area_height
            else:
                y_pos = margin_y
            
            # Process each message
            for msg in reversed(visible_messages):
                if not msg.get('message'):
                    continue
                
                # Format message
                username, message_text = self._format_message(msg)
                username_text = f"{username}: "
                
                # Calculate text dimensions
                username_width = cv2.getTextSize(
                    username_text,
                    self.font,
                    self.font_scale,
                    self.thickness
                )[0][0]
                
                # Calculate available width for message
                max_msg_width = chat_area_width - username_width - 20
                message_lines = self.wrap_text(message_text, max_msg_width)
                
                # Calculate total height for this message
                message_height = len(message_lines) * self.line_spacing
                
                # Check if we have space to draw this message
                if y_pos < margin_y or y_pos + message_height > frame_height - margin_y:
                    continue
                
                # Set x position based on alignment
                if position_config['align'] == 'right':
                    x_pos = frame_width - chat_area_width + margin_x
                else:
                    x_pos = margin_x
                
                # Draw username
                cv2.putText(
                    frame,
                    username_text,
                    (x_pos, y_pos + self.line_spacing),
                    self.font,
                    self.font_scale,
                    self._hex_to_bgr(msg.get('color', '#FFFFFF')),
                    self.thickness
                )
                
                # Draw message lines
                for i, line in enumerate(message_lines):
                    line_y = y_pos + ((i + 1) * self.line_spacing)
                    cv2.putText(
                        frame,
                        line,
                        (x_pos + username_width, line_y),
                        self.font,
                        self.font_scale,
                        (255, 255, 255),  # White color for message
                        self.thickness
                    )
                
                # Move y position for next message
                y_pos += message_height + self.message_spacing
            
            return frame
            
        except Exception as e:
            logger.error(f"Error adding overlay to frame: {str(e)}")
            return frame 