import asyncio
import ollama
import random
import logging
import time
import re
import subprocess
import numpy as np
from typing import List, Dict
import hashlib
import os
import json

logger = logging.getLogger(__name__)

class ChatGenerator:
    def __init__(self):
        """Initialize chat generator with minimal settings"""
        try:
            # Try to start Ollama service if not running
            try:
                self.llm = ollama.Client()
                self.llm.list()  # Test connection
            except ConnectionRefusedError:
                logger.info("Starting Ollama service...")
                subprocess.Popen(['ollama', 'serve'])
                # Wait longer for service to start and retry multiple times
                max_retries = 3  # Reduced from 5
                for i in range(max_retries):
                    try:
                        time.sleep(2)  # Reduced from 3
                        self.llm = ollama.Client()
                        self.llm.list()  # Test connection
                        logger.info("Successfully connected to Ollama")
                        break
                    except ConnectionRefusedError:
                        if i == max_retries - 1:
                            raise
                        logger.info(f"Connection attempt {i+1} failed, retrying...")
            
            self.colors = ["#FF0000", "#00FF00", "#0000FF", "#FF7F50", "#1E90FF"]  # Reduced colors
            self.batch_size = 3  # Reduced from 5
            
            # Simplified username components
            self.prefixes = ["Pro", "Gaming", "Twitch", "Epic"]
            self.roots = ["Gamer", "Ninja", "Master", "Lord"]
            self.suffixes = ["TV", "Live", str(random.randint(1, 999))]
            
            # Add message display configuration
            self.message_format = {
                'internal': True,  # Keep timestamps for internal use
                'display': {
                    'show_timestamps': True,  # Show timestamps in preview only
                    'preview_format': '{timestamp:.1f}s {username}: {message}',
                    'overlay_format': '{username}: {message}'
                }
            }
            
            logger.info("ChatGenerator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            raise

    # Update the class-level prompt to be more focused
    TWITCH_PROMPT = """
You are simulating a livestream chat reacting to what's happening right now. Generate short, authentic messages that feel like they are from real viewers. Use this exact format:

username: message

Rules:
1. Focus only on what is currently happening on screen.
2. Keep each message short, casual, and natural—like typical live chat.
3. Do not include timestamps, numbers, hashtags, asterisks, or other special formatting.
4. Do not include meta descriptions, secondary usernames, or emoji numbers.
5. No references to future or external context—comment only on what is visibly or audibly happening in the stream.
"""


    def _generate_username(self) -> str:
        """Generate a random, authentic-looking livestream chat username. The username should:
- Reflect the casual and creative style of human users.
- Incorporate a mix of letters, numbers, or playful symbols that feel natural (e.g., 'xXGamerProXx' or 'BeastModeTV').
- Avoid generic or overly simplistic names, ensuring it does not appear AI-generated.

The resulting username must blend seamlessly into a realistic livestream chat environment."""
        username_parts = []
        
        # 50% chance to add prefix
        if random.random() < 0.5:
            username_parts.append(random.choice(self.prefixes))
        
        # Always add a root
        username_parts.append(random.choice(self.roots))
        
        # 70% chance to add suffix
        if random.random() < 0.7:
            suffix = random.choice(self.suffixes)
            # If numeric suffix, format it properly
            if suffix.isdigit():
                suffix = str(random.randint(1, 9999))
            username_parts.append(suffix)
            
        # Join with either underscore or nothing
        separator = "_" if random.random() < 0.3 else ""
        username = separator.join(username_parts)
        
        # 20% chance to add xX...Xx format
        if random.random() < 0.2:
            username = f"xX{username}Xx"
            
        return username

    def format_message_for_preview(self, message: Dict) -> str:
        """Format a message for the chat preview"""
        return self.message_format['display']['preview_format'].format(
            timestamp=message['_timestamp'],
            username=message['username'],
            message=message['message']
        )

    async def generate_chat(self, video_analysis: Dict, config: Dict) -> List[Dict]:
        try:
            start_time = time.time()
            logger.info(f"Starting chat generation with config: {config}")
            
            # Check if we already have full chat results cached
            cache_key = hashlib.md5(f"{video_analysis['duration']}_{config}".encode()).hexdigest()
            cached_result = self._check_chat_cache(cache_key)
            if cached_result:
                logger.info(f"Using cached chat results: {len(cached_result)} messages")
                return cached_result
            
            # Group frames by scene for more coherent chat
            scenes = self._group_frames_by_scene(video_analysis.get('frame_analysis', []))
            logger.info(f"Identified {len(scenes)} distinct scenes in video")
            
            transcript_segments = video_analysis.get('transcription', {}).get('segments', [])
            
            chat_messages = []
            
            # Process scenes sequentially to reduce load
            for scene in scenes:
                logger.info(f"Processing scene at {scene[0].get('timestamp', 0):.1f}s")
                scene_messages = await self._generate_batch_messages(
                    scene,
                    config['messages_per_minute'],
                    config['num_chatters'],
                    config['emote_frequency'],
                    transcript_segments
                )
                chat_messages.extend(scene_messages)
            
            chat_messages.sort(key=lambda x: x['_timestamp'])
            
            # Cache the final results
            self._save_chat_cache(cache_key, chat_messages)
            
            # Log some stats
            total_time = time.time() - start_time
            logger.info(f"Chat generation complete: {len(chat_messages)} messages in {total_time:.1f}s")
            
            return chat_messages
            
        except Exception as e:
            logger.error(f"Chat generation failed: {str(e)}")
            return []

    def _is_valid_message(self, message: str) -> bool:
        """Validate chat message content"""
        # Skip if message is too short or empty
        if not message or len(message.strip()) < 3:
            return False
            
        # Skip if message looks like just a username/word
        if len(message.split()) <= 1:
            return False
            
        # Skip messages that look like usernames
        username_patterns = [
            r'^[A-Za-z]+\d+$',  # word followed by numbers
            r'^[A-Za-z]+_?[A-Za-z]+\d*$',  # word_word or wordword
            r'^\w+(?:Gaming|Stream|Live|TV|YT|Pro)(?:\d+)?$',  # Gaming/Stream suffix
        ]
        if any(re.match(pattern, message.strip()) for pattern in username_patterns):
            return False
            
        return True

    async def _generate_batch_messages(self, frames: List[Dict], msgs_per_minute: int, num_chatters: int, emote_frequency: float, transcript_segments=None) -> List[Dict]:
        try:
            if not frames:
                logger.error("No frames provided for message generation")
                return []

            # Get scene time window
            start_time = frames[0].get('timestamp', 0)
            end_time = frames[-1].get('timestamp', start_time + 2)

            # Aggregate transcript for this scene
            scene_transcript = ""
            if transcript_segments:
                relevant_segments = [
                    seg['text'] for seg in transcript_segments
                    if ('start' in seg and 'end' in seg and seg['end'] >= start_time and seg['start'] <= end_time)
                ]
                scene_transcript = " ".join(relevant_segments).strip()

            # Get context from frames - use only the last frame for efficiency
            visual_context = frames[-1].get('visual_analysis', '')
            audio_context = scene_transcript if scene_transcript else frames[-1].get('transcription', {}).get('text', '')
            
            # Create focused context
            context = f"""Current scene:\nVisual: {visual_context}\nAudio: {audio_context}"""

            batch_duration = end_time - start_time
            messages_needed = max(1, int((msgs_per_minute / 60) * batch_duration * 0.8))  # 80% of original

            prompt = f"{self.TWITCH_PROMPT}\n\nScene to react to:\n{context}\n\nGenerate {messages_needed} messages:"

            logger.info(f"Generating messages for scene...")
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm.generate(
                    model='llama2',
                    prompt=prompt,
                    stream=False
                )
            )
            
            response_text = response['response'] if isinstance(response, dict) else str(response)
            raw_messages = []
            
            for line in response_text.split('\n'):
                line = line.strip()
                if not line or ':' not in line:
                    continue
                
                try:
                    username, message = line.split(':', 1)
                    message = message.strip()
                    
                    # Validate message content
                    if not self._is_valid_message(message):
                        continue
                        
                    # Skip messages with formatting markers
                    if any(x in message for x in ['|', '#', '*', '[', ']']):
                        continue
                        
                    raw_messages.append(line)
                except ValueError:
                    continue

            if raw_messages:
                timestamps = np.linspace(start_time, end_time, len(raw_messages))
                messages = []
                
                for msg, timestamp in zip(raw_messages, timestamps):
                    try:
                        _, message = msg.split(':', 1)
                        username = self._generate_username()
                        message = message.strip()
                        
                        messages.append({
                            '_timestamp': float(timestamp),  # Internal timestamp
                            'timestamp': float(timestamp),   # For UI compatibility
                            'username': username,
                            'message': message,
                            'color': random.choice(self.colors),
                            'display': f"{username}: {message}"
                        })
                    except ValueError:
                        continue

                return messages
            
            return []
            
        except Exception as e:
            logger.error(f"Batch message generation failed: {str(e)}")
            return []

    def _check_chat_cache(self, cache_key: str) -> List[Dict]:
        """Check if we have cached chat messages for this video"""
        try:
            cache_file = os.path.join('cache', 'chat', f"{cache_key}.json")
            if os.path.exists(cache_file):
                logger.info(f"Found chat cache file: {cache_file}")
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                return cached_data
            return None
        except Exception as e:
            logger.warning(f"Failed to read chat cache: {e}")
            return None
            
    def _save_chat_cache(self, cache_key: str, messages: List[Dict]) -> None:
        """Save chat messages to cache"""
        try:
            os.makedirs(os.path.join('cache', 'chat'), exist_ok=True)
            cache_file = os.path.join('cache', 'chat', f"{cache_key}.json")
            
            # Make sure the messages are JSON serializable
            serializable_messages = []
            for msg in messages:
                clean_msg = {k: v for k, v in msg.items() if k != '_timestamp'}
                clean_msg['timestamp'] = msg.get('_timestamp', 0)
                serializable_messages.append(clean_msg)
                
            with open(cache_file, 'w') as f:
                json.dump(serializable_messages, f)
                
            logger.info(f"Saved {len(messages)} messages to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save chat cache: {e}")
            
    def _group_frames_by_scene(self, frames: List[Dict]) -> List[List[Dict]]:
        """Group frames into scenes for more coherent chat generation"""
        if not frames:
            return []
            
        scenes = []
        current_scene = [frames[0]]
        scene_threshold = 15.0  # Increased from 10.0 to reduce number of scenes
        
        for i in range(1, len(frames)):
            prev_timestamp = frames[i-1].get('timestamp', 0)
            curr_timestamp = frames[i].get('timestamp', 0)
            
            # Check if this is a new scene (significant time gap or explicit scene change)
            time_diff = curr_timestamp - prev_timestamp
            is_new_scene = time_diff > scene_threshold
            
            if is_new_scene:
                scenes.append(current_scene)
                current_scene = [frames[i]]
            else:
                current_scene.append(frames[i])
                
        # Add the last scene
        if current_scene:
            scenes.append(current_scene)
            
        logger.info(f"Grouped {len(frames)} frames into {len(scenes)} scenes")
        return scenes