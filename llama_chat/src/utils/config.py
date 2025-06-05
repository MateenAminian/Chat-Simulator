from typing import Any, Dict, Optional, Union
import os
import yaml
import json
from pathlib import Path
import threading
from datetime import datetime
import shutil
from src.utils.logger import logger

class ConfigurationManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._config = {}
            self._env_vars = {}
            self._config_path = None
            self._last_load_time = None
            self._auto_reload = False
            self._reload_interval = 300  # 5 minutes
            self._backup_count = 5
            self._initialized = True
            logger.info("Configuration Manager initialized")

    def load(self, config_path: Union[str, Path]) -> bool:
        """Load configuration with validation and backup"""
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            # Create backup before loading
            self._backup_config(config_path)

            # Load and parse config
            with open(config_path, 'r') as f:
                new_config = yaml.safe_load(f)

            # Validate configuration
            if self._validate_config(new_config):
                self._config = new_config
                self._config_path = config_path
                self._last_load_time = datetime.now()
                logger.info(f"Configuration loaded successfully from {config_path}")
                return True
            else:
                logger.error("Configuration validation failed")
                return False

        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return False

    def _backup_config(self, config_path: Path):
        """Create timestamped backup of config file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = config_path.parent / 'config_backups'
            backup_dir.mkdir(exist_ok=True)
            
            backup_path = backup_dir / f"config_{timestamp}.yaml"
            shutil.copy2(config_path, backup_path)
            
            # Clean up old backups
            backups = sorted(backup_dir.glob('config_*.yaml'))
            if len(backups) > self._backup_count:
                for old_backup in backups[:-self._backup_count]:
                    old_backup.unlink()
                    
            logger.debug(f"Created config backup: {backup_path}")
            
        except Exception as e:
            logger.error(f"Error creating config backup: {str(e)}")

    def _validate_config(self, config: Dict) -> bool:
        """Validate configuration structure and values"""
        required_sections = {
            'app': {'name', 'temp_dir', 'debug'},
            'paths': {'outputs', 'cache', 'assets', 'emotes', 'fonts'},
            'cache': {'duration', 'max_size'},
            'ollama': {'url', 'model', 'sample_interval'},
            'chat': {
                'default_style', 'default_position', 'default_speed',
                'default_chatters', 'default_emote_frequency',
                'default_spam_level'
            },
            'logging': {'level', 'file', 'max_size', 'backup_count'}
        }

        try:
            # Check required sections and fields
            for section, fields in required_sections.items():
                if section not in config:
                    logger.error(f"Missing required section: {section}")
                    return False
                
                for field in fields:
                    if field not in config[section]:
                        logger.error(f"Missing required field: {section}.{field}")
                        return False

            # Validate specific values
            if not isinstance(config['cache']['max_size'], (int, float)):
                logger.error("Cache max_size must be a number")
                return False

            if not 0 <= config['chat']['default_emote_frequency'] <= 1:
                logger.error("Emote frequency must be between 0 and 1")
                return False

            # Validate paths
            for path_key, path_value in config['paths'].items():
                if not os.path.exists(path_value):
                    try:
                        os.makedirs(path_value, exist_ok=True)
                        logger.info(f"Created missing directory: {path_value}")
                    except Exception as e:
                        logger.error(f"Could not create directory {path_value}: {str(e)}")
                        return False

            return True

        except Exception as e:
            logger.error(f"Configuration validation error: {str(e)}")
            return False

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation with auto-reload check"""
        try:
            # Check if reload is needed
            if self._auto_reload and self._should_reload():
                self.load(self._config_path)

            # Split the key path
            keys = key_path.split('.')
            value = self._config

            # Traverse the config dictionary
            for key in keys:
                value = value[key]

            return value

        except KeyError:
            logger.warning(f"Configuration key not found: {key_path}")
            return default
        except Exception as e:
            logger.error(f"Error accessing configuration: {str(e)}")
            return default

    def set(self, key_path: str, value: Any, persist: bool = True) -> bool:
        """Set configuration value using dot notation"""
        try:
            # Split the key path
            keys = key_path.split('.')
            config = self._config

            # Traverse to the parent of the target key
            for key in keys[:-1]:
                config = config.setdefault(key, {})

            # Set the value
            config[keys[-1]] = value

            # Persist changes if requested
            if persist and self._config_path:
                return self._save_config()

            return True

        except Exception as e:
            logger.error(f"Error setting configuration: {str(e)}")
            return False

    def _save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            # Create backup before saving
            self._backup_config(self._config_path)

            # Save configuration
            with open(self._config_path, 'w') as f:
                yaml.safe_dump(self._config, f, default_flow_style=False)

            logger.info("Configuration saved successfully")
            return True

        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False

    def _should_reload(self) -> bool:
        """Check if configuration should be reloaded"""
        if not self._last_load_time or not self._config_path:
            return False

        try:
            # Check file modification time
            mtime = os.path.getmtime(self._config_path)
            last_modified = datetime.fromtimestamp(mtime)
            
            # Reload if file was modified after last load
            return last_modified > self._last_load_time

        except Exception as e:
            logger.error(f"Error checking configuration reload: {str(e)}")
            return False

    def enable_auto_reload(self, interval: int = 300):
        """Enable automatic configuration reloading"""
        self._auto_reload = True
        self._reload_interval = interval
        logger.info(f"Auto-reload enabled with {interval}s interval")

    def disable_auto_reload(self):
        """Disable automatic configuration reloading"""
        self._auto_reload = False
        logger.info("Auto-reload disabled")

    def get_all(self) -> Dict:
        """Get complete configuration"""
        return self._config.copy()

    def reset(self):
        """Reset configuration to default values"""
        try:
            default_config_path = Path(__file__).parent.parent / 'config.default.yaml'
            if default_config_path.exists():
                return self.load(default_config_path)
            else:
                logger.error("Default configuration file not found")
                return False
        except Exception as e:
            logger.error(f"Error resetting configuration: {str(e)}")
            return False

# Global configuration instance
config = ConfigurationManager() 