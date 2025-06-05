import os
import requests
import shutil
from pathlib import Path
from typing import Dict, Any
import json
from src.utils.logger import logger
from src.utils.config import config

class AssetManager:
    """Manages application assets (emotes, fonts, etc.)"""
    
    ASSET_URLS = {
        'emotes': {
            'Kappa.png': 'https://static-cdn.jtvnw.net/emoticons/v2/25/default/dark/3.0',
            'PogChamp.png': 'https://static-cdn.jtvnw.net/emoticons/v2/305954156/default/dark/3.0',
            'KEKW.png': 'https://cdn.betterttv.net/emote/5e9c6c187e090362f8b0b9e8/3x',
            'OMEGALUL.png': 'https://cdn.betterttv.net/emote/583089f4737a8e61abb0186b/3x',
            'monkaS.png': 'https://cdn.betterttv.net/emote/56e9f494fff3cc5c35e5287e/3x',
            'PepeLaugh.png': 'https://cdn.betterttv.net/emote/5c548025009a2e73916b3a37/3x',
            'Sadge.png': 'https://cdn.betterttv.net/emote/5e0fa9d40550d42106b8a489/3x',
            'LULW.png': 'https://cdn.betterttv.net/emote/5dc79d1b27360247dd6516ec/3x',
            'xqcL.png': 'https://cdn.betterttv.net/emote/5e76d338d6581c3724c0f0b2/3x',
            'PeepoGlad.png': 'https://cdn.betterttv.net/emote/5e1a76dd8af14b5f1b438c04/3x'
        },
        'fonts': {
            'Roboto-Regular.ttf': 'https://github.com/googlefonts/roboto/raw/main/src/hinted/Roboto-Regular.ttf'
        }
    }
    
    def __init__(self):
        self.base_path = Path(config.get('paths.assets'))
        self.status_file = self.base_path / 'asset_status.json'
    
    def verify_assets(self) -> bool:
        """Verify all required assets exist"""
        try:
            missing_assets = self.get_missing_assets()
            if missing_assets:
                logger.warning(f"Missing assets: {missing_assets}")
                return False
            return True
        except Exception as e:
            logger.error(f"Asset verification error: {str(e)}")
            return False
    
    async def setup_assets(self) -> bool:
        """Download and set up all required assets"""
        try:
            logger.info("Starting asset setup...")
            
            # Create asset directories
            for asset_type in self.ASSET_URLS.keys():
                path = self.base_path / asset_type
                path.mkdir(parents=True, exist_ok=True)
            
            # Download missing assets
            missing_assets = self.get_missing_assets()
            success = True
            
            for asset_type, assets in missing_assets.items():
                for asset in assets:
                    if not await self._download_asset(asset_type, asset):
                        success = False
            
            # Save status
            if success:
                self._save_status()
            
            return success
            
        except Exception as e:
            logger.error(f"Asset setup error: {str(e)}")
            return False
    
    def get_missing_assets(self) -> Dict[str, list]:
        """Get list of missing assets"""
        missing = {}
        
        for asset_type, assets in self.ASSET_URLS.items():
            missing_assets = []
            for asset in assets:
                path = self.base_path / asset_type / asset
                if not path.exists():
                    missing_assets.append(asset)
            
            if missing_assets:
                missing[asset_type] = missing_assets
        
        return missing
    
    async def _download_asset(self, asset_type: str, asset: str) -> bool:
        """Download a single asset"""
        try:
            url = self.ASSET_URLS[asset_type][asset]
            path = self.base_path / asset_type / asset
            
            logger.info(f"Downloading {asset} from {url}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Successfully downloaded: {asset}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {asset}: {str(e)}")
            return False
    
    def _save_status(self):
        """Save asset status to file"""
        status = {
            'assets': {
                asset_type: {
                    asset: str(self.base_path / asset_type / asset)
                    for asset in assets.keys()
                }
                for asset_type, assets in self.ASSET_URLS.items()
            }
        }
        
        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=2) 