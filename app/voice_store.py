import os
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import HTTPException, status


VOICES_DIR = Path(os.getenv("VOICES_DIR", "/voices"))


class VoiceStore:
    def __init__(self, voices_dir: Path = VOICES_DIR):
        self.voices_dir = voices_dir
        self.voices_dir.mkdir(parents=True, exist_ok=True)

    def _get_voice_path(self, voice_id: str) -> Path:
        return self.voices_dir / voice_id

    def _get_metadata_path(self, voice_id: str) -> Path:
        return self._get_voice_path(voice_id) / "metadata.json"

    def _get_anchor_path(self, voice_id: str, version: Optional[int] = None) -> Path:
        if version:
            return (
                self.voices_dir / voice_id / "versions" / f"v{version}" / "anchor.wav"
            )
        return self.voices_dir / voice_id / "anchor.wav"

    def _load_metadata(self, voice_id: str) -> Dict[str, Any]:
        meta_path = self._get_metadata_path(voice_id)
        if not meta_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Voice {voice_id} not found",
            )
        with open(meta_path, "r") as f:
            return json.load(f)

    def _save_metadata(self, voice_id: str, metadata: Dict[str, Any]):
        meta_path = self._get_metadata_path(voice_id)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def list_voices(self) -> List[Dict[str, Any]]:
        voices = []
        if not self.voices_dir.exists():
            return voices

        for voice_dir in self.voices_dir.iterdir():
            if not voice_dir.is_dir():
                continue
            try:
                metadata = self._load_metadata(voice_dir.name)
                voices.append(
                    {
                        "voice_id": voice_dir.name,
                        "name": metadata.get("name", "Unknown"),
                        "labels": metadata.get("labels", {}),
                        "category": metadata.get("category", "cloned"),
                        "description": metadata.get("description", ""),
                    }
                )
            except Exception:
                continue
        return voices

    def get_voice(self, voice_id: str) -> Dict[str, Any]:
        metadata = self._load_metadata(voice_id)
        anchor_path = self._get_anchor_path(voice_id, metadata.get("voice_version"))

        return {
            "voice_id": voice_id,
            "name": metadata.get("name", "Unknown"),
            "labels": metadata.get("labels", {}),
            "category": metadata.get("category", "cloned"),
            "description": metadata.get("description", ""),
            "voice_version": metadata.get("voice_version", 1),
            "created_at": metadata.get("created_at", ""),
            "metadata": metadata,
        }

    def create_voice(
        self,
        name: str,
        anchor_wav_path: Path,
        labels: Optional[Dict[str, str]] = None,
        category: str = "cloned",
        description: str = "",
        method: str = "clone",
        prompts: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        ref_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        voice_id = str(uuid.uuid4())
        voice_dir = self.voices_dir / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)

        anchor_dest = voice_dir / "anchor.wav"
        shutil.copy2(anchor_wav_path, anchor_dest)
        anchor_wav_path.unlink()

        hf_rev_base = os.getenv("HF_REV_BASE", "")
        hf_rev_design = os.getenv("HF_REV_DESIGN", "")

        metadata = {
            "name": name,
            "labels": labels or {},
            "category": category,
            "description": description,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "method": method,
            "prompts": prompts or {},
            "source": source,
            "ref_text": ref_text or "",
            "voice_version": 1,
            "model_revisions": {
                "base": hf_rev_base or "default",
                "design": hf_rev_design or "default",
            },
            "preprocessing": {
                "vad_enabled": True,
                "target_sample_rate": 16000,
                "channels": 1,
                "format": "pcm_s16le",
            },
        }

        self._save_metadata(voice_id, metadata)

        return {
            "voice_id": voice_id,
            "name": name,
            "labels": labels or {},
            "category": category,
            "description": description,
            "created_at": metadata["created_at"],
        }

    def get_anchor_wav_path(self, voice_id: str, version: Optional[int] = None) -> Path:
        metadata = self._load_metadata(voice_id)
        if version is None:
            version = metadata.get("voice_version", 1)

        anchor_path = self._get_anchor_path(voice_id, version)

        if not anchor_path.exists():
            anchor_path = self._get_anchor_path(voice_id, None)

        if not anchor_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Anchor wav not found for voice {voice_id}",
            )

        return anchor_path
        
    def delete_voice(self, voice_id: str) -> Dict[str, Any]:
        """Delete a voice from the store.
        
        Args:
            voice_id: ID of the voice to delete
            
        Returns:
            Dictionary with voice details that was deleted
            
        Raises:
            HTTPException: If voice not found
        """
        # Get voice details before deletion for confirmation
        voice = self.get_voice(voice_id)
        
        # Get voice directory
        voice_dir = self._get_voice_path(voice_id)
        
        # Check if it exists
        if not voice_dir.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Voice {voice_id} not found",
            )
            
        # Delete the entire directory
        shutil.rmtree(voice_dir)
        
        return {
            "voice_id": voice_id,
            "name": voice["name"],
            "deleted": True,
        }


voice_store = VoiceStore()
