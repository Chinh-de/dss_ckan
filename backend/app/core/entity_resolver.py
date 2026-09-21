import json
import logging
from pathlib import Path
from typing import Optional, Any, Dict

logger = logging.getLogger("entity_resolver")

class EntityResolver:
    def __init__(self):
        self._name_map = {}
        self._loaded = False
        self._load()

    def _load(self):
        try:
            from app.core.config import settings
            names_path = Path(settings.DATA_DIR) / "entity_names.json"
            if names_path.exists():
                with open(names_path, "r", encoding="utf-8") as f:
                    self._name_map = json.load(f)
                logger.info(f"Loaded {len(self._name_map)} curated entity names.")
                self._loaded = True
        except Exception as e:
            logger.warning(f"Could not load entity_names.json: {e}")

    def resolve(self, entity_id: Any, entity_type: str = "Entity") -> str:
        if not self._loaded:
            self._load()

        id_str = str(entity_id).strip()

        # If it's already a non-numeric string (e.g. genre "Comedy" or already resolved name), return it
        if not id_str.isdigit():
            return id_str

        # Check curated map
        if id_str in self._name_map:
            return self._name_map[id_str]

        # Format gracefully with role rather than raw naked number
        type_prefix = entity_type if entity_type and entity_type != "Entity" else "Entity"
        return f"{type_prefix} #{id_str}"

    def get_all_mappings(self):
        if not self._loaded:
            self._load()
        return self._name_map

entity_resolver = EntityResolver()
