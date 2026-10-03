"""Holds the three champion models and swaps them atomically on /reload."""

from __future__ import annotations

import json
import logging
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    supervised: object
    semi_supervised: object
    anomaly: object
    versions: dict[str, str]
    representatives: dict | None = None
    loaded_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


class ModelStore:
    def __init__(self, alias: str = "champion"):
        self.alias = alias
        self._bundle: ModelBundle | None = None
        self._lock = threading.Lock()
        self.last_error: str | None = None

    @property
    def bundle(self) -> ModelBundle | None:
        return self._bundle

    def set_bundle(self, bundle: ModelBundle) -> None:
        with self._lock:
            self._bundle = bundle
            self.last_error = None

    def load_from_registry(self) -> ModelBundle:
        """Load ``models:/<name>@<alias>`` for all three models (+ the labelling images)."""
        from unsupervised_mlops import tracking

        models, versions, semi_run = {}, {}, None
        for key, name in tracking.MODEL_NAMES.items():
            mv = tracking.get_alias_version(name, self.alias)
            if mv is None:
                raise LookupError(f"No '{self.alias}' version of {name} in the registry yet")
            models[key] = tracking.load_model(name, self.alias)
            versions[key] = str(mv.version)
            if key == "semi_supervised":
                semi_run = mv.run_id
        reps = None
        if semi_run:
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    p = tracking.download_run_artifact(
                        semi_run, "labeling/representatives.json", tmp
                    )
                    reps = json.loads(Path(p).read_text())
            except Exception as exc:  # labelling page is optional
                log.warning("Could not load representative images: %s", exc)
        bundle = ModelBundle(
            models["supervised"], models["semi_supervised"], models["anomaly"], versions, reps
        )
        self.set_bundle(bundle)
        log.info("Loaded champion models %s", versions)
        return bundle

    def registry_versions(self) -> dict[str, str] | None:
        from unsupervised_mlops import tracking

        out = {}
        for key, name in tracking.MODEL_NAMES.items():
            mv = tracking.get_alias_version(name, self.alias)
            if mv is None:
                return None
            out[key] = str(mv.version)
        return out

    def refresh_if_stale(self) -> bool:
        """Reload when the registry's alias points at different versions than the ones served.

        Lets several API replicas/workers pick up a new champion without each receiving /reload."""
        try:
            current = self.registry_versions()
        except Exception as exc:
            log.warning("Registry poll failed: %s", exc)
            return False
        if current and (self._bundle is None or current != self._bundle.versions):
            log.info("Champion changed in the registry (%s) - reloading", current)
            return self.try_load()
        return False

    def try_load(self) -> bool:
        try:
            self.load_from_registry()
            return True
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("Models not loaded: %s", self.last_error)
            return False
