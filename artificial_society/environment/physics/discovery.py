"""Discovery-Registry der Physik v2: neuartige Eigenschaftsvektoren → stabile IDs.

Bewusst eigene Minimal-Implementierung statt Reuse der v1-Registry aus
environment/materials.py: die loggt v1-Dimensionsnamen über v1-Indizes (auf
v2-Vektoren falsch) und ist Teil des eingefrorenen Hot-File-Contracts.
Kein Print — Logging entscheidet später die Sim-Integration.
"""

from __future__ import annotations

import numpy as np


class DiscoveryV2:
    def __init__(self, similarity_threshold: float = 0.08):
        self.entries: list = []
        self.threshold = similarity_threshold

    def reset(self) -> None:
        self.entries.clear()

    def state_dict(self) -> dict:
        return {"entries": [{**e, "vector": e["vector"].copy()} for e in self.entries]}

    def load_state_dict(self, data: dict) -> None:
        self.entries = [
            {**e, "vector": np.asarray(e["vector"], dtype=np.float32).copy()}
            for e in data.get("entries", [])
        ]

    def register(self, vector: np.ndarray, discoverer_id: int = -1, tick: int = 0) -> str:
        vec = np.asarray(vector, dtype=np.float32)
        for entry in self.entries:
            if float(np.linalg.norm(vec - entry["vector"])) < self.threshold:
                return entry["id"]
        new_id = f"pmat_{len(self.entries):04d}"
        self.entries.append(
            {"id": new_id, "vector": vec.copy(), "discovered_by": discoverer_id, "tick": tick}
        )
        return new_id

    def get_vector(self, mat_id: str):
        for entry in self.entries:
            if entry["id"] == mat_id:
                return entry["vector"].copy()
        return None

    def known_ids(self) -> list:
        return [e["id"] for e in self.entries]


DISCOVERY_V2 = DiscoveryV2()
