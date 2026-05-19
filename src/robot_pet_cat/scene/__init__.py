"""Scene assets + scene-state extraction for the Tier 3 brain.

The brain's reward streams (comfort, play, curiosity) all consume a
`scene_state` dict that names what entities exist in the world, where they are,
and which semantic flags apply (elevated / soft / warm / play_target). This
package handles the bookkeeping: an MJCF scene under `assets/scenes/`, a flag
configuration in `semantic_flags.py`, and an extractor in `scene_state.py` that
binds them together at runtime.
"""

from .scene_state import (
    LIVING_ROOM_V0_FLAGS,
    SceneEntity,
    SceneState,
    SemanticFlags,
    extract_scene_state,
)

__all__ = [
    "LIVING_ROOM_V0_FLAGS",
    "SceneEntity",
    "SceneState",
    "SemanticFlags",
    "extract_scene_state",
]
