"""External control interfaces; the original vehicle model stays unchanged."""

from .remus_guidance import (
    ControlSample,
    ReferenceAction,
    ReferencePrediction,
    RemusGuidance,
    clone_plant,
    predict_reference,
    velocity_frames,
)

__all__ = [
    'ControlSample', 'ReferenceAction', 'ReferencePrediction', 'RemusGuidance',
    'clone_plant', 'predict_reference', 'velocity_frames',
]
