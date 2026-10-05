from .cnn3d import CNN3D
from .early_fusion import EarlyFusion
from .late_fusion import LateFusion
from .per_frame import PerFrame

MODELS = {
    'per_frame': PerFrame,
    'late_fusion': LateFusion,
    'early_fusion': EarlyFusion,
    'cnn3d': CNN3D,
}


def build_model(name, num_frames=10, num_classes=10):
    return MODELS[name](num_frames=num_frames, num_classes=num_classes)
