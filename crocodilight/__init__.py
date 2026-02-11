"""
CroCoDiLight - CroCo-based Delighting and Relighting

Extends CroCo (Cross-view Completion) for image relighting tasks.
"""

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add project root so `from croco.models.*` resolves (croco is a namespace package without __init__.py)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
# Add croco submodule dir so its internal bare `from models.*` imports resolve
_croco_dir = os.path.join(_project_root, 'croco')
if _croco_dir not in sys.path:
    sys.path.insert(0, _croco_dir)

from crocodilight.relighting_modules import (
    LightingExtractor,
    LightingEntangler,
    LightingDecoder,
)
from crocodilight.relighting_model import (
    CroCoDecode,
    RelightModule,
    LightingMapper,
)
from crocodilight.dataloader import (
    BigTimeDataset,
    DualDirectoryDataset,
    HypersimDataset,
    CGIntrinsicDataset,
    ScenePairDataset,
)
from crocodilight.inference import (
    get_device,
    load_model,
    load_mapper,
    get_transform,
    load_and_transform,
    save_tensor_image,
    pad_to_min_size,
    unpad,
    extract_features,
    process_input,
)

__version__ = "1.0.0"
