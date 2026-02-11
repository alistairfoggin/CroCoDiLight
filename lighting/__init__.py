"""
CroCoDiLight - CroCo-based Delighting and Relighting

Extends CroCo (Cross-view Completion) for image relighting tasks.
"""

import os
import sys
# Add croco submodule to sys.path so its internal bare `from models.*` imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'croco'))

from lighting.relighting_modules import (
    LightingExtractor,
    LightingEntangler,
    LightingDecoder,
)
from lighting.relighting_model import (
    CroCoDecode,
    RelightModule,
    LightingMapper,
)
from lighting.dataloader import (
    BigTimeDataset,
    DualDirectoryDataset,
    HypersimDataset,
    CGIntrinsicDataset,
    ScenePairDataset,
)

__version__ = "1.0.0"
