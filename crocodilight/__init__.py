"""
CroCoDiLight - CroCo-based Delighting and Relighting

Extends CroCo (Cross-view Completion) for image relighting tasks.
"""

from crocodilight.relighting_modules import (
    DelightingTransformer,
    RelightingTransformer,
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
    pil_to_tensor,
    tensor_to_pil,
    pad_to_min_size,
    unpad,
    extract_features,
    extract_features_pil,
    process_input,
)

__version__ = "1.0.0"
