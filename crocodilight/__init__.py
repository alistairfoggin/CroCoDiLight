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
    load_image,
    save_tensor_image,
    pil_to_tensor,
    tensor_to_pil,
    extract_lighting,
    relight,
    apply_mapper,
    process_input,
)

__version__ = "1.0.0"
