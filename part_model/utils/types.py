"""Define commonly used types."""

from __future__ import annotations

from typing import Any

import torch
from jaxtyping import Float, Int

# Batch of images
BatchImages = Float[torch.Tensor, "batch channels height width"]
# Predicted logits segmentation masks
BatchLogitMasks = Float[torch.Tensor, "batch classes height width"]
# Hard-label segmentation masks
BatchSegMasks = Int[torch.Tensor, "batch height width"]
# Predicted class logits
Logits = Float[torch.Tensor, "batch classes"]
# TODO(documentation): Consider impose a stricter type for detectron2
# List of samples represented by dicts
SamplesList = list[dict[str, Any]]
# Model ouput dictionary
OutputsDict = dict[
    str, Float[torch.Tensor, "batch ..."] | Float[torch.Tensor, "*loss"] | None
]
# List of detectron2 outputs
DetectronOutputsList = list[dict[str, Any]]
