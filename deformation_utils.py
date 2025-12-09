# deformation_utils.py (UPDATED)

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pyproj import datadir

# Import the new registry system
from velocity_model_registry import (
    VelocityModelInfo,
    load_registry,
    select_velocity_model as registry_select_model,
    find_model_file,
)


# # Load models from registry at module import
# VELOCITY_MODELS: List[VelocityModelInfo] =[]# load_registry(
# #    include_defaults=True,
# #)

VELOCITY_MODELS: List[VelocityModelInfo] = load_registry(
    include_defaults=True,
)


def select_velocity_model(
    bbox_4326: Tuple[float, float, float, float],
    src_epoch: float,
    dst_epoch: float,
    *,
    choice: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[VelocityModelInfo, List[VelocityModelInfo]]:
    """
    Select velocity/deformation model for transformation.
    
    Now delegates to the registry system for enhanced functionality.
    """
    # Use the registry's selection function with our loaded models
    return registry_select_model(
        bbox_4326=bbox_4326,
        src_epoch=src_epoch,
        dst_epoch=dst_epoch,
        models=VELOCITY_MODELS,
        choice=choice,
        verbose=verbose,
        auto_download=True,  # Enable auto-download
    )


# Keep your bbox intersection helper for compatibility
def _bbox_intersects(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> bool:
    """Check if two bounding boxes intersect."""
    aminx, aminy, amaxx, amaxy = a
    bminx, bminy, bmaxx, bmaxy = b
    return not (amaxx < bminx or bmaxx < aminx or amaxy < bminy or bmaxy < aminy)