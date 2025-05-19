"""
Configuration module for RetroGraph.

This module contains constants and configuration management for the RetroGraph package.
"""

import os
from dataclasses import dataclass
from typing import Optional
import colorcet as cc

# Default values
DEFAULT_OUTPUT_FILENAME = "retrograph"
DEFAULT_BG_COLOR = "#151515"
DEFAULT_TED_THRESHOLD = 3.0
DEFAULT_POINT_SCALE = 1.0
DEFAULT_MAX_POINT_SIZE = 10.0
IMAGE_WIDTH = 250
IMAGE_HEIGHT = 250

# Faerun constants
FAERUN_MASTER_PLOT_NAME = "all_trees_master"
FAERUN_PROXY_LAYER_NAME = "tree_node_hover_proxy"
FAERUN_EDGE_PLOT_NAME = "ted_connections"
FAERUN_EDGE_COLOR = "#666666"
FAERUN_GROUP_PLOT_PREFIX = "molecule_"
FAERUN_TEMPLATE_NAME = "template_tree.j2"

# Color constants
COLORMAP_CATEGORICAL = cc.glasbey_light
TRANSPARENT_COLOR = '#FFFFFF00'  # For invisible layers

@dataclass
class RetrographConfig:
    """Configuration class for RetroGraph."""
    input_file: str
    output_file: Optional[str] = None
    ted_threshold: Optional[float] = None
    ted_mode: str = "shape"
    visualize_all: bool = False
    title: str = ""
    bg_color: Optional[str] = None
    create_mst: bool = True
    point_scale: Optional[float] = None
    max_point_size: Optional[float] = None
    
    def __post_init__(self):
        """Set default values for optional parameters after initialization."""
        if self.output_file is None:
            self.output_file = DEFAULT_OUTPUT_FILENAME
        if self.ted_threshold is None:
            self.ted_threshold = DEFAULT_TED_THRESHOLD
        if self.bg_color is None:
            self.bg_color = DEFAULT_BG_COLOR
        if self.point_scale is None:
            self.point_scale = DEFAULT_POINT_SCALE
        if self.max_point_size is None:
            self.max_point_size = DEFAULT_MAX_POINT_SIZE
            
        # Ensure output_file has proper path handling
        self.output_file = os.path.normpath(self.output_file)
        if not os.path.dirname(self.output_file):
            # If no directory specified, use current directory
            self.output_file = os.path.join(os.getcwd(), self.output_file)
    
    def validate(self) -> None:
        """Validate the configuration."""
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        if not self.input_file.endswith((".json", ".json.gz")):
            raise ValueError(f"Input file must be a JSON file: {self.input_file}")
            
        if self.ted_mode not in ["shape", "classification_aware"]:
            raise ValueError(f"Invalid TED mode: {self.ted_mode}")
            
        if self.ted_threshold < 0:
            raise ValueError(f"TED threshold must be non-negative: {self.ted_threshold}")
            
        if self.point_scale <= 0:
            raise ValueError(f"Point scale must be positive: {self.point_scale}")
            
        if self.max_point_size <= 0:
            raise ValueError(f"Max point size must be positive: {self.max_point_size}")
            
        # Ensure output directory exists
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                raise ValueError(f"Could not create output directory {output_dir}: {e}") 