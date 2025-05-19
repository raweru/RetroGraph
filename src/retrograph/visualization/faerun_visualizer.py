"""
Visualization module for RetroGraph.

This module handles creating Faerun visualizations.
"""

import os
from typing import List, Dict, Tuple, Any
import numpy as np
import colorcet as cc
from matplotlib.colors import ListedColormap
from faerun import Faerun
from tqdm import tqdm

from ..config import (
    FAERUN_MASTER_PLOT_NAME,
    FAERUN_PROXY_LAYER_NAME,
    FAERUN_EDGE_PLOT_NAME,
    FAERUN_EDGE_COLOR,
    FAERUN_GROUP_PLOT_PREFIX,
    FAERUN_TEMPLATE_NAME,
    COLORMAP_CATEGORICAL,
    TRANSPARENT_COLOR
)
from ..utils.logging import logger
from .image_generator import ImageGenerator

class FaerunVisualizer:
    """Class for creating Faerun visualizations."""
    
    def __init__(self, config: Any):
        """Initialize the visualizer with configuration.
        
        Args:
            config: Configuration object containing visualization settings.
        """
        self.config = config
        self.faerun_plot = Faerun(
            clear_color=config.bg_color,
            view="front",
            coords=False,
            title=config.title
        )

    def prepare_faerun_data(
        self,
        trees: List[Any],
        original_trees_data: List[Tuple[int, int, Dict[str, Any]]]
    ) -> Tuple[List[int], List[str], Dict[int, str]]:
        """Processes tree data to generate labels and groupings for Faerun.

        Args:
            trees: List of SynthesisTree objects used for layout.
            original_trees_data: List of original (mol_idx, tree_idx, tree_dict) tuples.

        Returns:
            A tuple containing:
                - molecule_numeric_ids: List assigning a numeric group ID to each tree.
                - tooltip_labels: List of base64 image data URLs (trees) for tooltips.
                - mol_id_to_legend_html: Dict mapping group ID to HTML for legend.
        """
        print(f"Processing {len(trees)} trees for Faerun visualization data...")
        molecule_numeric_ids: List[int] = []
        tooltip_labels: List[str] = []
        mol_key_to_id: Dict[str, int] = {}
        mol_id_to_legend_html: Dict[int, str] = {}
        next_mol_id = 0

        # Use tqdm for progress visualization
        pbar = tqdm(
            zip(trees, original_trees_data),
            total=len(trees),
            desc="Generating Faerun data"
        )

        for idx, (tree, orig_data) in enumerate(pbar):
            _mol_idx, _tree_idx, tree_dict = orig_data
            root_node = tree.root

            # Determine Molecule Key (Group Identifier)
            root_smiles = root_node.data.get("smiles", "") if root_node else ""
            if root_smiles:
                mol_key = root_smiles
            else:
                label_part = root_node.label.split("-tree")[0] if root_node and root_node.label else f"unknown_mol_{idx}"
                mol_key = label_part
                logger.warning(f"Tree {idx} root node missing SMILES. Using label '{mol_key}' as group key.")

            # Assign Numeric Group ID and Create Legend Entry
            if mol_key not in mol_key_to_id:
                assigned_id = next_mol_id
                mol_key_to_id[mol_key] = assigned_id

                # Create legend HTML (try image first, fallback to text)
                mol_img = ImageGenerator.create_molecule_image(root_smiles)
                legend_text = mol_key
                legend_html = ImageGenerator.image_to_base64_data_url(mol_img)
                if legend_html:
                    legend_html = (
                        f'<img src="{legend_html}" '
                        f'alt="{legend_text}" title="{legend_text}" '
                        f'style="vertical-align: middle;">'
                    )
                else:
                    legend_html = legend_text

                mol_id_to_legend_html[assigned_id] = legend_html
                next_mol_id += 1

            current_mol_numeric_id = mol_key_to_id[mol_key]
            molecule_numeric_ids.append(current_mol_numeric_id)

            # Generate Tooltip Label (Tree Image)
            tree_img = ImageGenerator.create_tree_image(tree_dict)
            tooltip_label = ImageGenerator.image_to_base64_data_url(tree_img)
            tooltip_labels.append(tooltip_label)

        print(f"Processed {len(trees)} trees. Found {len(mol_key_to_id)} unique target molecules (groups).")
        return molecule_numeric_ids, tooltip_labels, mol_id_to_legend_html

    def _add_faerun_scatter_layers(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        tooltip_labels: List[str],
        molecule_numeric_ids: List[int],
        mol_id_to_legend_html: Dict[int, str]
    ) -> None:
        """Adds the necessary scatter plot layers to the Faerun object."""
        num_points = len(x_coords)
        tree_node_indices = [str(i) for i in range(num_points)]
        transparent_cmap = ListedColormap([TRANSPARENT_COLOR])

        # Add Invisible Master Scatter Plot (for Edges)
        self.faerun_plot.add_scatter(
            FAERUN_MASTER_PLOT_NAME,
            {
                "x": x_coords,
                "y": y_coords,
                "c": [0] * num_points,
                "labels": tooltip_labels,
            },
            shader="smoothCircle",
            point_scale=0.0,
            max_point_size=0,
            categorical=True,
            colormap=transparent_cmap,
            has_legend=False,
            interactive=False
        )

        # Add Invisible Interactive Proxy Layer (for Hover)
        self.faerun_plot.add_scatter(
            FAERUN_PROXY_LAYER_NAME,
            {
                "x": x_coords,
                "y": y_coords,
                "c": [0] * num_points,
                "labels": tree_node_indices
            },
            shader="smoothCircle",
            point_scale=5.0,
            max_point_size=1,
            categorical=True,
            colormap=transparent_cmap,
            has_legend=False,
            interactive=True
        )

        # Add Visible Scatter Plots per Molecule Group
        unique_mol_ids = sorted(mol_id_to_legend_html.keys())
        num_groups = len(unique_mol_ids)
        if num_groups == 0:
            print("Warning: No molecule groups found to plot.")
            return

        # Ensure colormap has enough colors, wrap around if necessary
        colors_list = [COLORMAP_CATEGORICAL[i % len(COLORMAP_CATEGORICAL)] for i in range(num_groups)]
        group_colors_cmap = ListedColormap(colors_list)

        for group_idx, mol_numeric_id in enumerate(unique_mol_ids):
            # Find indices belonging to this group
            indices = [i for i, m_id in enumerate(molecule_numeric_ids) if m_id == mol_numeric_id]
            if not indices:
                print(f"Warning: No points found for molecule group ID {mol_numeric_id}. Skipping plot.")
                continue

            # Extract data for this group
            group_x = [x_coords[i] for i in indices]
            group_y = [y_coords[i] for i in indices]
            group_tooltip_labels = [tooltip_labels[i] for i in indices]

            scatter_name = f"{FAERUN_GROUP_PLOT_PREFIX}{mol_numeric_id}"
            legend_label_html = mol_id_to_legend_html.get(mol_numeric_id, f"Group {mol_numeric_id}")

            self.faerun_plot.add_scatter(
                scatter_name,
                {
                    "x": group_x,
                    "y": group_y,
                    "c": [group_idx] * len(indices),
                    "labels": group_tooltip_labels,
                },
                shader="smoothCircle",
                colormap=group_colors_cmap,
                point_scale=self.config.point_scale,
                max_point_size=self.config.max_point_size,
                has_legend=True,
                legend_labels=[(group_idx, legend_label_html)],
                categorical=True,
                interactive=True
            )

    def create_visualization(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        edges: List[Tuple[int, int]],
        molecule_numeric_ids: List[int],
        tooltip_labels: List[str],
        mol_id_to_legend_html: Dict[int, str]
    ) -> None:
        """Creates and saves the Faerun interactive plot.

        Args:
            x_coords: X coordinates of the layout.
            y_coords: Y coordinates of the layout.
            edges: List of graph edges (source_idx, target_idx).
            molecule_numeric_ids: List mapping each point to its molecule group ID.
            tooltip_labels: List of base64 tree images for tooltips.
            mol_id_to_legend_html: Dict mapping group ID to legend HTML.
        """
        print("Creating Faerun visualization...")

        # Add all scatter layers
        self._add_faerun_scatter_layers(
            x_coords, y_coords, tooltip_labels,
            molecule_numeric_ids, mol_id_to_legend_html
        )

        # Add Tree Edges
        if edges:
            self.faerun_plot.add_tree(
                FAERUN_EDGE_PLOT_NAME,
                {"from": [e[0] for e in edges], "to": [e[1] for e in edges]},
                point_helper=FAERUN_MASTER_PLOT_NAME,
                color=FAERUN_EDGE_COLOR,
            )
        else:
            print("No edges found to add to the plot.")

        # Get output path components
        output_path = self.config.output_file
        output_dir = os.path.dirname(output_path)
        base_name = os.path.basename(output_path)
        
        # Ensure output directory exists
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(f"Saving visualization to {output_path}")
        try:
            self.faerun_plot.plot(
                file_name=base_name,
                path=output_dir or ".",  # Use current directory if no path specified
                template=FAERUN_TEMPLATE_NAME
            )
            print(f"Visualization complete!")
        except Exception as e_plot:
            logger.error(f"Faerun plot saving failed: {e_plot}")
            raise 