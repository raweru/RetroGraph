"""
Visualizes AiZynthFinder synthesis route trees using TED-based t-MAP layout.

Loads trees from an AiZynthFinder output JSON file (gzipped), converts them
to SynthesisTree objects, computes a t-MAP layout using Tree Edit Distance (TED)
with TEDLayoutGenerator, and generates an interactive HTML visualization using
Faerun.
"""

# Standard library imports
import argparse
import base64
import gzip
import io
import json
import os
import sys
from typing import List, Dict, Tuple, Any, Optional, Union

# Third-party imports
import colorcet as cc
import numpy as np
import pandas as pd
from faerun import Faerun
from matplotlib.colors import ListedColormap
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm

# Local application/library specific imports
# NOTE: These assume retrograph.py is run from the workspace root.
from src.retrograph.core.synthesis_tree import SynthesisTree, SynthesisTreeNode
from src.retrograph.layout_generators.ted_layout_generator import TEDLayoutGenerator
# Attempt AiZynthFinder import - required for core functionality
try:
    from aizynthfinder.reactiontree import ReactionTree
except ImportError:
    print("ERROR: AiZynthFinder is required but not installed.")
    print("Please install it (e.g., 'pip install aizynthfinder') and try again.")
    sys.exit(1)

# --- Constants ---
DEFAULT_OUTPUT_FILENAME = "retrograph"
DEFAULT_BG_COLOR = "#151515"
DEFAULT_TED_THRESHOLD = 3.0
IMAGE_WIDTH = 250 # For molecule legend images
IMAGE_HEIGHT = 250
FAERUN_MASTER_PLOT_NAME = "all_trees_master"
FAERUN_PROXY_LAYER_NAME = "tree_node_hover_proxy"
FAERUN_EDGE_PLOT_NAME = "ted_connections"
FAERUN_EDGE_COLOR = "#666666"
FAERUN_GROUP_PLOT_PREFIX = "molecule_"
FAERUN_TEMPLATE_NAME = "template_tree.j2" # Faerun should find this relative to its path

# Use a readily available colormap
# Using glasbey directly, ensure it has enough distinct colors
# If num_groups > len(cc.glasbey_light), Faerun might cycle colors.
COLORMAP_CATEGORICAL = cc.glasbey_light
TRANSPARENT_COLOR = '#FFFFFF00' # For invisible layers

# --- Tree Conversion Functions ---

def convert_aizynthfinder_node_to_synthesis_node(
    node_data: Dict[str, Any],
    parent_label: str = ""
) -> SynthesisTreeNode:
    """Recursively converts an AiZynthFinder node dictionary to a SynthesisTreeNode.

    Constructs a unique label based on node type, SMILES/ID, or classification.

    Args:
        node_data: Node data dictionary from AiZynthFinder.
        parent_label: Label suffix from the parent node to ensure uniqueness.

    Returns:
        The corresponding SynthesisTreeNode.
    """
    node_type = node_data.get("type", "mol")
    metadata = node_data.get("metadata", {})
    smiles = node_data.get("smiles", "")

    # Create a descriptive and hopefully unique label
    if node_type == "mol":
        mol_id = metadata.get("molecule_id", "")
        label_prefix = f"mol-{mol_id}" if mol_id else smiles if smiles else "Molecule"
    elif node_type == "reaction":
        classification = metadata.get("classification", "")
        label_prefix = f"Reaction-{classification}" if classification else "Reaction"
    else:
        label_prefix = node_type

    # Append parent label part for hierarchy uniqueness
    label = f"{label_prefix}-{parent_label}" if parent_label else label_prefix

    # Create SynthesisTreeNode, storing original data for potential reference
    node = SynthesisTreeNode(
        label=label,
        data={
            "type": node_type,
            "smiles": smiles, # Store SMILES even if used in label
            "mol_id": metadata.get("molecule_id", ""), # Store mol_id
            "metadata": metadata,
            "original_node_data": node_data # Keep original dict if needed
        }
    )

    # Recursively convert children
    children_data = node_data.get("children", [])
    if children_data:
        for i, child_data in enumerate(children_data):
            # Pass index to ensure unique labels among siblings
            child_node = convert_aizynthfinder_node_to_synthesis_node(
                child_data,
                parent_label=f"{label}-{i}" # More robust parent label part
            )
            node.add_child(child_node)

    return node

def convert_aizynthfinder_tree_to_synthesis_tree(
    tree_data: Dict[str, Any],
    tree_idx: Union[int, str] = 0
) -> SynthesisTree:
    """Converts an AiZynthFinder tree dictionary to a SynthesisTree.

    Args:
        tree_data: Tree data dictionary from AiZynthFinder.
        tree_idx: Index or identifier for the tree (used for root label uniqueness).

    Returns:
        The corresponding SynthesisTree.
    """
    root_node = convert_aizynthfinder_node_to_synthesis_node(
        tree_data, parent_label=f"tree{tree_idx}"
    )
    tree = SynthesisTree(root_node)
    # Optionally store original tree data on the SynthesisTree object
    # setattr(tree, 'original_tree_data', tree_data)
    return tree

# --- Data Loading Function ---

def _load_trees_from_json_gz(
    filepath: str
) -> Tuple[List[SynthesisTree], List[Tuple[int, int, Dict[str, Any]]]]:
    """Loads and converts AiZynthFinder trees from a .json.gz file.

    Handles different file structures (pandas orient='table' vs. direct list/dict).

    Args:
        filepath: Path to the input AiZynthFinder output.json.gz file.

    Returns:
        A tuple containing:
            - list[SynthesisTree]: List of converted SynthesisTree objects.
            - list[tuple]: List of original data tuples:
                           (molecule_index, tree_index_within_molecule, tree_dict).
    """
    synthesis_trees: List[SynthesisTree] = []
    original_trees_with_indices: List[Tuple[int, int, Dict[str, Any]]] = []

    # Try loading with pandas (handles 'table' orientation)
    try:
        print(f"Attempting to read AiZynthFinder data from {filepath} using pandas...")
        data = pd.read_json(filepath, orient="table")
        if not hasattr(data, "trees") or data.trees.empty:
            print("Pandas loaded the file, but no 'trees' column found or it's empty.")
            return [], []
        raw_tree_data = data.trees

    except ValueError as e_pandas:
        print(f"Pandas failed to load ({e_pandas}). Trying direct JSON/gzip loading...")
        try:
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                raw_data = json.load(f)

            # Determine structure: dict with 'trees' or just a list?
            if isinstance(raw_data, dict) and "trees" in raw_data:
                 if isinstance(raw_data["trees"], list):
                     raw_tree_data = raw_data["trees"]
                 else:
                     print("Error: 'trees' key found but does not contain a list.")
                     return [], []
            elif isinstance(raw_data, list):
                 print("Warning: Input file seems to be a list of trees (assuming single molecule). Wrapping in list.")
                 raw_tree_data = [raw_data]
            else:
                 print("Error: Unrecognized JSON structure in input file.")
                 return [], []

        except Exception as e_gzip:
            print(f"Fatal Error: Failed to load {filepath} with both pandas and gzip/json: {e_gzip}")
            return [], [] # Return empty lists on fatal error

    # --- Process the loaded tree data --- 
    total_trees = 0
    try:
        # Calculate total number of trees across all molecules for progress bar
        total_trees = sum(len(mol_trees) for mol_trees in raw_tree_data if isinstance(mol_trees, list))
    except TypeError:
        print("Error: Could not determine the total number of trees. Check input format.")
        return [], []

    if total_trees == 0:
        print("No synthesis trees found in the loaded data.")
        return [], []

    print(f"Found {total_trees} synthesis trees to process.")
    pbar = tqdm(total=total_trees, desc="Converting AiZynth trees")

    molecule_index = 0
    for mol_tree_list in raw_tree_data:
        if not isinstance(mol_tree_list, list):
            print(f"Warning: Skipping item at molecule index {molecule_index}, expected a list of trees, got {type(mol_tree_list)}.")
            molecule_index += 1
            continue

        for tree_index, tree_dict in enumerate(mol_tree_list):
            if not isinstance(tree_dict, dict):
                 print(f"Warning: Skipping tree at index {tree_index} for molecule {molecule_index}, expected dict, got {type(tree_dict)}.")
                 pbar.update(1) # Still count towards progress
                 continue
            try:
                # Create a unique ID combining molecule and tree index
                unique_tree_id = f"{molecule_index}_{tree_index}"
                synthesis_tree = convert_aizynthfinder_tree_to_synthesis_tree(
                    tree_dict, tree_idx=unique_tree_id
                )
                synthesis_trees.append(synthesis_tree)
                original_trees_with_indices.append(
                    (molecule_index, tree_index, tree_dict)
                )
            except Exception as e_conv:
                print(f"Error converting tree {molecule_index}_{tree_index}: {e_conv}")
                # Optionally log the problematic tree_dict
            finally:
                 pbar.update(1)

        molecule_index += 1 # Increment molecule index after processing its list

    pbar.close()
    return synthesis_trees, original_trees_with_indices

# --- Image Generation Functions ---

def create_aizynth_tree_image(
    tree_data: Dict[str, Any],
    width: Optional[int] = None,
    height: Optional[int] = None
) -> Optional[Image.Image]:
    """Creates a PIL Image of the reaction tree using AiZynthFinder drawing.

    Args:
        tree_data: Tree data dictionary from AiZynthFinder.
        width: Target width for resizing (optional).
        height: Target height for resizing (optional). If only one is given,
                aspect ratio might not be preserved depending on PIL's resize.

    Returns:
        A PIL Image object, or None if image generation fails.
    """
    try:
        reaction_tree = ReactionTree.from_dict(tree_data)
        img = reaction_tree.to_image()

        # Resize only if both width and height are provided
        if width and height and img:
            # Use LANCZOS for high-quality downsampling
            img = img.resize((width, height), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        print(f"Error creating tree image using AiZynthFinder: {e}")
        return None

def create_rdkit_molecule_image(
    smiles: str,
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT
) -> Optional[Image.Image]:
    """Generates a PIL Image from a SMILES string using RDKit.

    Args:
        smiles: The SMILES string of the molecule.
        width: Target image width.
        height: Target image height.

    Returns:
        A PIL Image object, or None if generation fails or SMILES is invalid.
    """
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # print(f"Warning: RDKit could not parse SMILES: {smiles}")
            return None
        # Generate image directly into PIL Image object
        img = Draw.MolToImage(mol, size=(width, height))
        return img
    except Exception as e:
        print(f"Error generating RDKit image for SMILES {smiles}: {e}")
        return None

def image_to_base64_data_url(img: Optional[Image.Image], format: str = "PNG") -> str:
    """Converts a PIL Image object to a base64 encoded data URL.

    Args:
        img: The PIL Image object, or None.
        format: The image format (e.g., "PNG", "JPEG").

    Returns:
        A base64 encoded data URL string (e.g., "data:image/png;base64,...").
        Returns an empty string if conversion fails or image is None.
    """
    if not img:
        return ""
    try:
        buffered = io.BytesIO()
        img.save(buffered, format=format)
        img_bytes = buffered.getvalue()
        img_str = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/{format.lower()};base64,{img_str}"
    except Exception as e:
        print(f"Error converting image to base64 data URL: {e}")
        return ""

# --- Faerun Data Preparation Function ---

def prepare_faerun_data(
    trees: List[SynthesisTree],
    original_trees_data: List[Tuple[int, int, Dict[str, Any]]],
) -> Tuple[List[int], List[str], Dict[int, str]]:
    """Processes tree data to generate labels and groupings for Faerun.

    Args:
        trees: List of SynthesisTree objects used for layout.
        original_trees_data: List of original (mol_idx, tree_idx, tree_dict) tuples.

    Returns:
        A tuple containing:
            - molecule_numeric_ids: List assigning a numeric group ID to each tree.
            - tooltip_labels: List of base64 image data URLs (trees) for tooltips.
            - mol_id_to_legend_html: Dict mapping group ID to HTML for legend
                                     (molecule image or SMILES).
    """
    print(f"Processing {len(trees)} trees for Faerun visualization data...")
    molecule_numeric_ids: List[int] = []
    tooltip_labels: List[str] = []
    mol_key_to_id: Dict[str, int] = {} # Map molecule identifier (SMILES/label) to numeric ID
    mol_id_to_legend_html: Dict[int, str] = {}
    next_mol_id = 0

    # Use tqdm for progress visualization
    pbar_viz = tqdm(
        zip(trees, original_trees_data),
        total=len(trees),
        desc="Generating Faerun data"
    )

    for idx, (tree, orig_data) in enumerate(pbar_viz):
        _mol_idx, _tree_idx, tree_dict = orig_data
        root_node = tree.root

        # --- Determine Molecule Key (Group Identifier) ---
        root_smiles = root_node.data.get("smiles", "") if root_node else ""
        if root_smiles:
            mol_key = root_smiles
        else:
            # Fallback if root SMILES is missing
            label_part = root_node.label.split("-tree")[0] if root_node and root_node.label else f"unknown_mol_{idx}"
            mol_key = label_part
            print(f"Warning: Tree {idx} root node missing SMILES. Using label '{mol_key}' as group key.")

        # --- Assign Numeric Group ID and Create Legend Entry --- 
        if mol_key not in mol_key_to_id:
            assigned_id = next_mol_id
            mol_key_to_id[mol_key] = assigned_id

            # Create legend HTML (try image first, fallback to text)
            mol_img = create_rdkit_molecule_image(root_smiles)
            legend_text = mol_key # Fallback text
            legend_html = image_to_base64_data_url(mol_img)
            if legend_html:
                 # Wrap image in HTML tag for legend - REMOVE max-height/max-width styles
                 legend_html = (
                     f'<img src="{legend_html}" '
                     f'alt="{legend_text}" title="{legend_text}" '
                     # f'style="max-height: 40px; max-width: 100px; vertical-align: middle;">' # REMOVED STYLE
                     f'style="vertical-align: middle;">'
                 )
            else:
                 legend_html = legend_text # Use fallback text if image failed

            mol_id_to_legend_html[assigned_id] = legend_html
            next_mol_id += 1

        current_mol_numeric_id = mol_key_to_id[mol_key]
        molecule_numeric_ids.append(current_mol_numeric_id)

        # --- Generate Tooltip Label (Tree Image) ---
        # Pass original tree dict to generate image
        tree_img = create_aizynth_tree_image(tree_dict) # Generate without specific size for tooltip
        tooltip_label = image_to_base64_data_url(tree_img)
        tooltip_labels.append(tooltip_label)

    print(f"Processed {len(trees)} trees. Found {len(mol_key_to_id)} unique target molecules (groups).")
    return molecule_numeric_ids, tooltip_labels, mol_id_to_legend_html

# --- Faerun Plotting Function ---

def _add_faerun_scatter_layers(
    faerun_plot: Faerun,
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

    # --- Add Invisible Master Scatter Plot (for Edges) ---
    faerun_plot.add_scatter(
        FAERUN_MASTER_PLOT_NAME,
        {
            "x": x_coords,
            "y": y_coords,
            "c": [0] * num_points, # Single dummy category
            "labels": tooltip_labels, # Tooltips show tree image here
        },
        shader="smoothCircle",
        point_scale=0.0, # Invisible points
        max_point_size=0,
        categorical=True,
        colormap=transparent_cmap,
        has_legend=False,
        interactive=False # Not interactive for hover
    )

    # --- Add Invisible Interactive Proxy Layer (for Hover) ---
    faerun_plot.add_scatter(
        FAERUN_PROXY_LAYER_NAME,
        {
            "x": x_coords,
            "y": y_coords,
            "c": [0] * num_points,
            "labels": tree_node_indices # Simple indices, not shown
        },
        shader="smoothCircle",
        point_scale=5.0, # Must have non-zero size for Octree
        max_point_size=1,
        categorical=True,
        colormap=transparent_cmap,
        has_legend=False,
        interactive=True # THIS layer handles hover
    )

    # --- Add Visible Scatter Plots per Molecule Group ---
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
        # Tooltips for group points should still show the individual tree image
        group_tooltip_labels = [tooltip_labels[i] for i in indices]

        scatter_name = f"{FAERUN_GROUP_PLOT_PREFIX}{mol_numeric_id}"
        legend_label_html = mol_id_to_legend_html.get(mol_numeric_id, f"Group {mol_numeric_id}")

        # print(f"    Adding plot: {scatter_name} ({len(indices)} trees)") # Verbose
        faerun_plot.add_scatter(
            scatter_name,
            {
                "x": group_x,
                "y": group_y,
                "c": [group_idx] * len(indices), # Use group index for consistent color mapping
                "labels": group_tooltip_labels, # Tooltip shows tree image
            },
            shader="smoothCircle",
            colormap=group_colors_cmap,
            point_scale=1,
            max_point_size=10,
            has_legend=True,
            legend_labels=[(group_idx, legend_label_html)], # Legend shows target molecule
            categorical=True,
            # Set interactive=True - This layer's labels are used for tooltips
            interactive=True # CHANGED FROM False
        )

def create_faerun_visualization(
    args: argparse.Namespace,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    edges: List[Tuple[int, int]],
    molecule_numeric_ids: List[int],
    tooltip_labels: List[str],
    mol_id_to_legend_html: Dict[int, str]
) -> None:
    """Creates and saves the Faerun interactive plot.

    Args:
        args: Command-line arguments namespace.
        x_coords: X coordinates of the layout.
        y_coords: Y coordinates of the layout.
        edges: List of graph edges (source_idx, target_idx).
        molecule_numeric_ids: List mapping each point to its molecule group ID.
        tooltip_labels: List of base64 tree images for tooltips.
        mol_id_to_legend_html: Dict mapping group ID to legend HTML.
    """
    print("Creating Faerun visualization...")
    faerun_plot = Faerun(clear_color=args.bg_color, view="front", coords=False, title=args.title)

    # Add all scatter layers
    _add_faerun_scatter_layers(
        faerun_plot, x_coords, y_coords, tooltip_labels,
        molecule_numeric_ids, mol_id_to_legend_html
    )

    # --- Add Tree Edges --- 
    if edges:
        faerun_plot.add_tree(
            FAERUN_EDGE_PLOT_NAME,
            {"from": [e[0] for e in edges], "to": [e[1] for e in edges]},
            point_helper=FAERUN_MASTER_PLOT_NAME, # Connect points in the master layer
            color=FAERUN_EDGE_COLOR,
        )
    else:
        print("  No edges found to add to the plot.")

    # --- Save the Plot --- 
    # Use the file_name and path arguments for robust saving
    output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else '.'
    base_name = os.path.basename(args.output)
    output_file_name = f"{base_name}.html"
    output_path = os.path.join(output_dir, output_file_name)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving visualization to {output_path}...")
    try:
        # Pass file_name (without extension) and path separately
        faerun_plot.plot(
            file_name=base_name,
            path=output_dir,
            template=FAERUN_TEMPLATE_NAME # Assuming Faerun finds this itself
        )
        print(f"Visualization complete: {output_path}")
    except Exception as e_plot:
        print(f"ERROR: Faerun plot saving failed: {e_plot}")
        # Consider exiting or providing more details

# --- Main Execution Block ---

def run_retrograph(args: argparse.Namespace) -> None:
    """Performs the main workflow: load, layout, visualize."""

    # 1. Load and Convert Trees
    trees, original_trees_data = _load_trees_from_json_gz(args.input_file)
    if not trees:
        print("No valid synthesis trees found or loaded. Exiting.")
        sys.exit(1)
    print(f"Successfully loaded and converted {len(trees)} trees.")

    # 2. Initialize Layout Generator
    print("Initializing TED layout generator...")
    print(f"  TED threshold={args.ted_threshold}, Force connect={args.force_connect}")
    layout_generator = TEDLayoutGenerator(
        create_mst=True, # Defaulting MST creation
        ted_threshold=args.ted_threshold,
        force_connect=args.force_connect
        # Other parameters could be exposed via args if needed
    )

    # 3. Generate Layout
    print("Generating t-MAP layout (this may take some time)...")
    try:
        embedding = layout_generator.layout(trees)
        print("Layout generation complete.")
    except Exception as e_layout:
        print(f"ERROR: Layout generation failed: {e_layout}")
        sys.exit(1)

    # Extract layout results
    x_coords = embedding.x
    y_coords = embedding.y
    # Ensure edges are tuples of integers
    edges = [(int(s), int(t)) for s, t in zip(embedding.s, embedding.t)]

    # 4. Prepare Data for Faerun
    molecule_numeric_ids, tooltip_labels, mol_id_to_legend_html = prepare_faerun_data(
        trees, original_trees_data
    )

    # 5. Create and Save Faerun Visualization
    create_faerun_visualization(
        args, x_coords, y_coords, edges,
        molecule_numeric_ids, tooltip_labels, mol_id_to_legend_html
    )

def main() -> None:
    """Parses command-line arguments and runs the main retrograph workflow."""
    parser = argparse.ArgumentParser(
        description="Visualize AiZynthFinder trees using TED-based t-MAP and Faerun.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    parser.add_argument(
        "input_file",
        help="Path to AiZynthFinder output JSON file (e.g., output.json.gz)"
    )
    parser.add_argument(
        "--ted-threshold",
        type=float,
        default=DEFAULT_TED_THRESHOLD,
        help="Maximum Tree Edit Distance (TED) for connecting nodes. Lower values enforce stricter similarity."
    )
    parser.add_argument(
        "--force-connect",
        action="store_true",
        help="Force connection between all components, potentially violating the TED threshold."
    )
    parser.add_argument(
        "--title",
        default="",
        help="Title for the Faerun visualization."
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILENAME,
        help="Output file name prefix (without .html extension). Can include relative/absolute path."
    )
    parser.add_argument(
        "--bg-color",
        default=DEFAULT_BG_COLOR,
        help="Background color for the Faerun plot (hex code)."
    )

    args = parser.parse_args()

    # Input file validation
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: '{args.input_file}'")
        sys.exit(1)
    if not args.input_file.endswith((".json", ".json.gz")):
         print(f"Warning: Input file '{args.input_file}' might not be a JSON file.")

    run_retrograph(args)

if __name__ == "__main__":
    main() 