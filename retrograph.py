"""
Visualizes AiZynthFinder synthesis route trees using TED-based t-MAP layout.

Loads trees from an AiZynthFinder output JSON file (gzipped), converts them
to SynthesisTree objects, computes a t-MAP layout using Tree Edit Distance (TED)
with TEDLayoutGenerator, and generates an interactive HTML visualization using
Faerun.
"""

import argparse
import sys
from typing import List, Dict, Tuple, Any

from src.retrograph.config import RetrographConfig
from src.retrograph.data.loader import DataLoader
from src.retrograph.layout_generators.ted_layout_generator import TEDLayoutGenerator
from src.retrograph.visualization.faerun_visualizer import FaerunVisualizer
from src.retrograph.utils.logging import logger

def parse_args() -> RetrographConfig:
    """Parse command line arguments and return a RetrographConfig object."""
    parser = argparse.ArgumentParser(
        description="Visualize AiZynthFinder trees using TED-based t-MAP and Faerun.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_file",
        help="Path to AiZynthFinder output JSON file (e.g., output.json.gz)"
    )
    parser.add_argument(
        "--ted-threshold",
        type=float,
        help="Maximum Tree Edit Distance (TED) for connecting nodes. (Not used if --visualize-all is set)."
    )
    parser.add_argument(
        "--ted-mode",
        type=str,
        default="shape",
        choices=["shape", "classification_aware"],
        help="TED calculation mode for clustering or distance calculation. "
             "'shape': Considers only tree structure and node types. "
             "'classification_aware': Considers reaction classifications for finer-grained similarity. "
             "Default: shape."
    )
    parser.add_argument(
        "--visualize-all",
        action="store_true",
        help="Visualize all trees in a global tmap layout without explicit TED threshold-based clustering."
    )
    parser.add_argument(
        "--title",
        default="",
        help="Title for the Faerun visualization."
    )
    parser.add_argument(
        "--output",
        help="Output file name prefix (without .html extension). Can include relative/absolute path."
    )
    parser.add_argument(
        "--bg-color",
        help="Background color for the Faerun plot (hex code)."
    )
    parser.add_argument(
        "--point-scale",
        type=float,
        help="Scale factor for plotted points."
    )
    parser.add_argument(
        "--max-point-size",
        type=float,
        help="Maximum size of plotted points."
    )

    args = parser.parse_args()
    
    # Create config object
    config = RetrographConfig(
        input_file=args.input_file,
        output_file=args.output,
        ted_threshold=args.ted_threshold,
        ted_mode=args.ted_mode,
        visualize_all=args.visualize_all,
        title=args.title,
        bg_color=args.bg_color,
        point_scale=args.point_scale,
        max_point_size=args.max_point_size
    )
    
    # Validate config
    config.validate()
    
    return config

def run_retrograph(config: RetrographConfig) -> None:
    """Performs the main workflow: load, layout, visualize."""
    try:
        # 1. Load and Convert Trees
        trees, original_trees_data = DataLoader.load_trees_from_json_gz(config.input_file)
        if not trees:
            logger.error("No valid synthesis trees found or loaded. Exiting.")
            sys.exit(1)
        logger.info(f"Successfully loaded and converted {len(trees)} trees.")

        # 2. Initialize Layout Generator
        logger.info("Initializing TED layout generator...")
        if config.visualize_all:
            logger.info("  Mode: Visualize all (global tmap layout).")
            logger.info(f"        (TED calculation mode for distances: {config.ted_mode})")
        else:
            logger.info(f"  Mode: Clustering (TED threshold={config.ted_threshold})")
            logger.info(f"        (TED calculation mode for clustering: {config.ted_mode})")
            logger.info(f"  Point Scale: {config.point_scale}, Max Point Size: {config.max_point_size}")

        layout_generator = TEDLayoutGenerator(
            ted_threshold=config.ted_threshold,
            visualize_all_mode=config.visualize_all,
            ted_mode=config.ted_mode
        )

        # 3. Generate Layout
        logger.info("Generating t-MAP layout (this may take some time)...")
        try:
            embedding = layout_generator.layout(trees, create_mst_override=config.create_mst) 
            logger.info("Layout generation complete.")
        except Exception as e_layout:
            logger.error(f"Layout generation failed: {e_layout}")
            sys.exit(1)

        x_coords = embedding.x
        y_coords = embedding.y
        edges = [(int(s), int(t)) for s, t in zip(embedding.s, embedding.t)]

        # 4. Prepare Data for Faerun
        visualizer = FaerunVisualizer(config)
        molecule_numeric_ids, tooltip_labels, mol_id_to_legend_html = visualizer.prepare_faerun_data(
            trees, original_trees_data
        )

        # 5. Create and Save Faerun Visualization
        visualizer.create_visualization(
            x_coords, y_coords, edges,
            molecule_numeric_ids, tooltip_labels, mol_id_to_legend_html
        )

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

def main() -> None:
    """Main entry point."""
    try:
        config = parse_args()
        run_retrograph(config)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 