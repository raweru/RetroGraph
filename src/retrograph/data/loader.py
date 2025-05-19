"""
Data loading module for RetroGraph.

This module handles loading and converting AiZynthFinder trees.
"""

import gzip
import json
from typing import List, Dict, Tuple, Any, Union
import pandas as pd
from tqdm import tqdm

from ..core.synthesis_tree import SynthesisTree, SynthesisTreeNode
from ..utils.logging import logger

class DataLoader:
    """Class for loading and converting AiZynthFinder trees."""
    
    @staticmethod
    def load_trees_from_json_gz(
        filepath: str
    ) -> Tuple[List[SynthesisTree], List[Tuple[int, int, Dict[str, Any]]]]:
        """Loads and converts AiZynthFinder trees from a .json.gz file.

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
            logger.info(f"Attempting to read AiZynthFinder data from {filepath} using pandas...")
            data = pd.read_json(filepath, orient="table")
            if not hasattr(data, "trees") or data.trees.empty:
                logger.warning("Pandas loaded the file, but no 'trees' column found or it's empty.")
                return [], []
            raw_tree_data = data.trees

        except ValueError as e_pandas:
            logger.info(f"Pandas failed to load ({e_pandas}). Trying direct JSON/gzip loading...")
            try:
                with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                    raw_data = json.load(f)

                # Determine structure: dict with 'trees' or just a list?
                if isinstance(raw_data, dict) and "trees" in raw_data:
                     if isinstance(raw_data["trees"], list):
                         raw_tree_data = raw_data["trees"]
                     else:
                         logger.error("Error: 'trees' key found but does not contain a list.")
                         return [], []
                elif isinstance(raw_data, list):
                     logger.warning("Input file seems to be a list of trees (assuming single molecule). Wrapping in list.")
                     raw_tree_data = [raw_data]
                else:
                     logger.error("Error: Unrecognized JSON structure in input file.")
                     return [], []

            except Exception as e_gzip:
                logger.error(f"Fatal Error: Failed to load {filepath} with both pandas and gzip/json: {e_gzip}")
                return [], []

        # Process the loaded tree data
        total_trees = 0
        try:
            # Calculate total number of trees across all molecules for progress bar
            total_trees = sum(len(mol_trees) for mol_trees in raw_tree_data if isinstance(mol_trees, list))
        except TypeError:
            logger.error("Error: Could not determine the total number of trees. Check input format.")
            return [], []

        if total_trees == 0:
            logger.warning("No synthesis trees found in the loaded data.")
            return [], []

        logger.info(f"Found {total_trees} synthesis trees to process.")
        pbar = tqdm(total=total_trees, desc="Converting AiZynth trees")

        molecule_index = 0
        for mol_tree_list in raw_tree_data:
            if not isinstance(mol_tree_list, list):
                logger.warning(f"Skipping item at molecule index {molecule_index}, expected a list of trees, got {type(mol_tree_list)}.")
                molecule_index += 1
                continue

            for tree_index, tree_dict in enumerate(mol_tree_list):
                if not isinstance(tree_dict, dict):
                     logger.warning(f"Skipping tree at index {tree_index} for molecule {molecule_index}, expected dict, got {type(tree_dict)}.")
                     pbar.update(1)
                     continue
                try:
                    # Create a unique ID combining molecule and tree index
                    unique_tree_id = f"{molecule_index}_{tree_index}"
                    synthesis_tree = DataLoader._convert_aizynthfinder_tree_to_synthesis_tree(
                        tree_dict, tree_idx=unique_tree_id
                    )
                    synthesis_trees.append(synthesis_tree)
                    original_trees_with_indices.append(
                        (molecule_index, tree_index, tree_dict)
                    )
                except Exception as e_conv:
                    logger.error(f"Error converting tree {molecule_index}_{tree_index}: {e_conv}")
                finally:
                     pbar.update(1)

            molecule_index += 1

        pbar.close()
        return synthesis_trees, original_trees_with_indices

    @staticmethod
    def _convert_aizynthfinder_node_to_synthesis_node(
        node_data: Dict[str, Any],
        parent_label: str = ""
    ) -> SynthesisTreeNode:
        """Recursively converts an AiZynthFinder node dictionary to a SynthesisTreeNode.

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
                "smiles": smiles,
                "mol_id": metadata.get("molecule_id", ""),
                "metadata": metadata,
                "original_node_data": node_data
            }
        )

        # Recursively convert children
        children_data = node_data.get("children", [])
        if children_data:
            for i, child_data in enumerate(children_data):
                child_node = DataLoader._convert_aizynthfinder_node_to_synthesis_node(
                    child_data,
                    parent_label=f"{label}-{i}"
                )
                node.add_child(child_node)

        return node

    @staticmethod
    def _convert_aizynthfinder_tree_to_synthesis_tree(
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
        root_node = DataLoader._convert_aizynthfinder_node_to_synthesis_node(
            tree_data, parent_label=f"tree{tree_idx}"
        )
        return SynthesisTree(root_node) 