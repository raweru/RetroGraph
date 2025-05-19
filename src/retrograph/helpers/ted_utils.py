"""
Utility functions for Tree Edit Distance (TED) calculations using the apted package.

This module provides functions to convert between SynthesisTree format and the
format required by the apted package, as well as utility functions for computing
pairwise distances between trees. It relies on the 'apted' library for the core
distance calculation.
"""

from typing import List, Dict, Any, Union, Tuple, Optional

import numpy as np

from ..core import SynthesisTree, SynthesisTreeNode
from ..config import ted_config

try:
    from apted import APTED, Config as AptedBaseConfig
    APTED_AVAILABLE = True
except ImportError:
    APTED_AVAILABLE = False
    AptedBaseConfig = object # Define a dummy base class if apted is not available


def _get_reaction_class_numbers(classification: str) -> List[int]:
    """Splits AiZynthFinder reaction classification string into numbers.

    Takes the numeric part (e.g., "1.2.3") from a string like "1.2.3 Heck Reaction".

    Args:
        classification: The classification string from AiZynthFinder metadata.

    Returns:
        A list of integers representing the reaction class numbers. Returns an
        empty list if parsing fails.
    """
    try:
        # Take only the part before the first space (numeric classes)
        numeric_part = classification.split(" ")[0]
        # Split by dot and convert to integers
        reaction_class_numbers = [int(i) for i in numeric_part.split(".")]
        return reaction_class_numbers
    except (ValueError, IndexError):
        # Handle cases where split or int conversion fails
        return []


# --- Custom APTED Configuration ---

class CustomAptedConfig(AptedBaseConfig):
    """Custom APTED config using specific costs for SynthesisTreeNode comparison.

    Defines costs for delete, insert, and rename operations based on node type
    (mol or reaction) and reaction classification similarity (if mode is 'classification_aware').
    """
    def __init__(self, mode: str = "shape"):
        """
        Args:
            mode (str): TED calculation mode.
                        "shape": Considers only node types for rename cost.
                        "classification_aware": Considers reaction classifications for rename cost.
        """
        if mode not in ["shape", "classification_aware"]:
            raise ValueError(f"Invalid TED mode: {mode}. Must be 'shape' or 'classification_aware'.")
        self.mode = mode
        super().__init__()


    def delete(self, node: Dict[str, Any]) -> float:
        """Calculates the cost of deleting a node."""
        return ted_config.COST_DELETE

    def insert(self, node: Dict[str, Any]) -> float:
        """Calculates the cost of inserting a node."""
        return ted_config.COST_INSERT

    def rename(self, node1: Dict[str, Any], node2: Dict[str, Any]) -> float:
        """Calculates the cost of renaming (substituting) node1 with node2.

        Costs depend on node types and, if mode is 'classification_aware',
        on reaction classification similarity.
        """
        type1 = node1.get("type")
        type2 = node2.get("type")

        if type1 != type2:
            return ted_config.COST_RENAME_TYPE_MISMATCH

        # If types are the same:
        if self.mode == "shape":
            return 0.0 # Identical nodes always have zero cost

        elif self.mode == "classification_aware":
            if type1 == "reaction":
                classification1 = node1.get("metadata", {}).get("classification", "")
                classification2 = node2.get("metadata", {}).get("classification", "")

                if not classification1 or not classification2:
                    # If one or both classifications are missing, consider it a significant difference,
                    # but not as much as a type mismatch if both are reactions.
                    # Or, if only one is missing, it's a mismatch.
                    # If both missing and types are 'reaction', could be COST_RENAME_MATCH.
                    # For simplicity, if either is empty, treat as COST_RENAME_REACTION_LEN_MISMATCH.
                    return ted_config.COST_RENAME_REACTION_LEN_MISMATCH if (classification1 or classification2) else ted_config.COST_RENAME_MATCH


                reaction_nums1 = _get_reaction_class_numbers(classification1)
                reaction_nums2 = _get_reaction_class_numbers(classification2)

                if not reaction_nums1 and not reaction_nums2: # Both parsed to empty (e.g. "Reaction" vs "Reaction")
                    return 0.0 # Identical nodes always have zero cost
                if not reaction_nums1 or not reaction_nums2: # One parsed to empty, other not
                    return ted_config.COST_RENAME_REACTION_LEN_MISMATCH
                if len(reaction_nums1) != len(reaction_nums2): # Different number of class parts
                    return ted_config.COST_RENAME_REACTION_LEN_MISMATCH


                diffs = np.not_equal(reaction_nums1, reaction_nums2)
                num_parts = len(reaction_nums1) # reaction_nums1 and reaction_nums2 have same len here

                if num_parts > 0 and diffs[0]:
                    return ted_config.COST_RENAME_REACTION_CLASS_1_DIFF
                elif num_parts > 1 and diffs[1]:
                    return ted_config.COST_RENAME_REACTION_CLASS_2_DIFF
                elif num_parts > 2 and diffs[2]:
                    return ted_config.COST_RENAME_REACTION_CLASS_3_DIFF
                else: # No differences found or fewer than 3 parts and all matched
                    return 0.0 # Identical nodes always have zero cost

            elif type1 == "mol":
                # Molecules of the same type are considered a match in this mode too
                return 0.0 # Identical nodes always have zero cost
            else: # Unknown type, but types are same (e.g. "foo" == "foo")
                  # This case should ideally not be hit if node types are standardized ('mol', 'reaction')
                return 0.0 # Identical nodes always have zero cost
        else:
             # This path should be unreachable due to __init__ validation
            raise ValueError(f"Internal error: Unhandled TED mode: {self.mode}")


    def children(self, node: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gets the children of a node in apted format."""
        return node.get("children", [])



def compute_ted_with_apted(tree1: Dict[str, Any], tree2: Dict[str, Any], mode: str = "shape") -> float:
    """Computes Tree Edit Distance using the APTED algorithm and CustomAptedConfig.

    Args:
        tree1: First tree in apted dictionary format.
        tree2: Second tree in apted dictionary format.
        mode: TED calculation mode ("shape" or "classification_aware").

    Returns:
        The Tree Edit Distance between the trees.
    """
    if not APTED_AVAILABLE:
        raise ImportError("APTED package is required. Please install it (e.g., 'pip install apted').")

    apted_calculator = APTED(tree1, tree2, CustomAptedConfig(mode=mode))
    return apted_calculator.compute_edit_distance()

def convert_to_apted_tree(tree: Union[SynthesisTree, SynthesisTreeNode]) -> Dict[str, Any]:
    """Converts a SynthesisTree or SynthesisTreeNode to the apted dict format.

    This format is required by the CustomAptedConfig and apted library.

    Args:
        tree: The SynthesisTree or SynthesisTreeNode to convert.

    Returns:
        A dictionary representing the tree in apted format, containing 'type',
        'label', 'metadata', and 'children' keys.
    """
    if isinstance(tree, SynthesisTree):
        node = tree.root # Use the root node if a full tree is provided
    elif isinstance(tree, SynthesisTreeNode):
        node = tree
    else:
        raise TypeError("Input must be a SynthesisTree or SynthesisTreeNode.")

    # Ensure node is not None before proceeding
    if node is None:
         # This case might indicate an empty SynthesisTree
         return {"type": "empty", "label": "empty", "metadata": {}, "children": []}


    # Extract data safely using .get() with defaults
    node_type = node.data.get("type", "mol") # Default to 'mol' if type is missing
    node_label = node.label if node.label is not None else ""
    metadata = node.data.get("metadata", {})

    apted_node: Dict[str, Any] = {
        "type": node_type,
        "label": node_label,
        "metadata": metadata,
        "children": []
    }

    # Recursively convert children if they exist
    if node.children:
        apted_node["children"] = [convert_to_apted_tree(child) for child in node.children]

    return apted_node

def compute_ted(tree1: Union[SynthesisTree, SynthesisTreeNode],
               tree2: Union[SynthesisTree, SynthesisTreeNode],
               mode: str = "shape") -> float:
    """Computes the Tree Edit Distance (TED) between two Synthesis Trees/Nodes.

    Args:
        tree1: The first SynthesisTree or SynthesisTreeNode.
        tree2: The second SynthesisTree or SynthesisTreeNode.
        mode: TED calculation mode ("shape" or "classification_aware").
              Defaults to "shape".

    Returns:
        The computed Tree Edit Distance.

    Raises:
        TypeError: If inputs are not SynthesisTree or SynthesisTreeNode objects.
    """
    if not APTED_AVAILABLE:
        print("Warning: 'apted' package not found. Falling back to node count difference for TED.")
        # Fallback logic (simple node count difference)
        def _count_nodes(start_node: Optional[SynthesisTreeNode]) -> int:
            if start_node is None:
                return 0
            count = 1
            if start_node.children:
                count += sum(_count_nodes(child) for child in start_node.children)
            return count

        # Determine root nodes safely
        root1 = tree1.root if isinstance(tree1, SynthesisTree) else tree1 if isinstance(tree1, SynthesisTreeNode) else None
        root2 = tree2.root if isinstance(tree2, SynthesisTree) else tree2 if isinstance(tree2, SynthesisTreeNode) else None

        if not isinstance(root1, (SynthesisTreeNode, type(None))) or not isinstance(root2, (SynthesisTreeNode, type(None))):
             raise TypeError("Inputs must be SynthesisTree or SynthesisTreeNode objects for fallback calculation.")

        count1 = _count_nodes(root1)
        count2 = _count_nodes(root2)
        return float(abs(count1 - count2)) # Return absolute difference as float

    # Use APTED if available
    try:
        apted_tree1 = convert_to_apted_tree(tree1)
        apted_tree2 = convert_to_apted_tree(tree2)
        return compute_ted_with_apted(apted_tree1, apted_tree2, mode=mode)
    except Exception as e:
        # Log the error with full context
        print(f"Error during apted computation or tree conversion: {e}")
        # Wrap the error with more context and preserve the original error
        raise RuntimeError(f"Failed to compute TED: {str(e)}") from e


# Batch Processing

def compute_pairwise_distances(trees: List[Union[SynthesisTree, SynthesisTreeNode]],
                               mode: str = "shape") -> np.ndarray:
    """Computes pairwise TED distances for a list of trees.

    Args:
        trees: A list of SynthesisTree or SynthesisTreeNode objects.
        mode: TED calculation mode ("shape" or "classification_aware")
              to be passed to compute_ted. Defaults to "shape".

    Returns:
        A numpy array (n x n) containing the pairwise TED distances, where n is
        the number of trees. distances[i, j] is the TED between trees[i] and
        trees[j].
    """
    n = len(trees)
    if n == 0:
        return np.array([]).reshape(0, 0) # Return empty 0x0 array

    distances = np.zeros((n, n), dtype=float) # Initialize distance matrix

    # Calculate upper triangle (including diagonal 0)
    for i in range(n):
        for j in range(i, n): # Start j from i
            if i == j:
                distances[i, j] = 0.0 # Distance to self is 0
                continue
            try:
                dist = compute_ted(trees[i], trees[j], mode=mode)
                distances[i, j] = dist
                distances[j, i] = dist # Symmetric matrix
            except Exception as e:
                 print(f"Error computing TED between tree {i} and tree {j} (mode: {mode}): {e}")
                 # Assign a high value or NaN? For now, let's use NaN
                 distances[i, j] = np.nan
                 distances[j, i] = np.nan


    return distances