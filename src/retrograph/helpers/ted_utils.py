"""
Utility functions for Tree Edit Distance (TED) calculations using the apted package.

This module provides functions to convert between tmap's SynthesisTree format and the
format required by the apted package, as well as utility functions for computing
pairwise distances between trees. It relies on the 'apted' library for the core
distance calculation.
"""

# Standard library imports
from typing import List, Dict, Any, Union, Tuple, Optional

# Third-party imports
import numpy as np

# Local application/library specific imports
from ..core import SynthesisTree, SynthesisTreeNode
from ..config import ted_config # Import TED cost constants

# Attempt to import APTED and set a flag
try:
    from apted import APTED, Config as AptedBaseConfig
    APTED_AVAILABLE = True
except ImportError:
    APTED_AVAILABLE = False
    AptedBaseConfig = object # Define a dummy base class if apted is not available

# --- Constants for TED Cost Configuration ---
# COST_DELETE = 1.0
# COST_INSERT = 1.0
# COST_RENAME_TYPE_MISMATCH = 1.0
# COST_RENAME_REACTION_LEN_MISMATCH = 1.0
# COST_RENAME_REACTION_CLASS_1_DIFF = 0.8
# COST_RENAME_REACTION_CLASS_2_DIFF = 0.5
# COST_RENAME_REACTION_CLASS_3_DIFF = 0.0 # Cost is 0 if only 3rd reaction class number differs
# COST_RENAME_MATCH = 0.0
# DELETED: TED cost constants are now defined in src.retrograph.config.ted_config

# --- Helper Function ---

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
    (mol or reaction) and reaction classification similarity.
    """

    def delete(self, node: Dict[str, Any]) -> float:
        """Calculates the cost of deleting a node."""
        return ted_config.COST_DELETE

    def insert(self, node: Dict[str, Any]) -> float:
        """Calculates the cost of inserting a node."""
        return ted_config.COST_INSERT

    def rename(self, node1: Dict[str, Any], node2: Dict[str, Any]) -> float:
        """Calculates the cost of renaming (substituting) node1 with node2.

        Costs depend on node types and reaction classification similarity.

        Args:
            node1: The first node dictionary (apted format).
            node2: The second node dictionary (apted format).

        Returns:
            The cost of renaming, between 0.0 and 1.0.
        """
        type1 = node1.get("type")
        type2 = node2.get("type")

        if type1 != type2:
            return ted_config.COST_RENAME_TYPE_MISMATCH

        if type1 == "reaction":
            # Extract classifications safely
            classification1 = node1.get("metadata", {}).get("classification", "")
            classification2 = node2.get("metadata", {}).get("classification", "")

            if not classification1 or not classification2:
                # Handle missing classifications - treat as max difference?
                return ted_config.COST_RENAME_TYPE_MISMATCH # Or another appropriate cost

            reaction_nums1 = _get_reaction_class_numbers(classification1)
            reaction_nums2 = _get_reaction_class_numbers(classification2)

            if len(reaction_nums1) != len(reaction_nums2):
                return ted_config.COST_RENAME_REACTION_LEN_MISMATCH

            if not reaction_nums1: # Handle empty lists after parsing
                return ted_config.COST_RENAME_MATCH if not reaction_nums2 else ted_config.COST_RENAME_REACTION_LEN_MISMATCH

            # Compare reaction class numbers element-wise
            diffs = np.not_equal(reaction_nums1, reaction_nums2)
            num_diffs = len(diffs)

            if num_diffs > 0 and diffs[0]:
                return ted_config.COST_RENAME_REACTION_CLASS_1_DIFF
            elif num_diffs > 1 and diffs[1]:
                return ted_config.COST_RENAME_REACTION_CLASS_2_DIFF
            elif num_diffs > 2 and diffs[2]:
                 # Per original logic, cost is 0 if only 3rd differs
                return ted_config.COST_RENAME_REACTION_CLASS_3_DIFF
            else: # No differences found
                return ted_config.COST_RENAME_MATCH

        elif type1 == "mol":
            # Assume renaming cost for identical molecule types is 0
            return ted_config.COST_RENAME_MATCH
        else:
            # Unknown type, assume maximum difference cost
            return ted_config.COST_RENAME_TYPE_MISMATCH


    def children(self, node: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gets the children of a node in apted format."""
        return node.get("children", [])


# --- Core TED Computation Functions ---

def compute_ted_with_apted(tree1: Dict[str, Any], tree2: Dict[str, Any]) -> float:
    """Computes Tree Edit Distance using the APTED algorithm and CustomAptedConfig.

    Args:
        tree1: First tree in apted dictionary format.
        tree2: Second tree in apted dictionary format.

    Returns:
        The Tree Edit Distance between the trees.

    Raises:
        ImportError: If the 'apted' package is not installed.
    """
    if not APTED_AVAILABLE:
        raise ImportError("APTED package is required for Tree Edit Distance calculation. "
                          "Please install it (e.g., 'pip install apted').")

    apted_calculator = APTED(tree1, tree2, CustomAptedConfig())
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
        "children": [] # Initialize children list
    }

    # Recursively convert children if they exist
    if node.children:
        apted_node["children"] = [convert_to_apted_tree(child) for child in node.children]

    return apted_node

def compute_ted(tree1: Union[SynthesisTree, SynthesisTreeNode],
               tree2: Union[SynthesisTree, SynthesisTreeNode]) -> float:
    """Computes the Tree Edit Distance (TED) between two Synthesis Trees/Nodes.

    Uses the 'apted' library with custom costs if available. Falls back to a
    simple node count difference if 'apted' is not installed.

    Args:
        tree1: The first SynthesisTree or SynthesisTreeNode.
        tree2: The second SynthesisTree or SynthesisTreeNode.

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

    # --- Use APTED if available ---
    try:
        apted_tree1 = convert_to_apted_tree(tree1)
        apted_tree2 = convert_to_apted_tree(tree2)
        return compute_ted_with_apted(apted_tree1, apted_tree2)
    except Exception as e:
        # Catch potential errors during conversion or computation if apted is installed
        print(f"Error during apted computation or tree conversion: {e}")
        # Optional: Fallback here too, or re-raise
        raise # Re-raise the exception for now


# --- Batch Processing and KNN ---

def compute_pairwise_distances(trees: List[Union[SynthesisTree, SynthesisTreeNode]]) -> np.ndarray:
    """Computes pairwise TED distances for a list of trees.

    Args:
        trees: A list of SynthesisTree or SynthesisTreeNode objects.

    Returns:
        A numpy array (n x n) containing the pairwise TED distances, where n is
        the number of trees. distances[i, j] is the TED between trees[i] and
        trees[j].
    """
    n = len(trees)
    if n == 0:
        return np.array([]).reshape(0, 0) # Return empty 0x0 array

    distances = np.zeros((n, n), dtype=float) # Initialize distance matrix

    # Calculate upper triangle (including diagonal, although it will be 0)
    for i in range(n):
        for j in range(i, n): # Start j from i
            if i == j:
                distances[i, j] = 0.0 # Distance to self is 0
                continue
            try:
                dist = compute_ted(trees[i], trees[j])
                distances[i, j] = dist
                distances[j, i] = dist # Symmetric matrix
            except Exception as e:
                 print(f"Error computing TED between tree {i} and tree {j}: {e}")
                 # Assign a high value or NaN? For now, let's use NaN
                 distances[i, j] = np.nan
                 distances[j, i] = np.nan


    return distances

# Remove the get_k_nearest_neighbors function entirely
# def get_k_nearest_neighbors(distances: np.ndarray, k: int) -> List[Tuple[int, int, float]]:
#     """Finds the k-nearest neighbors for each node based on a distance matrix.
#     ...
#     """
#     n = distances.shape[0]
#     # ... (rest of function code) ...
#     return edge_list 