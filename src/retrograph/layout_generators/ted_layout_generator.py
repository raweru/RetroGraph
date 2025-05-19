from typing import Iterable, Callable, List, Any, Dict, Tuple, Union, Set, Optional

import numpy as np

from ..core import TMAPEmbedding, SynthesisTree, SynthesisTreeNode
from ..helpers.ted_utils import compute_ted, compute_pairwise_distances
from .base_layout_generator import BaseLayoutGenerator

try:
    from tmap import ScalingType, Placer, Merger
except ImportError:
    raise ImportError("tmap package is required but could not be imported. Please install it.")

# Constants
DEFAULT_NODE_SIZE = 1 / 65
DEFAULT_TED_THRESHOLD = 5.0

# Helper Functions

def _find_connected_components(n: int, adj_list: List[List[int]]) -> List[List[int]]:
    """Finds connected components using Depth First Search (DFS).

    Args:
        n: Total number of nodes in the graph.
        adj_list: Adjacency list representation of the graph, where adj_list[i]
                  contains the neighbors of node i.

    Returns:
        A list of lists, where each inner list contains the node indices
        belonging to a single connected component.
    """
    visited = [False] * n
    components: List[List[int]] = []

    def _dfs(node: int, current_component: List[int]):
        """Recursive DFS helper."""
        visited[node] = True
        current_component.append(node)
        for neighbor in adj_list[node]:
            if not visited[neighbor]:
                _dfs(neighbor, current_component)

    for i in range(n):
        if not visited[i]:
            component: List[int] = []
            _dfs(i, component)
            components.append(component)

    return components

def _build_adjacency_list(n: int, edge_list: List[Tuple[int, int, float]]) -> List[List[int]]:
    """Builds an adjacency list from an edge list.

    Args:
        n: Total number of nodes.
        edge_list: List of edges as (source, target, weight) tuples.

    Returns:
        Adjacency list representation of the graph.
    """
    adj_list: List[List[int]] = [[] for _ in range(n)]
    for u, v, _ in edge_list:
        if 0 <= u < n and 0 <= v < n:
            adj_list[u].append(v)
            adj_list[v].append(u)
        else:
             # Log or raise error for invalid indices? For now, just print.
             print(f"Warning: Invalid edge index found ({u}, {v}) for n={n}. Skipping.")
    return adj_list

def _compute_minimum_inter_component_edges(
    components: List[List[int]],
    distances: np.ndarray
) -> List[Tuple[int, int, float]]:
    """Finds the single edge with the minimum distance between each pair of components.

    Args:
        components: A list of components (each component is a list of node indices).
        distances: The full n x n matrix of pairwise distances.

    Returns:
        A list of edges (u, v, distance) representing the cheapest connections
        needed to potentially merge all components.
    """
    min_edges: List[Tuple[int, int, float]] = []
    num_components = len(components)

    for i in range(num_components):
        for j in range(i + 1, num_components):
            min_dist_between = float('inf')
            best_edge_between: Optional[Tuple[int, int, float]] = None

            nodes_i = components[i]
            nodes_j = components[j]

            # Find the closest pair of nodes between component i and component j
            # Note: This is O(|Ci| * |Cj|), could be slow for large components.
            for node_i in nodes_i:
                for node_j in nodes_j:
                    dist = distances[node_i, node_j]
                    if not np.isnan(dist) and dist < min_dist_between:
                        min_dist_between = dist
                        best_edge_between = (node_i, node_j, dist)

            if best_edge_between:
                min_edges.append(best_edge_between)
                print(f"  Found potential connection between clusters {i+1} and {j+1} with TED {min_dist_between:.4f}")

    return min_edges

def _compute_mst_for_component(
    component: List[int],
    subgraph_edges: List[Tuple[int, int, float]]
) -> List[Tuple[int, int, float]]:
    """Computes the Minimum Spanning Tree (MST) for a single connected component.

    Uses Prim's algorithm (simple implementation).

    Args:
        component: List of node indices in this component.
        subgraph_edges: List of edges (u, v, weight) where both u and v are in the component.

    Returns:
        List of edges forming the MST for this component.
    """
    if not component:
        return []

    component_nodes = set(component)
    mst_nodes: Set[int] = {component[0]} # Start Prim's from the first node
    mst_edges: List[Tuple[int, int, float]] = []

    while len(mst_nodes) < len(component_nodes):
        best_edge: Optional[Tuple[int, int, float]] = None
        min_weight = float('inf')

        # Find the minimum weight edge connecting a node in the MST to a node outside
        for u, v, w in subgraph_edges:
            # Check if edge crosses the cut (one node in MST, one node not)
            u_in_mst = u in mst_nodes
            v_in_mst = v in mst_nodes
            if u_in_mst != v_in_mst: # XOR condition for crossing the cut
                if w < min_weight:
                    min_weight = w
                    best_edge = (u, v, w)

        if best_edge:
            mst_edges.append(best_edge)
            # Add the node that was outside the MST to the set
            # Using set union handles adding either u or v depending on which was outside
            mst_nodes.add(best_edge[0])
            mst_nodes.add(best_edge[1])
        else:
            # Should not happen if the component subgraph is truly connected
            print(f"Warning: Could not find connecting edge for MST in component. Component size: {len(component)}. Edges considered: {len(subgraph_edges)}. MST nodes found: {len(mst_nodes)}.")
            # As a fallback, return all edges for this subgraph to ensure connectivity
            return subgraph_edges

    return mst_edges


class TEDLayoutGenerator(BaseLayoutGenerator):
    """Generates graph layouts using Tree Edit Distance (TED).

    Measures similarities between synthesis route trees (SynthesisTree objects)
    using TED calculated via the 'apted' package with custom costs.
    It connects trees only if their TED is below a specified threshold, potentially
    resulting in multiple disconnected components (clusters) unless `visualize_all_mode` is True.
    """
    def __init__(
        self,
        create_mst: bool = True,
        fme_iterations: int = 100,
        fme_threads: int = 4,
        fme_precision: int = 4,
        sl_repeats: int = 1,
        sl_extra_scaling_steps: int = 2,
        sl_scaling_min: float = 1.0,
        sl_scaling_max: float = 1.0,
        sl_scaling_type: ScalingType = ScalingType.RelativeToDrawing,
        mmm_repeats: int = 1,
        placer: Placer = Placer.Barycenter,
        merger: Merger = Merger.LocalBiconnected,
        merger_factor: float = 2.0,
        merger_adjustment: int = 0,
        node_size: float = DEFAULT_NODE_SIZE,
        ted_threshold: float = DEFAULT_TED_THRESHOLD,
        visualize_all_mode: bool = False,
        ted_mode: str = "shape"
    ) -> None:
        """
        Initializes the TEDLayoutGenerator.

        Args:
            create_mst: Whether to create a Minimum Spanning Tree (MST).
                - If True and graph is connected:
                  Generates MST for the entire graph.
                - If True and graph is disconnected:
                  Generates separate MSTs within each connected component.
            fme_iterations: Iterations for Fast Multipole Embedder (passed to base).
            fme_threads: Threads for Fast Multipole Embedder (passed to base).
            fme_precision: Precision for Fast Multipole Embedder (passed to base).
            sl_repeats: Repeats for scaling layout (passed to base).
            sl_extra_scaling_steps: Extra scaling steps (passed to base).
            sl_scaling_min: Minimum scaling factor (passed to base).
            sl_scaling_max: Maximum scaling factor (passed to base).
            sl_scaling_type: Scaling type (passed to base).
            mmm_repeats: Repeats for multilevel layout (passed to base).
            placer: Vertex placement strategy (passed to base).
            merger: Vertex merging strategy (passed to base).
            merger_factor: Merger parameter (passed to base).
            merger_adjustment: Merger adjustment (passed to base).
            node_size: Node size in the layout (passed to base).
            ted_threshold: Maximum Tree Edit Distance (TED) allowed for an edge
                to connect two trees. Trees with TED <= threshold are connected.
                Lower values enforce stricter similarity. Default: 5.0.
                (Ignored if `visualize_all_mode` is True for initial graph construction).
            visualize_all_mode: If True, bypasses strict threshold-based clustering
                to generate a global tmap layout of all trees based on their
                pairwise TEDs. Default: False.
            ted_mode: The mode for TED calculation: "shape" (default) or
                      "classification_aware". This determines how node rename costs
                      are handled during TED computation.
        """
        super().__init__(
            create_mst=create_mst,
            fme_iterations=fme_iterations,
            fme_threads=fme_threads,
            fme_precision=fme_precision,
            sl_repeats=sl_repeats,
            sl_extra_scaling_steps=sl_extra_scaling_steps,
            sl_scaling_min=sl_scaling_min,
            sl_scaling_max=sl_scaling_max,
            sl_scaling_type=sl_scaling_type,
            mmm_repeats=mmm_repeats,
            placer=placer,
            merger=merger,
            merger_factor=merger_factor,
            merger_adjustment=merger_adjustment,
            node_size=node_size,
        )

        # Store TED-specific configuration
        self.ted_threshold = ted_threshold
        self.visualize_all_mode = visualize_all_mode
        self.ted_mode = ted_mode
        self._base_create_mst_config = create_mst

    def _compute_ted_distance(
        self,
        tree1: Union[SynthesisTree, SynthesisTreeNode],
        tree2: Union[SynthesisTree, SynthesisTreeNode]
    ) -> float:
        """Computes the Tree Edit Distance (TED) between two trees.

        Args:
            tree1: The first tree.
            tree2: The second tree.

        Returns:
            The computed Tree Edit Distance.
        """
        return compute_ted(tree1, tree2, mode=self.ted_mode)

    def _build_thresholded_edge_list(
        self,
        n: int,
        distances: np.ndarray,
        custom_threshold: Optional[float] = None
    ) -> List[Tuple[int, int, float]]:
        """Creates an edge list including only edges below the TED threshold.

        Args:
            n: Number of trees.
            distances: n x n matrix of pairwise TED distances.
            custom_threshold: If provided, overrides self.ted_threshold for this call.

        Returns:
            List of edges (u, v, weight) where weight <= threshold_to_use.
        """
        threshold_to_use = self.ted_threshold if custom_threshold is None else custom_threshold
        
        edge_list: List[Tuple[int, int, float]] = []
        edges_added = 0
        for i in range(n):
            for j in range(i + 1, n): # Avoid self-loops and duplicate edges
                dist = distances[i, j]
                # Add edge if distance is valid and below or equal to the threshold
                if not np.isnan(dist) and dist <= threshold_to_use:
                    edge_list.append((i, j, dist))
                    edges_added += 1

        # print(f"Created {edges_added} edges meeting TED threshold <= {threshold_to_use:.4f}")
        return edge_list

    def layout_from_edge_list(
        self,
        n: int,
        edge_list: List[Tuple[int, int, float]],
        create_mst: Optional[bool] = None
    ) -> TMAPEmbedding:
        """Generates layout from a pre-defined edge list.

        This override primarily exists to document that the edge list passed
        to the base class method should already incorporate TED thresholding
        and potential component merging logic handled by the `layout` method.

        Args:
            n: Number of nodes.
            edge_list: List of edges (source, target, weight) to use for layout.
            create_mst: Override base class MST setting (usually leave as None).

        Returns:
            The computed TMAPEmbedding.
        """
        # If overrides are not provided, use the values stored during init
        final_create_mst = create_mst if create_mst is not None else self._base_create_mst_config

        # Pass through to parent implementation with the final edge list
        return super().layout_from_edge_list(n, edge_list, final_create_mst)

    def layout(
        self,
        trees: Iterable[Union[SynthesisTree, SynthesisTreeNode]],
        create_mst_override: Optional[bool] = None
    ) -> TMAPEmbedding:
        """
        Generates a layout. If self.visualize_all_mode is True, it aims to create
        a global tmap layout of all trees based on their pairwise TEDs, typically
        underpinned by an MST generated from these comprehensive distances.
        Otherwise, it performs threshold-based clustering and layout.
        """
        # Initialization and Distance Calculation
        tree_list = list(trees)
        n = len(tree_list)
        if n == 0:
            raise ValueError("No trees provided for layout.")

        print(f"Computing pairwise TED distances (mode: {self.ted_mode})...")
        distances = compute_pairwise_distances(tree_list, mode=self.ted_mode)

        final_edge_list: List[Tuple[int, int, float]]
        mst_for_base_super_call: bool

        if self.visualize_all_mode:
            print("Visualize-all mode: using all pairwise TEDs for global tmap layout.")
            final_edge_list = self._build_thresholded_edge_list(n, distances, custom_threshold=float('inf'))
            if create_mst_override is None:
                mst_for_base_super_call = True 
            else:
                mst_for_base_super_call = create_mst_override
        else:
            # Normal clustering and layout mode
            print(f"Building graph with TED threshold <= {self.ted_threshold:.4f}...")
            # Build graph based on instance's configured ted_threshold
            current_edge_list = self._build_thresholded_edge_list(n, distances, custom_threshold=self.ted_threshold)
            
            adj_list = _build_adjacency_list(n, current_edge_list)
            components = _find_connected_components(n, adj_list)
            num_components = len(components)
            print(f"Found {num_components} initial clusters based on TED threshold.")

            final_edge_list = current_edge_list # Start with thresholded edges
            
            # Determine MST policy for this run (respects override, then instance default)
            mst_policy_for_this_run = self._base_create_mst_config if create_mst_override is None else create_mst_override
            mst_for_base_super_call = mst_policy_for_this_run # Assume base will do MST unless we do it component-wise

            # Component-wise MST logic:
            # If MST is desired for this run, AND
            # if the graph based on self.ted_threshold was disconnected
            if mst_policy_for_this_run and num_components > 1:
                print(f"Graph has {num_components} clusters. Computing MST within each cluster...")
                mst_edges_all_components: List[Tuple[int, int, float]] = []
                for i, component_nodes in enumerate(components): # Iterate over component node lists
                    if len(component_nodes) <= 1:
                        continue 
                    # Get edges belonging only to this component from the original thresholded list
                    component_subgraph_edges = [
                        (u, v, w) for u, v, w in current_edge_list 
                        if u in component_nodes and v in component_nodes
                    ]
                    if not component_subgraph_edges and len(component_nodes) > 1:
                         print(f"Warning: Component {i+1} (size {len(component_nodes)}) has no internal edges based on threshold. Skipping MST for this component.")
                         continue
                    component_mst_edges = _compute_mst_for_component(component_nodes, component_subgraph_edges)
                    mst_edges_all_components.extend(component_mst_edges)
                
                final_edge_list = mst_edges_all_components # These MSTs become the graph for layout
                mst_for_base_super_call = False # We've already built the relevant MSTs, base should not redo
                print(f"Generated {len(final_edge_list)} total MST edges across {num_components} clusters.")
            # Else (graph was connected, or MST not desired for disconnected components),
            # final_edge_list is already set (thresholded), and
            # mst_for_base_super_call reflects whether the base should compute an MST on that.
        
        # 6. Generate Final Layout
        print("Generating final layout...")
        # Pass the prepared edge list and MST flag to the base class for tmap layout
        return super().layout_from_edge_list(
            n,
            final_edge_list,
            create_mst=mst_for_base_super_call
        )