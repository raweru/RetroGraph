# Standard library imports
from typing import Iterable, Callable, List, Any, Dict, Tuple, Union, Set, Optional

# Third-party imports
import numpy as np
# import tmap as tm # OLD absolute import - Keep tm import for access to its types/enums

# Local application/library specific imports
from ..core import TMAPEmbedding, SynthesisTree, SynthesisTreeNode
from ..helpers.ted_utils import compute_ted, compute_pairwise_distances # Removed get_k_nearest_neighbors as it wasn't used
from .base_layout_generator import BaseLayoutGenerator

# Attempt to import enums directly from _tmap extension
try:
    from tmap import ScalingType, Placer, Merger
except ImportError:
    # Fallback or raise error if _tmap doesn't expose them
    print("Warning: Could not import layout enums from _tmap. Using dummy types.")
    class ScalingType: pass
    class Placer: pass
    class Merger: pass
    ScalingType.RelativeToDrawing = None
    Placer.Barycenter = None
    Merger.LocalBiconnected = None

# --- Constants ---
DEFAULT_NODE_SIZE = 1 / 65
DEFAULT_TED_THRESHOLD = 5.0

# --- Helper Functions (moved from inside layout method) ---

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

# --- TEDLayoutGenerator Class ---

class TEDLayoutGenerator(BaseLayoutGenerator):
    """Generates graph layouts using Tree Edit Distance (TED).

    Measures similarities between synthesis route trees (SynthesisTree objects)
    using TED calculated via the 'apted' package (if available) with custom costs.
    It connects trees only if their TED is below a specified threshold, potentially
    resulting in multiple disconnected components (clusters) unless `force_connect`
    is enabled.
    """
    def __init__(
        self,
        distance_function: Optional[Callable[[Any, Any], float]] = None,
        create_mst: bool = True,
        keep_knn: bool = False,
        kc: int = 10,
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
        force_connect: bool = False,
    ) -> None:
        """Initializes the TEDLayoutGenerator.

        Args:
            distance_function: An optional custom function to compute the distance
                (e.g., TED) between two tree-like objects. If None, the default
                `compute_ted` function from `ted_utils` (using apted) is used.
            create_mst: Whether to create a Minimum Spanning Tree (MST).
                - If True and graph is connected (or `force_connect`=True):
                  Generates MST for the entire graph.
                - If True and graph is disconnected (`force_connect`=False):
                  Generates separate MSTs within each connected component.
            keep_knn: Inherited from BaseLayoutGenerator, but has **no effect**
                in TEDLayoutGenerator as the graph is built from the TED
                threshold, not k-NN. Kept for base class compatibility.
            kc: Number of candidates for nearest neighbor search (passed to base).
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
            force_connect: If True, ensures the final graph is fully connected
                by adding the cheapest edges between components, even if they
                exceed `ted_threshold`. If False, allows disconnected components
                based on the threshold. Default: False.
        """
        super().__init__(
            create_mst=create_mst,
            keep_knn=False,
            kc=kc,
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
        self.distance_function = distance_function
        self.ted_threshold = ted_threshold
        self.force_connect = force_connect
        # Store base class MST flag separately, as we might override it locally
        self._base_create_mst_config = create_mst

    def _compute_ted_distance(
        self,
        tree1: Union[SynthesisTree, SynthesisTreeNode],
        tree2: Union[SynthesisTree, SynthesisTreeNode]
    ) -> float:
        """Computes the Tree Edit Distance (TED) between two trees.

        Uses the configured `distance_function` if provided, otherwise defaults
        to `compute_ted` from `ted_utils`.

        Args:
            tree1: The first tree.
            tree2: The second tree.

        Returns:
            The computed Tree Edit Distance.
        """
        if self.distance_function:
            return self.distance_function(tree1, tree2)
        else:
            # Use the helper function from ted_utils by default
            return compute_ted(tree1, tree2)

    def _build_thresholded_edge_list(
        self,
        n: int,
        distances: np.ndarray
    ) -> List[Tuple[int, int, float]]:
        """Creates an edge list including only edges below the TED threshold.

        Args:
            n: Number of trees.
            distances: n x n matrix of pairwise TED distances.

        Returns:
            List of edges (u, v, weight) where weight <= self.ted_threshold.
        """
        edge_list: List[Tuple[int, int, float]] = []
        edges_added = 0
        for i in range(n):
            for j in range(i + 1, n): # Avoid self-loops and duplicate edges
                dist = distances[i, j]
                # Add edge if distance is valid and below or equal to the threshold
                if not np.isnan(dist) and dist <= self.ted_threshold:
                    edge_list.append((i, j, dist))
                    edges_added += 1

        # print(f"Created {edges_added} edges meeting TED threshold <= {self.ted_threshold:.4f}")
        return edge_list

    def layout_from_edge_list(
        self,
        n: int,
        edge_list: List[Tuple[int, int, float]],
        create_mst: Optional[bool] = None,
        keep_knn: Optional[bool] = None
    ) -> TMAPEmbedding:
        """Generates layout from a pre-defined edge list.

        This override primarily exists to document that the edge list passed
        to the base class method should already incorporate TED thresholding
        and potential component merging logic handled by the `layout` method.

        Args:
            n: Number of nodes.
            edge_list: List of edges (source, target, weight) to use for layout.
            create_mst: Override base class MST setting (usually leave as None).
            keep_knn: Override base class KNN setting. Has **no effect** in
                this generator. Set to False internally.

        Returns:
            The computed TMAPEmbedding.
        """
        # If overrides are not provided, use the values stored during init
        final_create_mst = create_mst if create_mst is not None else self._base_create_mst_config

        # Pass through to parent implementation with the final edge list
        return super().layout_from_edge_list(n, edge_list, final_create_mst, keep_knn=False)

    def layout(
        self,
        trees: Iterable[Union[SynthesisTree, SynthesisTreeNode]],
        create_mst: Optional[bool] = None,
        keep_knn: Optional[bool] = None
    ) -> TMAPEmbedding:
        """Generates a layout for the given trees using Tree Edit Distance (TED).

        Steps:
        1. Compute pairwise TED distances between all trees.
        2. Build an initial graph connecting only trees with TED <= `ted_threshold`.
        3. Find connected components (clusters) in this thresholded graph.
        4. If `force_connect`=True and multiple components exist, add the cheapest
           edges between components to connect the graph.
        5. If `create_mst`=True (from init or override):
            - If graph is connected: Compute MST on the final graph.
            - If graph is disconnected: Compute MST within each component separately.
        6. Generate the final layout using the base class method with the resulting edge list.

        Args:
            trees: An iterable of SynthesisTree or SynthesisTreeNode objects.
            create_mst: Optional override for the `create_mst` setting from `__init__`.
            keep_knn: Optional override for the `keep_knn` setting. Has **no effect**
                in this generator. Kept for compatibility.

        Returns:
            The generated TMAPEmbedding layout.

        Raises:
            ValueError: If the input `trees` iterable is empty.
        """
        # --- 1. Initialization and Distance Calculation ---
        local_create_mst = create_mst if create_mst is not None else self._base_create_mst_config

        tree_list = list(trees)
        n = len(tree_list)
        if n == 0:
            raise ValueError("No trees provided for layout.")

        # Compute pairwise distances using the selected TED function
        # Note: Assumes _compute_ted_distance handles potential errors appropriately
        print("Computing pairwise TED distances...")
        distances = compute_pairwise_distances(tree_list) # Using the refactored helper

        # Optional: Print distance stats (consider using logging)
        # valid_distances = distances[~np.isnan(distances)]
        # if valid_distances.size > 0:
        #     print(f"  TED distance stats (excluding NaN): min={np.nanmin(distances):.4f}, max={np.nanmax(distances):.4f}, mean={np.nanmean(distances):.4f}")
        # else:
        #     print("  No valid TED distances computed.")


        # --- 2. Build Initial Thresholded Graph ---
        print(f"Building initial graph with TED threshold <= {self.ted_threshold:.4f}...")
        thresholded_edge_list = self._build_thresholded_edge_list(n, distances)

        # --- 3. Find Connected Components ---
        adj_list = _build_adjacency_list(n, thresholded_edge_list)
        components = _find_connected_components(n, adj_list)
        num_components = len(components)
        print(f"Found {num_components} initial clusters based on TED threshold.")
        if num_components > 1:
            print("Trees per cluster:")
            for i, comp in enumerate(components):
                print(f"  Cluster {i+1}: {len(comp)} trees")

        # --- 4. Handle Force Connect ---
        final_edge_list = thresholded_edge_list
        is_forced_connection = False
        if self.force_connect and num_components > 1:
            print("Forcing connection between clusters (may violate TED threshold)...")
            min_inter_component_edges = _compute_minimum_inter_component_edges(components, distances)

            # To connect all components, we need to add edges forming an MST on the component graph.
            # A simpler approach (used here) is to just add *all* the minimum connecting edges found.
            # This guarantees connectivity but might add redundant edges if components were close.
            final_edge_list.extend(min_inter_component_edges)
            is_forced_connection = True

            # Verify connectivity after adding edges
            # merged_adj_list = _build_adjacency_list(n, final_edge_list)
            # merged_components = _find_connected_components(n, merged_adj_list)
            # print(f"After forcing connection: {len(merged_components)} components.")

        # --- 5. Handle MST Creation (Component-wise or Global) ---
        base_should_create_mst = local_create_mst # Flag for the final call to base class

        if local_create_mst:
            if num_components > 1 and not is_forced_connection:
                 # Case: MST needed, but graph is disconnected and not force-connected.
                 # We compute MST *within* each component.
                print(f"Graph has {num_components} clusters. Computing MST within each cluster...")
                mst_edges_all_components: List[Tuple[int, int, float]] = []
                for i, component in enumerate(components):
                    if len(component) <= 1:
                        continue # Skip single-node components

                    # Get edges belonging only to this component
                    component_subgraph_edges = [
                        (u, v, w) for u, v, w in thresholded_edge_list
                        if u in component and v in component
                    ]

                    if not component_subgraph_edges and len(component) > 1:
                         print(f"Warning: Component {i+1} (size {len(component)}) has no internal edges based on threshold.")
                         continue # Cannot compute MST without edges

                    component_mst_edges = _compute_mst_for_component(component, component_subgraph_edges)
                    mst_edges_all_components.extend(component_mst_edges)

                final_edge_list = mst_edges_all_components
                # The base layout should NOT try to compute MST again, as we've done it per component.
                base_should_create_mst = False
                print(f"Generated {len(final_edge_list)} total MST edges across {num_components} clusters.")
            else:
                 # Case: MST needed, graph is connected (or forced). Base class handles MST.
                 print("Graph is connected (or connection forced). Base class will compute MST if enabled.")
                 # Keep `base_should_create_mst = True`, base class handles MST.
        else:
             # Case: MST not requested.
             print("MST creation is disabled.")
             base_should_create_mst = False


        # --- 6. Generate Final Layout ---
        print("Generating final layout...")
        # We call the *base* class layout method directly here, providing the
        # final computed edge list and the decision on whether *it* should run MST.
        return super().layout_from_edge_list(
            n,
            final_edge_list,
            create_mst=base_should_create_mst,
            keep_knn=False
        )