import math
from operator import ge
from typing import Any, Callable

import torch.fx as fx

from iree.turbine.kernel._support.indexing import IndexSequence, IndexSymbol
from iree.turbine.kernel.wave.utils.symbol_utils import subs_idxc

from .._support.dtype import i1
from .._support.tracing import CapturedTrace
from ..ops.wave_ops import (
    Add,
    AtomicMin,
    Cumsum,
    CustomOp,
    Extract,
    NewRegister,
    Read,
    Reshape,
    ScanOp,
    ScatterOp,
    SelectOp,
    ShuffleOp,
    get_custom,
)

from .constraints import (
    Constraint,
    HardwareConstraint,
    TilingConstraint,
    WaveConstraint,
    WorkgroupConstraint,
)
from .utils.classes import ShuffleMode
from .utils.graph_utils import DCE


def emit_global_scatter_min(
    out: fx.Node,
    index: fx.Node,
    src: fx.Node,
    graph: fx.Graph,
    dim: int,
) -> fx.Node:
    """
    Naive Approach using global atomics
    """
    read_src = Read(src).add_to_graph(graph)
    read_src.index = get_custom(index).index
    # scanop_result.expanded_dims = get_custom(src).expanded_dims
    # scanop_result.vector_shapes = get_custom(src).vector_shapes

    AtomicMin(read_src, out).add_to_graph(graph)

    return


def decompose_scatter_ops(
    trace: CapturedTrace,
    constraints: list[Constraint],
):
    """
    TODO: docstring
    """

    # Get reducte nodes.
    scatter_nodes = trace.walk(lambda node: isinstance(get_custom(node), ScatterOp))
    if not scatter_nodes:
        return

    # Setup constraints
    # hardware_constraint = next(
    #     c for c in constraints if isinstance(c, HardwareConstraint)
    # )
    # induction_vars = [
    #     c.induction_var for c in constraints if isinstance(c, TilingConstraint)
    # ]

    # wave_constraint_map = {
    #     c.dim: c for c in constraints if isinstance(c, WaveConstraint)
    # }
    # workgroup_constraint_map = {
    #     c.dim: c for c in constraints if isinstance(c, WorkgroupConstraint)
    # }
    # subgroup_size = hardware_constraint.threads_per_wave

    for node in scatter_nodes:
        custom = get_custom(node)
        if not isinstance(custom, Cumsum):
            raise NotImplementedError(f"ScatterOp '{custom}' not supported")

        with custom.graph.inserting_before(custom.fx_node):
            scatter_out, scatter_idx, scatter_src, scatter_dim = node.args
            emit_global_scatter_min(scatter_out, scatter_idx, scatter_src, scatter_dim)
