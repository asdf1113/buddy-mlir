# RUN: %PYTHON %s

import tempfile
from pathlib import Path

import numpy as np
import torch
from buddy.compiler.graph import (
    Graph,
    NodeType,
    ParallelTemplatePartitionedGraphDriver,
    TensorDType,
    TransformerParallelConfig,
    build_transformer_parallel_plan,
    build_transformer_partition_plan,
)
from buddy.compiler.graph.graph import GraphImporter
from buddy.compiler.graph.operation import (
    AddMMOp,
    AddOp,
    CallOp,
    FuncOp,
    IndexPutOp,
    OutputOp,
    PlaceholderOp,
    ReshapeOp,
)
from buddy.compiler.graph.source_meta import SourceMeta
from buddy.compiler.graph.transformer_partition import (
    CollectiveBoundary,
    CollectiveKind,
    ComputeSegment,
    FeatureAxis,
    GraphValueRef,
    LayoutKind,
    OperandUseRef,
    ParameterRole,
    ParameterShardSpec,
    RankSlice,
    TemplateParallelPlan,
    TensorLayout,
    TransformerParallelPlan,
    ValueLayoutSpec,
)
from buddy.compiler.ops import func, linalg, tosa
from buddy_mlir import ir


def node(cls, name, shape=(2, 4), path=None):
    op = cls()
    op.name = name
    op.tensor_meta = {"shape": shape, "dtype": TensorDType.Float32}
    if path is not None:
        op._source_meta = (
            SourceMeta(
                module_path=path,
                module_class="example.DecoderLayer",
            ),
        )
    return op


def add(graph, op, kind=NodeType.OtherNode):
    graph.add_node(op, kind)
    return op


def bind(parent, child):
    parent.add_children(child.name)
    child.add_parent(parent.name)
    child.add_argument(parent.name)


def connect(parent, child):
    parent.add_children(child.name)
    child.add_parent(parent.name)


# A decode KV cache keeps its global head extent in graph metadata, while the
# TP rank-local storage follows the already-planned sharded update. The same
# local shape must reach the segment input, wrapper declaration, and imported
# subgraph argument/result types.
cache_graph = Graph(
    {**tosa.ops_registry, **linalg.ops_registry, **func.ops_registry},
    "forward_decode_cache",
)
cache_activation = add(
    cache_graph,
    node(PlaceholderOp, "cache_activation", (1, 4)),
    NodeType.InputNode,
)
cache_input = add(
    cache_graph,
    node(PlaceholderOp, "cache_input", (1, 2, 4, 1)),
    NodeType.InputNode,
)
cache_position = add(
    cache_graph,
    node(PlaceholderOp, "cache_position", (1,)),
    NodeType.InputNode,
)
cache_position.tensor_meta["dtype"] = TensorDType.Int64
cache_bias = add(
    cache_graph,
    node(PlaceholderOp, "cache_bias", (2,)),
    NodeType.FakeNode,
)
cache_weight = add(
    cache_graph,
    node(PlaceholderOp, "cache_weight", (4, 2)),
    NodeType.FakeNode,
)
cache_projection = add(
    cache_graph,
    node(
        AddMMOp,
        "cache_projection",
        (1, 2),
        "model.layers.0.self_attn.k_proj",
    ),
)
for operand in (cache_bias, cache_activation, cache_weight):
    bind(operand, cache_projection)
cache_update = add(
    cache_graph,
    node(
        ReshapeOp,
        "cache_update",
        (1, 2, 1, 1),
        "model.layers.0.self_attn.k_proj",
    ),
)
bind(cache_projection, cache_update)
cache_update.add_argument((1, 2, 1, 1))
cache_update._newshape = (1, 2, 1, 1)
cache_store = add(
    cache_graph,
    node(
        IndexPutOp,
        "cache_store",
        (1, 2, 4, 1),
        "model.layers.0.self_attn.k_proj",
    ),
)
for operand in (cache_input, cache_position, cache_update):
    connect(operand, cache_store)
cache_store.add_argument(cache_input.name)
cache_store.add_argument([None, None, cache_position.name, None])
cache_store.add_argument(cache_update.name)
cache_store.add_argument(False)
cache_output = add(cache_graph, node(OutputOp, "cache_output", ()))
bind(cache_store, cache_output)

cache_partition_plan = build_transformer_partition_plan(cache_graph)
cache_parallel_plan = build_transformer_parallel_plan(
    cache_graph,
    cache_partition_plan,
    TransformerParallelConfig(tp_size=2),
)
cache_template = cache_parallel_plan.templates[0]
cache_layouts = {spec.value: spec for spec in cache_template.value_layouts}
cache_input_value = GraphValueRef(cache_input)
cache_update_value = GraphValueRef(cache_update)
cache_store_value = GraphValueRef(cache_store)
expected_cache_shapes = ((1, 1, 4, 1), (1, 1, 4, 1))


def tensor_shape(meta):
    return tuple(meta.shape if hasattr(meta, "shape") else meta["shape"])


assert cache_layouts[cache_input_value].global_shape == (1, 2, 4, 1)
assert cache_layouts[cache_input_value].local_shapes == expected_cache_shapes
assert cache_layouts[cache_input_value].layout == TensorLayout(
    LayoutKind.SHARDED, 1
)
assert cache_layouts[cache_update_value].local_shapes == (
    (1, 1, 1, 1),
    (1, 1, 1, 1),
)
assert cache_layouts[cache_update_value].layout == TensorLayout(
    LayoutKind.SHARDED, 1
)
assert cache_layouts[cache_store_value].local_shapes == expected_cache_shapes
assert cache_layouts[cache_store_value].layout == TensorLayout(
    LayoutKind.SHARDED, 1
)

cache_segment = cache_template.segments[0]
cache_slot = cache_segment.ordered_inputs.index(cache_input_value)
assert cache_store_value in cache_segment.ordered_outputs
for rank in range(2):
    cache_driver = ParallelTemplatePartitionedGraphDriver(
        cache_graph, cache_partition_plan, cache_parallel_plan, rank
    )
    cache_subgraph = cache_driver.build_parallel_template_subgraphs()[0]
    assert tensor_shape(cache_subgraph.inputs[cache_slot].tensor_meta) == (
        1,
        1,
        4,
        1,
    )
    cache_wrapper = cache_driver.construct_parallel_segment_wrappers()[0]
    cache_wrapper_input = next(
        op
        for op in cache_wrapper.body
        if isinstance(op, PlaceholderOp)
        and op.name == f"__wrapper_arg{cache_slot}"
    )
    assert tensor_shape(cache_wrapper_input.tensor_meta) == (1, 1, 4, 1)
    cache_declaration = next(
        op for op in cache_wrapper.body if isinstance(op, FuncOp)
    )
    assert tensor_shape(cache_declaration.args[cache_slot]) == (1, 1, 4, 1)
    cache_runtime_plan = cache_driver.build_parallel_runtime_plan()
    cache_resource = cache_runtime_plan["runtime_inputs"][1]
    assert cache_runtime_plan["resources"][cache_resource]["shape"] == (
        1,
        1,
        4,
        1,
    )
    cache_subgraph.lower_to_top_level_ir()
    cache_mlir = str(cache_subgraph._imported_module)
    assert "tensor<1x1x4x1xf32>" in cache_mlir
    assert not (
        "memref<1x2x4x1xf32>" in cache_mlir
        and "tensor<1x1x4x1xf32>" in cache_mlir
    )


# The implicit parameter-pack path retains its dtype-local sequential offsets.
pack_graph = Graph(func.ops_registry, "implicit_pack")
pack0 = add(
    pack_graph,
    node(PlaceholderOp, "pack0", (2,)),
    NodeType.FakeNode,
)
pack1 = add(
    pack_graph,
    node(PlaceholderOp, "pack1", (3,)),
    NodeType.FakeNode,
)
pack_output = add(pack_graph, node(OutputOp, "output", ()))
pack_output.add_argument(pack0.name)
pack_output.add_argument(pack1.name)
with ir.Location.unknown(ir.Context()):
    implicit_module = GraphImporter(
        pack_graph.body,
        pack_graph.params_shapes,
        pack_graph.inputs_shapes,
        pack_graph._func_name,
        pack_graph._ops_registry,
        True,
    ).import_main_graph()
implicit_text = str(implicit_module)
assert "@implicit_pack(%arg0: memref<5xf32>)" in implicit_text
assert "%arg0[0] [2] [1]" in implicit_text
assert "%arg0[2] [3] [1]" in implicit_text


# Two structurally identical layers share a two-segment reusable template. Each
# layer uses one parameter in both of its segments.
graph = Graph(
    {**tosa.ops_registry, **func.ops_registry},
    "forward_decode",
)
tokens = add(graph, node(PlaceholderOp, "tokens"), NodeType.InputNode)
weights = [
    add(
        graph,
        node(PlaceholderOp, f"weight{index}"),
        NodeType.FakeNode,
    )
    for index in range(2)
]
previous = tokens
layer_nodes = []
for layer_index, weight in enumerate(weights):
    first = add(
        graph,
        node(
            AddOp,
            f"layer{layer_index}_first",
            path=f"model.layers.{layer_index}.self_attn.q_proj",
        ),
    )
    bind(previous, first)
    bind(weight, first)
    second = add(
        graph,
        node(
            AddOp,
            f"layer{layer_index}_second",
            path=f"model.layers.{layer_index}.mlp.down_proj",
        ),
    )
    bind(first, second)
    bind(weight, second)
    layer_nodes.append((first, second))
    previous = second
output = add(graph, node(OutputOp, "output", ()))
bind(previous, output)

partition_plan = build_transformer_partition_plan(graph)
assert len(partition_plan.templates) == 1
unit = partition_plan.templates[0]
assert len(unit.instances) == 2
data_value, parameter_value = (
    item.value for item in unit.representative.interface.ordered_inputs
)
first, second = unit.representative.nodes
first_value = GraphValueRef(first)
second_value = GraphValueRef(second)
local_shapes = ((2, 2), (2, 2))
value_layouts = tuple(
    ValueLayoutSpec(
        value,
        (2, 4),
        local_shapes,
        TensorLayout(LayoutKind.SHARDED, 1),
    )
    for value in (
        data_value,
        parameter_value,
        first_value,
        second_value,
    )
)
parameter_shard = ParameterShardSpec(
    parameter=parameter_value,
    global_parameter_index=0,
    template_parameter_slot=0,
    role=ParameterRole.Q_WEIGHT,
    semantic_axis=FeatureAxis.OUTPUT_FEATURE,
    consumer_use=OperandUseRef(first, ("args", 1)),
    consumer_shard_axis=1,
    storage_shard_axis=1,
    storage_to_consumer_permutation=(0, 1),
    global_shape=(2, 4),
    rank_slices=(RankSlice(0, 0, 2), RankSlice(1, 2, 2)),
)
template_plan = TemplateParallelPlan(
    template_id=unit.template_id,
    parameter_shards=(parameter_shard,),
    value_layouts=value_layouts,
    op_rewrites=(),
    collectives=(),
    segments=(
        ComputeSegment(
            0,
            (first,),
            (data_value, parameter_value),
            (first_value,),
        ),
        ComputeSegment(
            1,
            (second,),
            (first_value, parameter_value),
            (second_value,),
        ),
    ),
)
parallel_plan = TransformerParallelPlan(
    graph_name=graph._func_name,
    world_size=2,
    templates=(template_plan,),
)
driver = ParallelTemplatePartitionedGraphDriver(
    graph, partition_plan, parallel_plan, rank=0
)
wrappers = driver.construct_parallel_segment_wrappers()
assert [wrapper._func_name for wrapper in wrappers] == [
    "forward_decode_layer0_seg0",
    "forward_decode_layer0_seg1",
    "forward_decode_layer1_seg0",
    "forward_decode_layer1_seg1",
]

layer0_call = next(
    op
    for op in driver._parallel_segment_wrappers[(0, 0)].body
    if isinstance(op, CallOp)
)
layer1_call = next(
    op
    for op in driver._parallel_segment_wrappers[(1, 0)].body
    if isinstance(op, CallOp)
)
assert layer0_call.call_func_name == layer1_call.call_func_name
assert layer0_call.call_func_name == driver.parallel_template_symbol(
    unit.template_id, 0
)
assert driver._wrapper_parameter_bindings[(0, 0)] == {"weight0": 0}
assert driver._wrapper_parameter_bindings[(1, 0)] == {"weight1": 1}

# The shard spec's representative global index is not reused for layer 1.
# Both layer-1 segments bind parameter 1, and its rank-global entry is unique.
assert driver._wrapper_parameter_bindings[(1, 0)]["weight1"] == 1
assert driver._wrapper_parameter_bindings[(1, 1)]["weight1"] == 1
assert list(driver._rank_parameter_layout) == [0, 1]
assert driver._rank_parameter_layout[0] == {
    "dtype": TensorDType.Float32,
    "shape": (2, 2),
    "offset": 0,
    "numel": 4,
}
assert driver._rank_parameter_layout[1] == {
    "dtype": TensorDType.Float32,
    "shape": (2, 2),
    "offset": 4,
    "numel": 4,
}
for key in ((1, 0), (1, 1)):
    parameter_index = driver._wrapper_parameter_bindings[key]["weight1"]
    assert driver._rank_parameter_layout[parameter_index]["offset"] == 4

# An individual wrapper contains only its used parameter placeholder, while its
# explicit offset addresses that parameter in the shared rank-global pack.
wrapper = driver._parallel_segment_wrappers[(1, 1)]
wrapper.lower_to_top_level_ir(
    do_param_pack=True,
    param_pack_sizes={TensorDType.Float32: 8},
    param_pack_offsets={"weight1": 4},
)
explicit_text = str(wrapper._imported_module)
assert "@forward_decode_layer1_seg1(%arg0: memref<8xf32>" in explicit_text
assert "%arg0[4] [4] [1]" in explicit_text
assert "weight0" not in driver._wrapper_parameter_bindings[(1, 1)]


# Stage 4A resolves every actual instance in graph order. Its dispatch argument
# order is the packed-parameter ABI order: used typed packs, runtime inputs,
# then output buffers.
opt_in_driver = ParallelTemplatePartitionedGraphDriver(
    graph, partition_plan, parallel_plan, rank=0
)
assert opt_in_driver._parallel_segment_wrappers == {}
assert opt_in_driver._rank_parameter_layout == {}
runtime_plan = opt_in_driver.build_parallel_runtime_plan()
assert runtime_plan == opt_in_driver.build_parallel_runtime_plan()
dispatches = [
    operation
    for operation in runtime_plan["operations"]
    if operation["kind"] == "dispatch"
]
assert [dispatch["wrapper"] for dispatch in dispatches] == [
    "forward_decode_layer0_seg0",
    "forward_decode_layer0_seg1",
    "forward_decode_layer1_seg0",
    "forward_decode_layer1_seg1",
]
for dispatch in dispatches:
    assert dispatch["arguments"] == (
        dispatch["parameter_packs"] + dispatch["inputs"] + dispatch["outputs"]
    )

# Segment and actual-instance data flow uses the exact same resolved resource.
assert dispatches[0]["outputs"][0] == dispatches[1]["inputs"][0]
assert dispatches[1]["outputs"][0] == dispatches[2]["inputs"][0]
assert dispatches[2]["outputs"][0] == dispatches[3]["inputs"][0]
assert runtime_plan["runtime_inputs"] == dispatches[0]["inputs"]
assert runtime_plan["runtime_outputs"] == dispatches[-1]["outputs"]

# All wrappers use the one rank-global f32 pack, but retain only their concrete
# parameter binding and its existing dtype-local offset.
pack_resource = runtime_plan["parameter_packs"][0]["resource"]
assert {dispatch["parameter_packs"] for dispatch in dispatches} == {
    (pack_resource,)
}
assert dispatches[0]["parameter_bindings"] == (
    {
        "name": "weight0",
        "parameter_index": 0,
        "pack": pack_resource,
        "dtype": "float32",
        "shape": (2, 2),
        "offset": 0,
        "numel": 4,
    },
)
assert dispatches[2]["parameter_bindings"] == (
    {
        "name": "weight1",
        "parameter_index": 1,
        "pack": pack_resource,
        "dtype": "float32",
        "shape": (2, 2),
        "offset": 4,
        "numel": 4,
    },
)

# The two first-segment values do not overlap and reuse one compatible slot.
# Adjacent values overlap at the consuming dispatch and cannot alias.
runtime_values = runtime_plan["values"]
layer0_first_resource = runtime_values["instance0_segment0_result0"]["resource"]
layer0_second_resource = runtime_values["instance0_segment1_result0"][
    "resource"
]
layer1_first_resource = runtime_values["instance1_segment0_result0"]["resource"]
assert layer0_first_resource == layer1_first_resource
assert layer0_first_resource != layer0_second_resource


# Runtime-plan outputs preserve OutputOp order by default and use the same
# explicit post-resolution remap convention as combined template graphs.
saved_output_args = list(output.args)
saved_output_indices = list(output._args_index)
output._arguments = [layer_nodes[1][1].name, layer_nodes[0][1].name]
output._args_index = [0, 0]
output_order_driver = ParallelTemplatePartitionedGraphDriver(
    graph, partition_plan, parallel_plan, rank=0
)
semantic_output_plan = output_order_driver.build_parallel_runtime_plan()
assert semantic_output_plan == output_order_driver.build_parallel_runtime_plan()
semantic_dispatches = [
    operation
    for operation in semantic_output_plan["operations"]
    if operation["kind"] == "dispatch"
]
assert semantic_dispatches[3]["outputs"] == ("output0",)
assert semantic_dispatches[1]["outputs"] == ("output1",)
assert semantic_output_plan["runtime_outputs"] == ("output0", "output1")

remapped_output_plan = output_order_driver.build_parallel_runtime_plan(
    output_remap=[1, 0]
)
assert remapped_output_plan == output_order_driver.build_parallel_runtime_plan(
    output_remap=[1, 0]
)
remapped_dispatches = [
    operation
    for operation in remapped_output_plan["operations"]
    if operation["kind"] == "dispatch"
]
assert remapped_dispatches[1]["outputs"] == ("output0",)
assert remapped_dispatches[3]["outputs"] == ("output1",)
assert remapped_output_plan["runtime_outputs"] == ("output0", "output1")

for invalid_remap, message in (
    ([0], "output_remap length"),
    ([0, 2], "invalid output index"),
):
    try:
        output_order_driver.build_parallel_runtime_plan(
            output_remap=invalid_remap
        )
    except ValueError as error:
        assert message in str(error)
    else:
        raise AssertionError(f"expected ValueError containing {message!r}")
output._arguments = saved_output_args
output._args_index = saved_output_indices


def build_collective_runtime_plan(
    kind,
    global_shape,
    source_shapes,
    source_layout,
    target_shapes,
    target_layout,
):
    layouts = []
    for spec in value_layouts:
        if spec.value == first_value:
            layouts.append(
                ValueLayoutSpec(
                    spec.value,
                    global_shape,
                    source_shapes,
                    source_layout,
                )
            )
        elif spec.value == second_value:
            layouts.append(
                ValueLayoutSpec(
                    spec.value,
                    global_shape,
                    target_shapes,
                    target_layout,
                )
            )
        else:
            layouts.append(spec)
    boundary = CollectiveBoundary(
        producer=first_value,
        consumers=(OperandUseRef(second, ("args", 0)),),
        kind=kind,
        target_layout=target_layout,
        target_local_shapes=target_shapes,
    )
    collective_template = TemplateParallelPlan(
        template_id=unit.template_id,
        parameter_shards=(parameter_shard,),
        value_layouts=tuple(layouts),
        op_rewrites=(),
        collectives=(boundary,),
        segments=template_plan.segments,
    )
    collective_plan = TransformerParallelPlan(
        graph_name=graph._func_name,
        world_size=2,
        templates=(collective_template,),
    )
    return ParallelTemplatePartitionedGraphDriver(
        graph, partition_plan, collective_plan, rank=0
    ).build_parallel_runtime_plan()


# AllReduce is inserted at the consumer ingress and is naturally in-place.
replicated_shapes = ((2, 4), (2, 4))
all_reduce_plan = build_collective_runtime_plan(
    CollectiveKind.ALL_REDUCE,
    (2, 4),
    replicated_shapes,
    TensorLayout(LayoutKind.PARTIAL),
    replicated_shapes,
    TensorLayout(LayoutKind.REPLICATED),
)
first_three = all_reduce_plan["operations"][:3]
assert [operation["kind"] for operation in first_three] == [
    "dispatch",
    "collective",
    "dispatch",
]
all_reduce = first_three[1]
assert all_reduce["collective"] == "all_reduce"
assert all_reduce["reduction"] == "sum"
assert all_reduce["input"] == first_three[0]["outputs"][0]
assert all_reduce["input"] == all_reduce["output"]
assert all_reduce["output"] == first_three[2]["inputs"][0]

# AllGatherV derives flattened receive counts and prefix-sum displacements from
# every source rank-local shape. Its result is a distinct resource.
unequal_source_shapes = ((3, 4), (1, 4))
gather_target_shapes = ((4, 4), (4, 4))
all_gather_plan = build_collective_runtime_plan(
    CollectiveKind.ALL_GATHERV,
    (4, 4),
    unequal_source_shapes,
    TensorLayout(LayoutKind.SHARDED, 0),
    gather_target_shapes,
    TensorLayout(LayoutKind.REPLICATED),
)
all_gather_ops = all_gather_plan["operations"][:3]
all_gather = all_gather_ops[1]
assert all_gather["collective"] == "all_gatherv"
assert all_gather["recv_counts"] == (12, 4)
assert all_gather["displacements"] == (0, 12)
assert all_gather["input"] == all_gather_ops[0]["outputs"][0]
assert all_gather["input"] != all_gather["output"]
assert all_gather["output"] == all_gather_ops[2]["inputs"][0]

# ReduceScatter derives its flattened counts from target_local_shapes and is
# likewise out-of-place with sum reduction.
scatter_source_shapes = ((4, 4), (4, 4))
scatter_target_shapes = ((3, 4), (1, 4))
reduce_scatter_plan = build_collective_runtime_plan(
    CollectiveKind.REDUCE_SCATTER,
    (4, 4),
    scatter_source_shapes,
    TensorLayout(LayoutKind.PARTIAL),
    scatter_target_shapes,
    TensorLayout(LayoutKind.SHARDED, 0),
)
reduce_scatter_ops = reduce_scatter_plan["operations"][:3]
reduce_scatter = reduce_scatter_ops[1]
assert reduce_scatter["collective"] == "reduce_scatter"
assert reduce_scatter["recv_counts"] == (12, 4)
assert reduce_scatter["reduction"] == "sum"
assert reduce_scatter["input"] == reduce_scatter_ops[0]["outputs"][0]
assert reduce_scatter["input"] != reduce_scatter["output"]
assert reduce_scatter["output"] == reduce_scatter_ops[2]["inputs"][0]

# A row-major axis-1 shard with multiple outer blocks cannot be represented by
# one direct flat rank block in the current RAX collective ABI.
try:
    build_collective_runtime_plan(
        CollectiveKind.ALL_GATHERV,
        (2, 4),
        ((2, 3), (2, 1)),
        TensorLayout(LayoutKind.SHARDED, 1),
        ((2, 4), (2, 4)),
        TensorLayout(LayoutKind.REPLICATED),
    )
except ValueError as error:
    assert "current flat RAX collective cannot represent" in str(error)
else:
    raise AssertionError("expected unsupported flat shard layout rejection")


# Rank-global packs follow the Stage 3A offsets, shard on the planned storage
# axis, and contain shared parameters only once despite their use by two
# wrappers per layer.
params = [
    torch.arange(8, dtype=torch.float32).reshape(2, 4),
    torch.arange(100, 108, dtype=torch.float32).reshape(2, 4),
]
with tempfile.TemporaryDirectory() as temporary_directory:
    output_root = Path(temporary_directory)
    for rank in range(2):
        rank_output = output_root / f"sharded_rank{rank}"
        rank_driver = ParallelTemplatePartitionedGraphDriver(
            graph, partition_plan, parallel_plan, rank=rank
        )
        rank_driver.build_rank_parameter_pack(params, rank_output)
        pack_files = list(rank_output.glob("*.data"))
        assert [path.name for path in pack_files] == [
            f"rank{rank}_params_float32.data"
        ]
        packed = np.fromfile(pack_files[0], dtype=np.float32)
        rank_slice = parameter_shard.rank_slices[rank]
        expected = np.concatenate(
            [
                tensor[
                    :,
                    rank_slice.offset : rank_slice.offset + rank_slice.size,
                ]
                .numpy()
                .reshape(-1)
                for tensor in params
            ]
        )
        np.testing.assert_array_equal(packed, expected)
        assert packed.size == sum(
            entry["numel"]
            for entry in rank_driver._rank_parameter_layout.values()
        )
        assert [
            entry["offset"]
            for entry in rank_driver._rank_parameter_layout.values()
        ] == [0, 4]
        assert (
            sum(
                parameter_index in bindings.values()
                for bindings in rank_driver._wrapper_parameter_bindings.values()
                for parameter_index in (0, 1)
            )
            == 4
        )


# With no shard spec, each concrete parameter is replicated unchanged on both
# ranks. The parameter value layout is likewise rank-global for this plan.
replicated_layouts = tuple(
    ValueLayoutSpec(
        spec.value,
        spec.global_shape,
        ((2, 4), (2, 4)),
        TensorLayout(LayoutKind.REPLICATED),
    )
    for spec in value_layouts
)
replicated_template_plan = TemplateParallelPlan(
    template_id=unit.template_id,
    parameter_shards=(),
    value_layouts=replicated_layouts,
    op_rewrites=(),
    collectives=(),
    segments=template_plan.segments,
)
replicated_parallel_plan = TransformerParallelPlan(
    graph_name=graph._func_name,
    world_size=2,
    templates=(replicated_template_plan,),
)
with tempfile.TemporaryDirectory() as temporary_directory:
    output_root = Path(temporary_directory)
    for rank in range(2):
        rank_output = output_root / f"replicated_rank{rank}"
        rank_driver = ParallelTemplatePartitionedGraphDriver(
            graph, partition_plan, replicated_parallel_plan, rank=rank
        )
        rank_driver.build_rank_parameter_pack(params, rank_output)
        packed = np.fromfile(
            rank_output / f"rank{rank}_params_float32.data",
            dtype=np.float32,
        )
        np.testing.assert_array_equal(packed[:8], params[0].numpy().reshape(-1))
        np.testing.assert_array_equal(packed[8:], params[1].numpy().reshape(-1))


# Mixed dtype packs stream independently, retain dtype-local zero offsets, and
# preserve BFloat16 tensors as their raw 16-bit representation.
mixed_params = [
    torch.tensor([1.5, -2.25], dtype=torch.float32),
    torch.tensor([1.0, -2.0, 3.5], dtype=torch.bfloat16),
]
mixed_driver = ParallelTemplatePartitionedGraphDriver(
    graph, partition_plan, replicated_parallel_plan, rank=0
)
mixed_driver._rank_parameter_layout = {
    0: {
        "dtype": TensorDType.Float32,
        "shape": (2,),
        "offset": 0,
        "numel": 2,
    },
    1: {
        "dtype": TensorDType.BFloat16,
        "shape": (3,),
        "offset": 0,
        "numel": 3,
    },
}
with tempfile.TemporaryDirectory() as temporary_directory:
    output_path = Path(temporary_directory)
    mixed_driver.build_rank_parameter_pack(mixed_params, output_path)
    assert {path.name for path in output_path.glob("*.data")} == {
        "rank0_params_float32.data",
        "rank0_params_bfloat16.data",
    }
    float32_pack = np.fromfile(
        output_path / "rank0_params_float32.data", dtype=np.float32
    )
    bfloat16_pack = np.fromfile(
        output_path / "rank0_params_bfloat16.data", dtype=np.uint16
    )
    np.testing.assert_array_equal(float32_pack, mixed_params[0].numpy())
    np.testing.assert_array_equal(
        bfloat16_pack, mixed_params[1].view(torch.uint16).numpy()
    )
    assert mixed_driver._rank_parameter_layout[0]["offset"] == 0
    assert mixed_driver._rank_parameter_layout[1]["offset"] == 0
    assert float32_pack.size == mixed_driver._rank_parameter_layout[0]["numel"]
    assert bfloat16_pack.size == mixed_driver._rank_parameter_layout[1]["numel"]
