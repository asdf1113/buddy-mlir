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
    build_transformer_partition_plan,
)
from buddy.compiler.graph.graph import GraphImporter
from buddy.compiler.graph.operation import (
    AddOp,
    CallOp,
    OutputOp,
    PlaceholderOp,
)
from buddy.compiler.graph.source_meta import SourceMeta
from buddy.compiler.graph.transformer_partition import (
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
from buddy.compiler.ops import func, tosa
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
