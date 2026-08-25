#!/usr/bin/env python3

import argparse
import json
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
from buddy.compiler.graph.operation import (
    AddOp,
    MatmulOp,
    OutputOp,
    PlaceholderOp,
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
from buddy.compiler.ops import func, linalg


def make_node(cls, name, shape, *, layer_path=None):
    op = cls()
    op.name = name
    op.tensor_meta = {"shape": shape, "dtype": TensorDType.Float32}
    if layer_path is not None:
        op._source_meta = (
            SourceMeta(
                module_path=layer_path,
                module_class="stage2e.TinyRowParallelLayer",
            ),
        )
    return op


def add_node(graph, op, kind=NodeType.OtherNode):
    graph.add_node(op, kind)
    return op


def bind(parent, child):
    parent.add_children(child.name)
    child.add_parent(parent.name)
    child.add_argument(parent.name)


def build_fixture():
    graph = Graph({**linalg.ops_registry, **func.ops_registry}, "stage2e_tp")
    input_op = add_node(
        graph,
        make_node(PlaceholderOp, "input", (1, 4)),
        NodeType.InputNode,
    )
    weight_op = add_node(
        graph,
        make_node(PlaceholderOp, "weight", (4, 2)),
        NodeType.FakeNode,
    )
    bias_op = add_node(
        graph,
        make_node(PlaceholderOp, "bias", (1, 2)),
        NodeType.FakeNode,
    )
    layer_path = "model.layers.0.mlp.down_proj"
    matmul_op = add_node(
        graph,
        make_node(MatmulOp, "local_matmul", (1, 2), layer_path=layer_path),
    )
    add_op = add_node(
        graph, make_node(AddOp, "bias_add", (1, 2), layer_path=layer_path)
    )
    output_op = add_node(graph, make_node(OutputOp, "output", ()))
    bind(input_op, matmul_op)
    bind(weight_op, matmul_op)
    bind(matmul_op, add_op)
    bind(bias_op, add_op)
    bind(add_op, output_op)

    partition_plan = build_transformer_partition_plan(graph)
    if len(partition_plan.templates) != 1:
        raise AssertionError("Stage 2E fixture must materialize one template")
    unit = partition_plan.templates[0]
    if len(unit.instances) != 1:
        raise AssertionError("Stage 2E fixture must materialize one instance")

    data_value, weight_value, bias_value = (
        item.value for item in unit.representative.interface.ordered_inputs
    )
    matmul_value = GraphValueRef(matmul_op)
    add_value = GraphValueRef(add_op)
    replicated = TensorLayout(LayoutKind.REPLICATED)
    parallel_plan = TransformerParallelPlan(
        graph_name=graph._func_name,
        world_size=2,
        templates=(
            TemplateParallelPlan(
                template_id=unit.template_id,
                parameter_shards=(
                    ParameterShardSpec(
                        parameter=weight_value,
                        global_parameter_index=0,
                        template_parameter_slot=0,
                        role=ParameterRole.DOWN_WEIGHT,
                        semantic_axis=FeatureAxis.INPUT_FEATURE,
                        consumer_use=OperandUseRef(matmul_op, ("args", 1)),
                        consumer_shard_axis=0,
                        storage_shard_axis=0,
                        storage_to_consumer_permutation=(0, 1),
                        global_shape=(4, 2),
                        rank_slices=(RankSlice(0, 0, 2), RankSlice(1, 2, 2)),
                    ),
                ),
                value_layouts=(
                    ValueLayoutSpec(
                        data_value,
                        (1, 4),
                        ((1, 2), (1, 2)),
                        TensorLayout(LayoutKind.SHARDED, 1),
                    ),
                    ValueLayoutSpec(
                        weight_value,
                        (4, 2),
                        ((2, 2), (2, 2)),
                        TensorLayout(LayoutKind.SHARDED, 0),
                    ),
                    ValueLayoutSpec(
                        bias_value,
                        (1, 2),
                        ((1, 2), (1, 2)),
                        replicated,
                    ),
                    ValueLayoutSpec(
                        matmul_value,
                        (1, 2),
                        ((1, 2), (1, 2)),
                        TensorLayout(LayoutKind.PARTIAL),
                    ),
                    ValueLayoutSpec(
                        add_value,
                        (1, 2),
                        ((1, 2), (1, 2)),
                        replicated,
                    ),
                ),
                op_rewrites=(),
                collectives=(
                    CollectiveBoundary(
                        producer=matmul_value,
                        consumers=(OperandUseRef(add_op, ("args", 0)),),
                        kind=CollectiveKind.ALL_REDUCE,
                        target_layout=replicated,
                        target_local_shapes=((1, 2), (1, 2)),
                    ),
                ),
                segments=(
                    ComputeSegment(
                        0,
                        (matmul_op,),
                        (data_value, weight_value),
                        (matmul_value,),
                    ),
                    ComputeSegment(
                        1,
                        (add_op,),
                        (matmul_value, bias_value),
                        (add_value,),
                    ),
                ),
            ),
        ),
    )
    parameters = [
        torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
            dtype=torch.float32,
        ),
        torch.tensor([[10.0, 20.0]], dtype=torch.float32),
    ]
    return graph, partition_plan, parallel_plan, parameters


def pack_sizes(driver):
    sizes = {}
    for layout in driver._rank_parameter_layout.values():
        dtype = layout["dtype"]
        sizes[dtype] = max(
            sizes.get(dtype, 0), layout["offset"] + layout["numel"]
        )
    return sizes


def lower_wrapper(driver, key):
    wrapper = driver._parallel_segment_wrappers[key]
    offsets = {
        name: driver._rank_parameter_layout[parameter_index]["offset"]
        for name, parameter_index in driver._wrapper_parameter_bindings[
            key
        ].items()
    }
    wrapper.lower_to_top_level_ir(
        do_param_pack=True,
        param_pack_sizes=pack_sizes(driver),
        param_pack_offsets=offsets,
    )
    return wrapper


def emit_rhal(path, rank, library, wrapper_symbols, pack_path):
    text = f"""rhal.module @stage2e_tp_rank{rank} attributes {{version = "0.1.0"}} {{
  rhal.constant @params {{id = 7 : i32, storage = "external",
                         type = tensor<6xf32>, uri = "file:{pack_path}"}}

  rhal.codeobj @segment0 {{id = 10 : i32, kind = "host_shared_lib",
                           backend = "cpu", uri = "file:{library}",
                           entry_symbol = "rax_{wrapper_symbols[0]}"}}
  rhal.codeobj @segment1 {{id = 11 : i32, kind = "host_shared_lib",
                           backend = "cpu", uri = "file:{library}",
                           entry_symbol = "rax_{wrapper_symbols[1]}"}}

  rhal.buffer @input {{space = "host", type = tensor<1x2xf32>}}
  rhal.buffer @boundary {{space = "host", type = tensor<1x2xf32>}}
  rhal.buffer @output {{space = "host", type = tensor<1x2xf32>}}

  rhal.func @forward {{inputs = ["input"], outputs = ["output"]}} body {{
    rhal.dispatch @segment0 [@params, @input, @boundary]
    rhal.collective [@boundary] {{
      kind = "all_reduce",
      reduction = "sum"
    }}
    rhal.dispatch @segment1 [@params, @boundary, @output]
  }}
}}
"""
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rank0-library", required=True)
    parser.add_argument("--rank1-library", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    graph, partition_plan, parallel_plan, parameters = build_fixture()
    summary = {"ranks": {}}
    for rank, library in enumerate((args.rank0_library, args.rank1_library)):
        rank_dir = args.output_dir / f"rank{rank}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        driver = ParallelTemplatePartitionedGraphDriver(
            graph, partition_plan, parallel_plan, rank=rank
        )
        subgraphs = driver.build_parallel_template_subgraphs()
        wrappers = driver.construct_parallel_segment_wrappers()
        if len(subgraphs) != 2 or len(wrappers) != 2:
            raise AssertionError(
                "Stage 2E fixture must generate two segments/wrappers"
            )
        driver.build_rank_parameter_pack(parameters, rank_dir)

        for segment_index, subgraph in enumerate(subgraphs):
            subgraph.lower_to_top_level_ir()
            (rank_dir / f"segment{segment_index}.mlir").write_text(
                str(subgraph._imported_module)
            )
        selected_wrappers = []
        for segment_index in range(2):
            wrapper = lower_wrapper(driver, (0, segment_index))
            selected_wrappers.append(wrapper)
            (rank_dir / f"wrapper{segment_index}.mlir").write_text(
                str(wrapper._imported_module)
            )

        pack_path = rank_dir / f"rank{rank}_params_float32.data"
        packed = np.fromfile(pack_path, dtype=np.float32)
        expected_weight = (
            parameters[0][rank * 2 : rank * 2 + 2].numpy().reshape(-1)
        )
        expected = np.concatenate(
            (expected_weight, parameters[1].numpy().reshape(-1))
        )
        np.testing.assert_array_equal(packed, expected)
        if driver._rank_parameter_layout != {
            0: {
                "dtype": TensorDType.Float32,
                "shape": (2, 2),
                "offset": 0,
                "numel": 4,
            },
            1: {
                "dtype": TensorDType.Float32,
                "shape": (1, 2),
                "offset": 4,
                "numel": 2,
            },
        }:
            raise AssertionError("unexpected frontend rank parameter layout")
        wrapper_symbols = [wrapper._func_name for wrapper in selected_wrappers]
        emit_rhal(
            args.output_dir / f"rank{rank}.rhal.mlir",
            rank,
            library,
            wrapper_symbols,
            pack_path.resolve(),
        )
        summary["ranks"][str(rank)] = {
            "segments": [subgraph._func_name for subgraph in subgraphs],
            "wrappers": wrapper_symbols,
            "parameter_layout": {
                str(index): {
                    **layout,
                    "dtype": layout["dtype"].value,
                    "shape": list(layout["shape"]),
                }
                for index, layout in driver._rank_parameter_layout.items()
            },
            "wrapper_parameter_bindings": {
                f"instance{key[0]}_segment{key[1]}": value
                for key, value in driver._wrapper_parameter_bindings.items()
            },
            "pack": packed.tolist(),
            "collective": "ALL_REDUCE",
        }
    (args.output_dir / "materialization.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
