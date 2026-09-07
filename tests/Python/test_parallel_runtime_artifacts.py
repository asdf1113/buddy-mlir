# RUN: %PYTHON %s rax-pack rax-inspect

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

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
from buddy.compiler.graph.operation import (
    AddMMOp,
    AddOp,
    OutputOp,
    PlaceholderOp,
)
from buddy.compiler.graph.source_meta import SourceMeta
from buddy.compiler.ops import func, tosa

SOURCE_ROOT = Path(
    os.environ.get("BUDDY_SRC_ROOT", Path(__file__).resolve().parents[2])
)
IMPORT_MODEL = SOURCE_ROOT / "tools" / "buddy-codegen" / "import_model.py"
GEN_MANIFEST = SOURCE_ROOT / "tools" / "buddy-codegen" / "gen_manifest.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


import_model = load_module("stage4c_import_model", IMPORT_MODEL)


def node(cls, name, shape, path=None):
    op = cls()
    op.name = name
    op.tensor_meta = {"shape": shape, "dtype": TensorDType.Float32}
    if path is not None:
        op._source_meta = (SourceMeta(module_path=path),)
    return op


def add(graph, op, kind=NodeType.OtherNode):
    graph.add_node(op, kind)
    return op


def bind(parent, child):
    parent.add_children(child.name)
    child.add_parent(parent.name)
    child.add_argument(parent.name)


def transformer_graph(function_name):
    graph = Graph({**tosa.ops_registry, **func.ops_registry}, function_name)
    tokens = add(
        graph,
        node(PlaceholderOp, "tokens", (2, 4)),
        NodeType.InputNode,
    )
    weights = []
    previous = tokens
    last_values = None
    for layer in range(2):
        q_bias = add(
            graph,
            node(PlaceholderOp, f"q_bias{layer}", (4,)),
            NodeType.FakeNode,
        )
        q_weight = add(
            graph,
            node(PlaceholderOp, f"q_weight{layer}", (4, 4)),
            NodeType.FakeNode,
        )
        o_bias = add(
            graph,
            node(PlaceholderOp, f"o_bias{layer}", (4,)),
            NodeType.FakeNode,
        )
        o_weight = add(
            graph,
            node(PlaceholderOp, f"o_weight{layer}", (4, 4)),
            NodeType.FakeNode,
        )
        weights.extend(
            [
                torch.arange(layer * 40, layer * 40 + 4, dtype=torch.float32),
                torch.arange(
                    layer * 40 + 4, layer * 40 + 20, dtype=torch.float32
                ).reshape(4, 4),
                torch.arange(
                    layer * 40 + 20, layer * 40 + 24, dtype=torch.float32
                ),
                torch.arange(
                    layer * 40 + 24, layer * 40 + 40, dtype=torch.float32
                ).reshape(4, 4),
            ]
        )
        q_proj = add(
            graph,
            node(
                AddMMOp,
                f"q_proj{layer}",
                (2, 4),
                f"model.layers.{layer}.self_attn.q_proj",
            ),
        )
        bind(q_bias, q_proj)
        bind(previous, q_proj)
        bind(q_weight, q_proj)
        o_proj = add(
            graph,
            node(
                AddMMOp,
                f"o_proj{layer}",
                (2, 4),
                f"model.layers.{layer}.self_attn.o_proj",
            ),
        )
        bind(o_bias, o_proj)
        bind(q_proj, o_proj)
        bind(o_weight, o_proj)
        residual = add(
            graph,
            node(
                AddOp,
                f"residual{layer}",
                (2, 4),
                f"model.layers.{layer}.post_attention_layernorm",
            ),
        )
        bind(previous, residual)
        bind(o_proj, residual)
        previous = residual
        last_values = (q_proj, o_proj, residual)

    output = add(graph, node(OutputOp, "output", ()))
    # This deliberately differs from name/definition order. Prefill's existing
    # output remap swaps the first two values and must survive JSON/RAX emission.
    for value in (last_values[1], last_values[0], last_values[2]):
        bind(value, output)
    return graph, weights


prefill_graph, params = transformer_graph("forward_prefill")
decode_graph, decode_params = transformer_graph("forward_decode")
assert all(
    torch.equal(a, b) for a, b in zip(params, decode_params, strict=True)
)

rax_pack = sys.argv[1] if len(sys.argv) > 1 else shutil.which("rax-pack")
rax_inspect = sys.argv[2] if len(sys.argv) > 2 else shutil.which("rax-inspect")
assert rax_pack is not None, "rax-pack must be available on PATH"
assert rax_inspect is not None, "rax-inspect must be available on PATH"

config = {
    "model_family": "stage4c",
    "model_id": "stage4c_frontend_test",
    "tokens": {"vocab_file": "vocab.txt"},
    "compilation": {"so_name": "stage4c_model.so"},
}

artifact_override = os.environ.get("STAGE4C_TEST_OUTPUT")
temporary = None
if artifact_override:
    output_root = Path(artifact_override)
    output_root.mkdir(parents=True, exist_ok=True)
else:
    temporary = tempfile.TemporaryDirectory()
    output_root = Path(temporary.name)
(output_root / "config.json").write_text(json.dumps(config, indent=2) + "\n")

manifest = import_model.export_template_partitioned_mlir(
    prefill_graph,
    decode_graph,
    str(output_root),
    params=params,
    tensor_parallel_size=2,
)
partition_dir = output_root / "layer_partitioned"
runtime_dir = partition_dir / "runtime"

assert manifest["runtime"]["world_size"] == 2
assert [item["rank"] for item in manifest["runtime"]["ranks"]] == [0, 1]
for function_name in ("forward_prefill", "forward_decode"):
    provenance = manifest["runtime"]["plans"][function_name]
    assert provenance["eligible_node_count"] == provenance["covered_node_count"]
    assert provenance["region_instance_count"] == 2
    assert provenance["layer_region_count"] == 2
    assert provenance["non_layer_region_count"] == 0
    assert [item["template_id"] for item in provenance["templates"]] == [0, 1]
    assert [item["segment_indices"] for item in provenance["instances"]] == [
        [0, 1],
        [0, 1],
    ]
assert (
    json.loads((partition_dir / "partition_manifest.json").read_text())
    == manifest
)

runtime_plans = {}
first_serialization = {}
for rank in range(2):
    rank_plans = {}
    for function_name in ("forward_prefill", "forward_decode"):
        path = runtime_dir / f"rank{rank}_{function_name}.json"
        assert path.is_file()
        first_serialization[path.name] = path.read_text()
        rank_plans[function_name] = json.loads(path.read_text())
        assert rank_plans[function_name]["rank"] == rank
        assert rank_plans[function_name]["world_size"] == 2
    runtime_plans[rank] = rank_plans
    assert (output_root / f"rank{rank}_params_float32.data").is_file()
    assert manifest["runtime"]["ranks"][rank]["parameter_packs"] == [
        f"../rank{rank}_params_float32.data"
    ]
    for function_name, runtime_plan in rank_plans.items():
        schedule = manifest["runtime"]["plans"][function_name][
            "rank_schedules"
        ][str(rank)]
        assert schedule["runtime_outputs"] == runtime_plan["runtime_outputs"]
        assert schedule["operations"] == [
            {
                "kind": operation["kind"],
                **(
                    {"wrapper": operation["wrapper"]}
                    if operation["kind"] == "dispatch"
                    else {"collective": operation["collective"]}
                ),
            }
            for operation in runtime_plan["operations"]
        ]

# Re-emission from the same frontend objects is byte-for-byte deterministic.
import_model.export_template_partitioned_mlir(
    prefill_graph,
    decode_graph,
    str(output_root),
    params=params,
    tensor_parallel_size=2,
)
for filename, contents in first_serialization.items():
    assert (runtime_dir / filename).read_text() == contents

# The internal exporter also accepts a phase subset and does not synthesize an
# empty schedule for an absent function.
decode_only_root = output_root / "decode_only"
decode_only_partition = decode_only_root / "layer_partitioned"
decode_only_partition.mkdir(parents=True)
decode_only_plan = build_transformer_partition_plan(decode_graph)
decode_only_index = import_model._export_parallel_template_runtime_artifacts(
    [(decode_graph, decode_only_plan, None)],
    params,
    str(decode_only_root),
    str(decode_only_partition),
    2,
)
assert all("forward_prefill" not in item for item in decode_only_index["ranks"])
assert all("forward_decode" in item for item in decode_only_index["ranks"])
assert not list((decode_only_partition / "runtime").glob("*prefill*.json"))

# The exporter calls the real Stage 4A driver and preserves the established
# model-specific Prefill output remap instead of sorting runtime outputs.
prefill_partition = build_transformer_partition_plan(prefill_graph)
prefill_parallel = build_transformer_parallel_plan(
    prefill_graph,
    prefill_partition,
    TransformerParallelConfig(tp_size=2),
)
for rank in range(2):
    direct_driver = ParallelTemplatePartitionedGraphDriver(
        prefill_graph, prefill_partition, prefill_parallel, rank
    )
    expected = direct_driver.build_parallel_runtime_plan(
        output_remap=import_model._prefill_output_remap(prefill_graph)
    )
    serialized_expected = json.loads(json.dumps(expected))
    assert runtime_plans[rank]["forward_prefill"] == serialized_expected
    unremapped = direct_driver.build_parallel_runtime_plan()
    assert serialized_expected != json.loads(json.dumps(unremapped))

for rank in range(2):
    prefill_path = runtime_dir / f"rank{rank}_forward_prefill.json"
    decode_path = runtime_dir / f"rank{rank}_forward_decode.json"
    rhal_path = output_root / f"rank{rank}.rhal.mlir"
    rax_path = output_root / f"rank{rank}.rax"
    subprocess.run(
        [
            sys.executable,
            str(GEN_MANIFEST),
            "--config",
            str(output_root / "config.json"),
            "--runtime-plan",
            f"forward_prefill={prefill_path}",
            "--runtime-plan",
            f"forward_decode={decode_path}",
            "--kernel-library",
            "stage4c_model.so",
            "--runner-library",
            "stage4c_runner.so",
            "-o",
            str(rhal_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [rax_pack, str(rhal_path), "-o", str(rax_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    inspected = subprocess.run(
        [rax_inspect, str(rax_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert inspected.index("@forward_prefill") < inspected.index(
        "@forward_decode"
    )

    for function_name in ("forward_prefill", "forward_decode"):
        start = inspected.index(f"@{function_name}")
        other = "forward_decode" if function_name == "forward_prefill" else None
        end = inspected.index(f"@{other}", start) if other else len(inspected)
        body = inspected[start:end]
        actual_order = re.findall(r"\[\d+\] (Dispatch|Collective)", body)
        expected_order = [
            "Dispatch" if operation["kind"] == "dispatch" else "Collective"
            for operation in runtime_plans[rank][function_name]["operations"]
        ]
        assert actual_order == expected_order
    assert "Collective kind=AllReduce reduction=Sum" in inspected
    print(f"rank{rank} schedule:\n{inspected}")

if temporary is not None:
    temporary.cleanup()
