#!/usr/bin/env python3

import os
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, StaticCache
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import (
    simply_fuse,
    apply_classic_fusion,
    eliminate_transpose,
    eliminate_matmul_transpose_reshape,
    flash_attention_prefill,
    gqa_attention_fusion,
)
from buddy.compiler.graph.type import DeviceType


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSeekR1 AOT Importer (f32 only)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./",
        help="Directory to save output files.",
    )
    return parser.parse_args()


def load_model():
    model_path = os.environ.get(
        "DEEPSEEKR1_MODEL_PATH",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torchscript=True,
    ).eval()
    model.config.use_cache = False
    return model


def build_compiler(func_name: str):
    return DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
        func_name=func_name,
    )


def import_graphs(model, prefill_compiler, decode_compiler):
    input_ids_prefill = torch.zeros((1, 1024), dtype=torch.int64)
    input_ids_decode = torch.zeros((1, 1), dtype=torch.int64)
    cache_position = torch.tensor([200], dtype=torch.int64)

    past_key_values_decode = StaticCache(
        config=model.config,
        max_cache_len=1024,
    )

    graphs_prefill = prefill_compiler.importer(
        model,
        input_ids=input_ids_prefill,
        use_cache=True,
        cache_implementation="static",
    )

    # 先跑一次，初始化 decode 用到的 KV cache
    model(
        input_ids=input_ids_decode,
        past_key_values=past_key_values_decode,
        use_cache=True,
        cache_implementation="static",
    )

    graphs_decode = decode_compiler.importer(
        model,
        input_ids=input_ids_decode,
        use_cache=True,
        cache_position=cache_position,
        past_key_values=past_key_values_decode,
        cache_implementation="static",
    )

    assert len(graphs_prefill) == 1
    assert len(graphs_decode) == 1
    return graphs_prefill[0], graphs_decode[0]


def optimize_graph(graph, mode: str):
    if mode == "prefill":
        # 只对 prefill 做 transpose / matmul-transpose-reshape 消除
        graph.perform([eliminate_transpose, eliminate_matmul_transpose_reshape])

        pattern_list = [
            simply_fuse,
            apply_classic_fusion,
            flash_attention_prefill,
        ]
        new_name = "subgraph0_prefill"
    else:
        # decode 不做这两个 pass
        pattern_list = [
            simply_fuse,
            apply_classic_fusion,
            gqa_attention_fusion,
        ]
        new_name = "subgraph0_decode"

    graph.fuse_ops(pattern_list)
    graph.op_groups[new_name] = graph.op_groups.pop("subgraph0")
    graph.group_map_device[new_name] = DeviceType.CPU


def lower_graph(graph):
    driver = GraphDriver(graph)
    driver.subgraphs[0].lower_to_top_level_ir()
    return driver


def save_outputs(output_dir, driver_prefill, driver_decode, params):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "subgraph0_prefill.mlir"), "w") as f:
        print(driver_prefill.subgraphs[0]._imported_module, file=f)

    with open(os.path.join(output_dir, "forward_prefill.mlir"), "w") as f:
        print(driver_prefill.construct_main_graph(True), file=f)

    all_param = np.concatenate(
        [param.detach().numpy().reshape([-1]) for param in params]
    )
    all_param.tofile(os.path.join(output_dir, "arg0.data"))

    with open(os.path.join(output_dir, "subgraph0_decode.mlir"), "w") as f:
        print(driver_decode.subgraphs[0]._imported_module, file=f)

    with open(os.path.join(output_dir, "forward_decode.mlir"), "w") as f:
        print(driver_decode.construct_main_graph(True), file=f)


def main():
    args = parse_args()
    model = load_model()

    prefill_compiler = build_compiler("forward_prefill")
    decode_compiler = build_compiler("forward_decode")

    with torch.no_grad():
        graph_prefill, graph_decode = import_graphs(
            model,
            prefill_compiler,
            decode_compiler,
        )

    params = prefill_compiler.imported_params[graph_prefill]

    optimize_graph(graph_prefill, "prefill")
    optimize_graph(graph_decode, "decode")

    driver_prefill = lower_graph(graph_prefill)
    driver_decode = lower_graph(graph_decode)

    save_outputs(args.output_dir, driver_prefill, driver_decode, params)


if __name__ == "__main__":
    main()