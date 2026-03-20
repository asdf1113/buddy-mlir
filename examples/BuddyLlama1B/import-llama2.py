# ===- import-llama2.py --------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# This is the test of llama2 model.
#
# ===---------------------------------------------------------------------------

import os
import argparse
import torch
import torch._dynamo as dynamo
from transformers import AutoModelForCausalLM
from torch._inductor.decomposition import decompositions as inductor_decomp
import numpy

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse, apply_classic_fusion
# from buddy.compiler.graph.transform import (
#     simply_fuse,
#     apply_classic_fusion,
#     eliminate_transpose,
#     eliminate_matmul_transpose_reshape,
#     flash_attention_prefill,
#     gqa_attention_fusion,
# )
# from buddy.compiler.graph.type import DeviceType
# from buddy.compiler.graph.operation import *

parser = argparse.ArgumentParser(description="LLaMA 3.2 model AOT importer")
parser.add_argument(
    "--output-dir",
    type=str,
    default="./",
    help="Directory to save output files.",
)
args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

model_path = os.environ.get("LLAMA_MODEL_PATH")
if model_path is None:
    raise EnvironmentError(
        "The environment variable 'LLAMA_MODEL_PATH' is not set or is invalid."
    )

# 直接加载模型；tokenizer在这个脚本里并没有实际用到
model = AutoModelForCausalLM.from_pretrained(model_path)
model.config.use_cache = False

dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

with torch.no_grad():
    data = torch.tensor([[1 for _ in range(40)]], dtype=torch.int64)
    graphs = dynamo_compiler.importer(model, data)

assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]

# graphs[0].perform(
#     [eliminate_transpose, eliminate_matmul_transpose_reshape]
# )

# pattern_list = [
#     simply_fuse,
#     apply_classic_fusion,
#     flash_attention_prefill,
# ]
pattern_list = [simply_fuse]

graphs[0].fuse_ops(pattern_list)

driver = GraphDriver(graphs[0])
driver.subgraphs[0].lower_to_top_level_ir()

with open(os.path.join(output_dir, "subgraph0.mlir"), "w") as module_file:
    print(driver.subgraphs[0]._imported_module, file=module_file)

with open(os.path.join(output_dir, "forward.mlir"), "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)

all_param = numpy.concatenate(
    [param.detach().numpy().reshape([-1]) for param in params]
)
all_param.tofile(os.path.join(output_dir, "arg0.data"))