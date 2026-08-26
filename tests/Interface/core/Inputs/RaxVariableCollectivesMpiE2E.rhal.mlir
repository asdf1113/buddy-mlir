rhal.module @rax_variable_collectives_mpi_e2e attributes {version = "0.1.0"} {
  rhal.buffer @all_gatherv_input {space = "host", type = tensor<2xf32>}
  rhal.buffer @all_gatherv_output {space = "host", type = tensor<4xf32>}
  rhal.buffer @reduce_scatter_input {space = "host", type = tensor<4xf32>}
  rhal.buffer @reduce_scatter_output {space = "host", type = tensor<2xf32>}

  rhal.func @all_gatherv {inputs = ["all_gatherv_input"],
                          outputs = ["all_gatherv_output"]} body {
    rhal.collective [@all_gatherv_input] {
      kind = "all_gatherv",
      output_buffers = [@all_gatherv_output],
      recv_counts = array<i64: 2, 2>,
      displacements = array<i64: 0, 2>
    }
  }

  rhal.func @reduce_scatter {inputs = ["reduce_scatter_input"],
                             outputs = ["reduce_scatter_output"]} body {
    rhal.collective [@reduce_scatter_input] {
      kind = "reduce_scatter",
      output_buffers = [@reduce_scatter_output],
      recv_counts = array<i64: 2, 2>,
      reduction = "sum"
    }
  }
}
