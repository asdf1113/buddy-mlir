//===- MpiCommunicator.cpp - MPI collective backend ----------------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/communication/MpiCommunicator.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace buddy {
namespace runtime {

static void checkMpi(int status, const char *operation) {
  if (status != MPI_SUCCESS)
    throw std::runtime_error(std::string("MpiCommunicator: ") + operation +
                             " failed");
}

MpiCommunicator::MpiCommunicator(MPI_Comm communicator)
    : communicator_(communicator) {}

int MpiCommunicator::rank() const {
  int value = 0;
  checkMpi(MPI_Comm_rank(communicator_, &value), "MPI_Comm_rank");
  return value;
}

int MpiCommunicator::size() const {
  int value = 0;
  checkMpi(MPI_Comm_size(communicator_, &value), "MPI_Comm_size");
  return value;
}

void MpiCommunicator::broadcast(void *buffer, size_t bytes, int root) {
  if (bytes > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error(
        "MpiCommunicator: Broadcast count exceeds MPI int");
  checkMpi(
      MPI_Bcast(buffer, static_cast<int>(bytes), MPI_BYTE, root, communicator_),
      "MPI_Bcast");
}

void MpiCommunicator::allReduce(const void *sendBuffer, void *recvBuffer,
                                size_t count, DataType dataType,
                                ReductionOp reduction) {
  if (dataType != DataType::F32 || reduction != ReductionOp::Sum)
    throw std::runtime_error(
        "MpiCommunicator: unsupported AllReduce datatype or reduction");
  if (count > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error(
        "MpiCommunicator: AllReduce count exceeds MPI int");
  const void *mpiSendBuffer =
      sendBuffer == recvBuffer ? MPI_IN_PLACE : sendBuffer;
  checkMpi(MPI_Allreduce(mpiSendBuffer, recvBuffer, static_cast<int>(count),
                         MPI_FLOAT, MPI_SUM, communicator_),
           "MPI_Allreduce");
}

} // namespace runtime
} // namespace buddy
