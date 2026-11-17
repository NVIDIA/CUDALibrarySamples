# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED BIN2C OR NOT DEFINED INPUT OR NOT DEFINED OUTPUT)
    message(FATAL_ERROR "BIN2C, INPUT, and OUTPUT must be specified")
endif()

execute_process(
    COMMAND "${BIN2C}" --name cufft_lto_store_multiply --type longlong "${INPUT}"
    OUTPUT_FILE "${OUTPUT}"
    COMMAND_ERROR_IS_FATAL ANY)
