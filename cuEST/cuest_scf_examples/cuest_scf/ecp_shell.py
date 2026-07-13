# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
    
class ECPShell(object):

    def __init__(
        self,
        *,
        L: int,
        ns: list,
        ws: list,
        es: list,
        is_active: bool,
    ):
        self.L = L
        self.ns = ns
        self.ws = ws
        self.es = es
        self.is_active = is_active

    @staticmethod
    def create_inactive_shell():
        return ECPShell(
            L=0,
            ns=[],
            ws=[],
            es=[],
            is_active=False,
        )

