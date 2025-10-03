# ! /usr/bin/python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

def normalize_subscript(subscript):
    if '->' in subscript:
        subscript = subscript.split('->')
        lhs = subscript[0]
        rhs = subscript[1]
    else:
        lhs = subscript
        rhs = ''.join(sorted([s for s in set(subscript) if s != ',' and subscript.count(s) == 1]))
    if '...' in lhs:
        raise RuntimeError('Elipsis is currently unsupported')
    return lhs + '->' + rhs, ',' in lhs
