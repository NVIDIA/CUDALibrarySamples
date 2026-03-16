# Copyright 2025-2026 NVIDIA CORPORATION & AFFILIATES.
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

import cuest.bindings as ce

class CuestResults(object):

    def __init__(
        self,
        *,
        resultsType,
        results_handle,
        ):

        self.initialized = False

        self.resultsType = resultsType
        self.results = results_handle

        status = ce.cuestResultsCreate(
            resultsType=self.resultsType,
            outResults=self.results,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestResultsCreate failed: %d' % status)

        self.initialized = True

    def __del__(self):

        if not self.initialized: return

        status = ce.cuestResultsDestroy(
            resultsType=self.resultsType,
            results=self.results,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestResultsDestroy failed: %d' % status)
