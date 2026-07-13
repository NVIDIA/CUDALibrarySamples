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

import cuda.bindings.runtime as cuda

class Timer(object):

    def __init__(self):
        self._start_events = {}
        self._stop_events = {}

        self._times = {}
        self._counts = {}

        self._ordered_keys = []

    @property
    def ordered_keys(self):
        return self._ordered_keys

    @property
    def times(self):
        return self._times

    @property
    def counts(self):
        return self._counts

    def string(self):
        s = ''
        k = 'key'
        c = 'count'
        t = 'time [ms]'
        s += f'{k:30s} {c:5s} {t:11s}\n'
        for ok in self._ordered_keys:
            s += f'{str(ok):30s} {self._counts[ok]:5d} {self._times[ok]:11.3E}\n'
        return s

    def print_out(self, header : str | None = None):
        if header is not None:
            print(header)
        print(self.string())

    def start(
        self,
        *,
        key,
        stream_handle=0,
    ):

        if key in self._start_events:
            raise RuntimeError('Key already started: ', key)

        status, start = cuda.cudaEventCreate()
        if status != cuda.cudaError_t.cudaSuccess:
            raise RuntimeError('cudaEventCreate failed: ', status)

        status, stop = cuda.cudaEventCreate()
        if status != cuda.cudaError_t.cudaSuccess:
            raise RuntimeError('cudaEventCreate failed: ', status)

        self._start_events[key] = start
        self._stop_events[key] = stop

        if key not in self._ordered_keys:
            self._ordered_keys.append(key)

        ret = cuda.cudaEventRecord(self._start_events[key], stream_handle)
        if ret[0] != cuda.cudaError_t.cudaSuccess:
            raise RuntimeError("cudaEventRecord failed:", ret[0])


    def stop(
        self,
        *,
        key,
        stream_handle=0,
    ):
    
        if key not in self._start_events:
            raise RuntimeError('Unknown key: ', key)

        status = cuda.cudaEventRecord(self._stop_events[key], stream_handle)
        if status[0] != cuda.cudaError_t.cudaSuccess:
            raise RuntimeError('cudaEventRecord failed: ', status)

        status = cuda.cudaEventSynchronize(self._stop_events[key])
        if status[0] != cuda.cudaError_t.cudaSuccess:
            raise RuntimeError('cudaEventSynchronize failed: ', status)

        status, time_ms = cuda.cudaEventElapsedTime(self._start_events[key], self._stop_events[key])
        if status != cuda.cudaError_t.cudaSuccess:
            raise RuntimeError('cudaEventElapsedTime failed: ', status)

        status = cuda.cudaEventDestroy(self._start_events[key])
        if status[0] != cuda.cudaError_t.cudaSuccess:
            raise RuntimeError('cudaEventDestroy failed: ', status)
        status = cuda.cudaEventDestroy(self._stop_events[key])
        if status[0] != cuda.cudaError_t.cudaSuccess:
            raise RuntimeError('cudaEventDestroy failed: ', status)

        del self._start_events[key]
        del self._stop_events[key]

        if key in self._times:
            self._times[key] += 1.0E-3 * time_ms
        else:
            self._times[key] = 1.0E-3 * time_ms

        if key in self._counts:
            self._counts[key] += 1
        else:
            self._counts[key] = 1

        return 1.0e-3 * time_ms
