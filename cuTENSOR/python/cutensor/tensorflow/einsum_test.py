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

from parameterized import parameterized
from parameterized import param

import tensorflow as tf
from tensorflow.python.platform import test
import tensorflow.test

import cutensor.tensorflow as cutensor

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class EinsumcuTENSORTest(tensorflow.test.TestCase):

    def setUp(self):
        super().setUp()
        tf.config.experimental.enable_tensor_float_32_execution(False)

    @parameterized.expand(
        # yapf: disable
        [
            param(
                "test 0",
                a_size=(50, 50),
                b_size=(50, 50),
                equation="ik,kj->ij",
                dtype=tf.float32,
            ),
            param(
                "test 1",
                a_size=(50, 50, 50),
                b_size=(50, 50, 50),
                equation="lik,lkj->lij",
                dtype=tf.float32,
            ),
            param(
                "test 2",
                a_size=(50, 50, 50, 20),
                b_size=(50, 50, 50, 20),
                equation="likm,lkjm->lij",
                dtype=tf.float32,
            ),
            param(
                "test 3",
                a_size=(20, 50, 50, 50),
                b_size=(50, 50, 50, 20),
                equation="mlik,lkjm->lij",
                dtype=tf.float32,
            ),
            param(
                "test 4",
                a_size=(50, 50),
                b_size=(50, 50),
                equation="ik,kj->ij",
                dtype=tf.float16,
            ),
            param("test 5",
                  a_size=(50, 50, 50),
                  b_size=(50, 50, 50),
                  equation="lik,lkj->lij",
                  dtype=tf.float16),
            param(
                "test 6",
                a_size=(50, 50, 50, 20),
                b_size=(50, 50, 50, 20),
                equation="likm,lkjm->lij",
                dtype=tf.float16,
            ),
            param(
                "test 7",
                a_size=(20, 50, 50, 50),
                b_size=(50, 50, 50, 20),
                equation="mlik,lkjm->lij",
                dtype=tf.float16,
            ),
            param(
                "test 8",
                a_size=(2, 5, 5, 5),
                b_size=(5, 5, 5, 2),
                equation="mlik,lkjm",
                dtype=tf.float16,
            ),
            param(
                "test 9",
                a_size=(20, 50, 50, 50),
                b_size=None,
                equation="mlik->imlk",
                dtype=tf.float16,
            ),
            # Activate when cuTENSOR supports it
            # param(
            #     "test 8",
            #     a_size=(20, 50, 50, 50),
            #     b_size=(50, 50, 50, 20),
            #     equation="mlik,lkjm->lij",
            #     dtype=tf.bfloat16,
            # ),
        ]
        # yapf: enable
    )
    def test_einsum_equivalent_results(self,
                                       _,
                                       a_size,
                                       b_size,
                                       equation,
                                       dtype=tf.float32):
        A = tf.random.normal(a_size, dtype=dtype)


        if b_size is not None:
            B = tf.random.normal(b_size, dtype=dtype)

            with tf.GradientTape() as tape:
                tape.watch([A, B])
                tf_native_rslt = tf.einsum(equation, A, B, name="tf_native_einsum")

            tf_native_grads = tape.gradient(tf_native_rslt, [A, B])

            with tf.GradientTape() as tape:
                tape.watch([A, B])
                tf_cutensor_rslt = cutensor.einsum(equation,
                                                   A,
                                                   B,
                                                   name="tf_cuTensor_einsum")
            
            tf_cutensor_grads = tape.gradient(tf_cutensor_rslt, [A, B])
        else:
            with tf.GradientTape() as tape:
                tape.watch([A])
                tf_native_rslt = tf.einsum(equation, A, name="tf_native_einsum")

            tf_native_grads = tape.gradient(tf_native_rslt, [A])

            with tf.GradientTape() as tape:
                tape.watch([A])
                tf_cutensor_rslt = cutensor.einsum(equation,
                                                   A,
                                                   name="tf_cuTensor_einsum")
            tf_cutensor_grads = tape.gradient(tf_cutensor_rslt, [A])

        self.assertEqual(tf_native_rslt.get_shape(),
                         tf_cutensor_rslt.get_shape())

        self.assertEqual(tf_native_rslt.dtype, tf_cutensor_rslt.dtype)
        self.assertEqual(len(tf_cutensor_grads), len(tf_native_grads))

        self.assertAllClose(tf_native_rslt,
                            tf_cutensor_rslt,
                            rtol=5e-03,
                            atol=5e-03)

        for tf_native_grad, tf_cutensor_grad in zip(tf_native_grads,
                                                    tf_cutensor_grads):
            self.assertAllClose(tf_native_grad,
                                tf_cutensor_grad,
                                rtol=5e-03,
                                atol=5e-03)
            self.assertEqual(tf_native_grad.dtype, tf_cutensor_grad.dtype)


if __name__ == '__main__':
    test.main()
