# ! /usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#  - Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  - Neither the name(s) of the copyright holder(s) nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

from parameterized import parameterized
from parameterized import param

import tensorflow as tf
from tensorflow.python.platform import test
import tensorflow.test

import cutensor.tensorflow as cutensor

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class EinsumcuTENSORTest(tensorflow.test.TestCase):

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
        A = tf.compat.v1.get_variable("A",
                                      shape=a_size,
                                      initializer=tf.random_normal_initializer,
                                      dtype=dtype)

        if b_size is not None:
            B = tf.compat.v1.get_variable("B",
                                          shape=b_size,
                                          initializer=tf.random_normal_initializer,
                                          dtype=dtype)

            tf_native_rslt = tf.einsum(equation, A, B, name="tf_native_einsum")
            tf_native_grads = tf.gradients(tf_native_rslt, [A, B])

            tf_cutensor_rslt = cutensor.einsum(equation,
                                               A,
                                               B,
                                               name="tf_cuTensor_einsum")
            tf_cutensor_grads = tf.gradients(tf_cutensor_rslt, [A, B])
        else:
            tf_native_rslt = tf.einsum(equation, A, name="tf_native_einsum")
            tf_native_grads = tf.gradients(tf_native_rslt, [A])

            tf_cutensor_rslt = cutensor.einsum(equation,
                                               A,
                                               name="tf_cuTensor_einsum")
            tf_cutensor_grads = tf.gradients(tf_cutensor_rslt, [A])

        self.assertEqual(tf_native_rslt.get_shape(),
                         tf_cutensor_rslt.get_shape())

        self.assertEqual(tf_native_rslt.dtype, tf_cutensor_rslt.dtype)
        self.assertEqual(len(tf_cutensor_grads), len(tf_native_grads))

        with self.session(use_gpu=True) as sess:

            sess.run(tf.compat.v1.global_variables_initializer())

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
