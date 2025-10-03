#! /usr/bin/python
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

import torch

import unittest
from parameterized import parameterized
from parameterized import param

import cutensor.torch as cutensor


class EinsumTest(unittest.TestCase):


    def setUp(self):
        torch.backends.cuda.matmul.allow_tf32 = False


    def assertClose(self, cutensor_tensor, torch_tensor):
        self.assertEqual(cutensor_tensor.shape, torch_tensor.shape)
        self.assertEqual(torch.is_complex(cutensor_tensor), torch.is_complex(torch_tensor))
        if torch.is_complex(cutensor_tensor):
            self.assertClose(torch.real(cutensor_tensor), torch.real(torch_tensor))
            self.assertClose(torch.imag(cutensor_tensor), torch.imag(torch_tensor))
        else:
            torch.testing.assert_close(cutensor_tensor, torch_tensor, rtol=5e-3, atol=6e-3)
    

    @parameterized.expand(
        # yapf: disable
        [
            param(
                "test 0",
                a_size=(48, 37),
                b_size=(37, 74),
                equation="ik,kj->ij",
                dtype=torch.float32,
            ),
            param(
                "test 0 (complex)",
                a_size=(50, 50),
                b_size=(50, 50),
                equation="ik,kj->ij",
                dtype=torch.complex64,
            ),
            param(
                "test 1",
                a_size=(50, 50, 50),
                b_size=(50, 50, 50),
                equation="lik,lkj->lij",
                dtype=torch.complex128,
            ),
            param(
                "test 2",
                a_size=(50, 50, 50, 20),
                b_size=(50, 50, 50, 20),
                equation="likm,lkjm->lij",
                dtype=torch.float32,
            ),
            param(
                "test 3",
                a_size=(20, 50, 50, 50),
                b_size=(50, 50, 50, 20),
                equation="mlik,lkjm->lij",
                dtype=torch.float32,
            ),
            param(
                "test 4",
                a_size=(50, 50),
                b_size=(50, 50),
                equation="ik,kj->ij",
                dtype=torch.float16,
            ),
            param("test 5",
                  a_size=(50, 50, 50),
                  b_size=(50, 50, 50),
                  equation="lik,lkj->lij",
                  dtype=torch.float16),
            param(
                "test 6",
                a_size=(50, 50, 50, 20),
                b_size=(50, 50, 50, 20),
                equation="likm,lkjm->lij",
                dtype=torch.float16,
            ),
            param(
                "test 7",
                a_size=(20, 50, 50, 50),
                b_size=(50, 50, 50, 20),
                equation="mlik,lkjm->lij",
                dtype=torch.float16,
            ),
            param(
                "test 8",
                a_size=(2, 5, 50, 2),
                b_size=(5, 2, 50, 2),
                equation="mlik,lkjm",
                dtype=torch.float64,
            ),
            # Activate when cuTENSOR supports it
            # param(
            #     "test 8",
            #     a_size=(20, 50, 50, 50),
            #     b_size=(50, 50, 50, 20),
            #     equation="mlik,lkjm->lij",
            #     dtype=torch.bfloat16,
            # ),
        ]
        # yapf: enable
    )
    def test_einsum_equivalent_results(self,
                                       _,
                                       a_size,
                                       b_size,
                                       equation,
                                       dtype=torch.float32):


        kwargs = {
            'dtype': dtype,
            'device': torch.device("cuda"),
            'requires_grad': True
        }

        torch.manual_seed(0)

        cutensor_A = torch.randn(*a_size, **kwargs)
        cutensor_B = torch.randn(*b_size, **kwargs)
        cutensor_rslt = cutensor.EinsumFunction.apply(equation, cutensor_A,
                                                      cutensor_B)
        cutensor_rslt.backward(torch.ones_like(cutensor_rslt))
        cutensor_rslt = cutensor_rslt
        cutensor_A_grad = cutensor_A.grad
        cutensor_B_grad = cutensor_B.grad

        torch_A = cutensor_A.clone().detach().requires_grad_(True)
        torch_B = cutensor_B.clone().detach().requires_grad_(True)
        torch_rslt = torch.einsum(equation, torch_A, torch_B)
        torch_rslt.backward(torch.ones_like(torch_rslt))
        torch_A_grad = torch_A.grad
        torch_B_grad = torch_B.grad

        self.assertClose(cutensor_rslt, torch_rslt)
        self.assertClose(cutensor_A_grad, torch_A_grad)
        self.assertClose(cutensor_B_grad, torch_B_grad)

    @parameterized.expand(
        # yapf: disable
        [
            param(
                "test 0",
                sizes=[(50, 60), (60, 40)],
                equation="ik,kj->ji",
                dtype=torch.float32,
            ),
            param(
                "test 1",
                sizes=[(50, 60), (60, 7), (7, 8)],
                equation="ik,kl,lj->ij",
                dtype=torch.float32,
            ),
            param(
                "test 2",
                sizes=[(50, 60), (60, 7), (7, 8)],
                equation="ik,kl,lj",
                dtype=torch.float32,
            ),
            param(
                "test 3",
                sizes=[(50, 60), (60, 7), (7, 8)],
                equation="ik,kl,lj->ij",
                dtype=torch.complex64,
            ),
            # single input currently not supported
            param(
                "test 4",
                sizes=[(50, 60)],
                equation="ij->ji",
                dtype=torch.float32,
            ),
        ]
        # yapf: enable
    )
    def test_einsum_general_equivalent_results(self,
                                               _,
                                               sizes,
                                               equation,
                                               dtype=torch.float32):

        kwargs = {
            'dtype': dtype,
            'device': torch.device("cuda"),
            'requires_grad': True
        }

        cutensor_tensors = [torch.randn(*size, **kwargs) for size in sizes]
        torch_tensors = [
            t.clone().detach().requires_grad_(True) for t in cutensor_tensors
        ]

        cutensor_rslt = cutensor.EinsumGeneral(equation, *cutensor_tensors)
        cutensor_rslt.backward(torch.ones_like(cutensor_rslt))
        cutensor_rslt = cutensor_rslt
        cutensor_grads = [
            t.grad for t in cutensor_tensors
        ]

        torch_rslt = torch.einsum(equation, *torch_tensors)
        torch_rslt.backward(torch.ones_like(torch_rslt))
        torch_rslt = torch_rslt
        torch_grads = [t.grad for t in torch_tensors]


        self.assertClose(cutensor_rslt, torch_rslt)
        for ct, tt in zip(cutensor_grads, torch_grads):
            self.assertClose(ct, tt)


if __name__ == '__main__':
    unittest.main()
