#! /usr/bin/python
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

import torch

import unittest
from parameterized import parameterized
from parameterized import param

import cutensor.torch as cutensor


class EinsumTest(unittest.TestCase):

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

        torch_rslt = torch_rslt

        self.assertEqual(cutensor_rslt.shape, torch_rslt.shape)
        self.assertEqual(cutensor_A_grad.shape, torch_A_grad.shape)
        self.assertEqual(cutensor_B_grad.shape, torch_B_grad.shape)
        torch.testing.assert_allclose(cutensor_rslt, torch_rslt, rtol=5e-3, atol=6e-3)
        torch.testing.assert_allclose(cutensor_A_grad, torch_A_grad, rtol=5e-3, atol=6e-3)
        torch.testing.assert_allclose(cutensor_B_grad, torch_B_grad, rtol=5e-3, atol=6e-3)

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

        self.assertEqual(cutensor_rslt.shape, torch_rslt.shape)
        for ct, tt in zip(cutensor_grads, torch_grads):
            self.assertEqual(ct.shape, tt.shape)
        torch.testing.assert_allclose(cutensor_rslt, torch_rslt, rtol=5e-3, atol=5e-3)
        for ct, tt in zip(cutensor_grads, torch_grads):
            torch.testing.assert_allclose(ct, tt, rtol=5e-3, atol=5e-3)


if __name__ == '__main__':
    unittest.main()
