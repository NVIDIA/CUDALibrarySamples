# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cuest.bindings as ce

class XCFunctionalInfo(object):

    @staticmethod
    def string_to_enum(
        functional_name : str,
        ):

        if functional_name.lower() == 'hf':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_HF
        elif functional_name.lower() == 'b3lyp1':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B3LYP1
        elif functional_name.lower() == 'b3lyp5':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B3LYP5
        elif functional_name.lower() == 'b97':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B97
        elif functional_name.lower() == 'blyp':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_BLYP
        elif functional_name.lower() == 'm06-l':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_M06L
        elif functional_name.lower() == 'pbe':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_PBE
        elif functional_name.lower() == 'pbe0':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_PBE0
        elif functional_name.lower() == 'r2scan':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_R2SCAN
        elif functional_name.lower() == 'svwn5':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_SVWN5
        elif functional_name.lower() == 'b97m-v':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B97MV
        else:
            raise RuntimeError('Unknown DFT functional')

    @staticmethod
    def enum_to_string(
        functional_enum : ce.CuestXCIntPlanParametersFunctional,
        ):

        if functional_enum == ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_HF:
            return 'HF'
        elif functional_enum == ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B3LYP1:
            return 'B3LYP1'
        elif functional_enum == ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B3LYP5:
            return 'B3LYP5'
        elif functional_enum == ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B97:
            return 'B97'
        elif functional_enum == ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_BLYP:
            return 'BLYP'
        elif functional_enum == ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_M06L:
            return 'M06-L'
        elif functional_enum == ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_PBE:
            return 'PBE'
        elif functional_enum == ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_PBE0:
            return 'PBE0'
        elif functional_enum == ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_R2SCAN:
            return 'r2SCAN'
        elif functional_enum == ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_SVWN5:
            return 'SVWN5'
        elif functional_enum == ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B97MV:
            return 'B97M-V'
        else:
            raise RuntimeError('Unknown DFT functional')

