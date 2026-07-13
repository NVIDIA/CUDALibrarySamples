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

import cuest.bindings as ce

class XCFunctionalInfo(object):

    @staticmethod
    def string_to_enum(
        functional_name : str,
        ):

        sanitized_name = functional_name.replace('-', '').lower()
        if sanitized_name == 'hf':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_HF
        elif sanitized_name == 'b3lyp1':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B3LYP1
        elif sanitized_name == 'b3lyp5':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B3LYP5
        elif sanitized_name == 'b97':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B97
        elif sanitized_name == 'blyp':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_BLYP
        elif sanitized_name == 'm06l':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_M06L
        elif sanitized_name == 'pbe':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_PBE
        elif sanitized_name == 'pbe0':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_PBE0
        elif sanitized_name == 'r2scan':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_R2SCAN
        elif sanitized_name == 'svwn5':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_SVWN5
        elif sanitized_name == 'b97mv':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B97MV
        elif sanitized_name == 'lcwpbe':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_LCWPBE
        elif sanitized_name == 'wb97x':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_WB97X
        elif sanitized_name == 'wb97xv':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_WB97XV
        elif sanitized_name == 'wb97mv':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_WB97MV
        elif functional_name.lower() == 'lc-wpbeh':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_LCWPBEH
        elif functional_name.lower() == 'cam-b3lyp':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_CAMB3LYP
        elif functional_name.lower() == 'hse06':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_HSE06
        elif functional_name.lower() == 'm06':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_M06
        elif functional_name.lower() == 'm06-2x':
            return ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_M062X
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
        elif functional_enum == ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_LCWPBE:
            return 'LC-wPBE'
        elif functional_enum == ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_WB97X:
            return 'wB97X'
        elif functional_enum == ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_WB97XV:
            return 'wB97X-V'
        elif functional_enum == ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_WB97MV:
            return 'wB97M-V'
        elif functional_enum == ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_LCWPBEH:
            return 'LC-wPBEh'
        elif functional_enum == ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_CAMB3LYP:
            return 'CAM-B3LYP'
        elif functional_enum == ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_HSE06:
            return 'HSE06'
        elif functional_enum == ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_M06:
            return 'M06'
        elif functional_enum == ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_M062X:
            return 'M06-2X'
        else:
            raise RuntimeError('Unknown DFT functional')

