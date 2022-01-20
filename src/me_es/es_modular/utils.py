# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
#
# THIS IS NOT THE ORIGINAL VERSION OF THE FILE.
#
# Last modified 2021-12-02
import os

import numpy as np


def fix_probas(probs):
    test_passed = False
    for ii in range(probs.shape[0]):
        if probs.sum() < 1:
            if probs[ii] + (1 - probs.sum()) <= 1:
                probs[ii] += (1 - probs.sum())
        elif probs.sum() > 1:
            if probs[ii] - (probs.sum() - 1) >= 0:
                probs[ii] -= (probs.sum() - 1)
        if probs.sum() == 1 and np.all(probs >= 0) and np.all(probs <= 1):
            test_passed = True
            break
    return probs, test_passed
