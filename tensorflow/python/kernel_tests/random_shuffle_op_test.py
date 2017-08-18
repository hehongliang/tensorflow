# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Functional tests for Random Shuffle Op."""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class RandomShuffleTest(test.TestCase):
    def testInt32VectorCPU(self):
        with self.test_session(use_gpu=False):
            vec = array_ops.placeholder(dtypes.int32, shape=[1000])
            c = random_ops.random_shuffle(vec)

            p = [i for i in range(0, 1000)]
            params = {
                vec: p
            }

            result = c.eval(feed_dict=params);
            print(result.shape)
            print(result)
            self.assertEqual(result.shape, c.get_shape());

class RandomShuffleV2Test(test.TestCase):
    def testInt32VectorCPU(self):
        with self.test_session(use_gpu=False):
            vec = array_ops.placeholder(dtypes.int32, shape=[1000])
            c = random_ops.random_shuffle_v2(vec)

            p = [i for i in range(0, 1000)]
            params = {
                vec: p
            }

            result = c.eval(feed_dict=params);
            print(result.shape)
            print(result)
            self.assertEqual(result.shape, c.get_shape());


class RandomShuffleV3Test(test.TestCase):
    def testInt32VectorCPU(self):
        with self.test_session(use_gpu=False):
            vec = array_ops.placeholder(dtypes.int32, shape=[1000])
            c = random_ops.random_shuffle_v3(vec)

            p = [i for i in range(0, 1000)]
            params = {
                vec: p
            }

            result = c.eval(feed_dict=params);
            print(result.shape)
            print(result)
            self.assertEqual(result.shape, c.get_shape());

    def testInt32VectorGPU(self):
        with self.test_session(use_gpu=True):
            vec = array_ops.placeholder(dtypes.int32, shape=[1000])
            c = random_ops.random_shuffle_v3(vec)

            p = [i for i in range(0, 1000)]
            params = {
                vec: p
            }

            result = c.eval(feed_dict=params);
            print(result.shape)
            print(result)
            self.assertEqual(result.shape, c.get_shape());


if __name__ == "__main__":
    test.main()














