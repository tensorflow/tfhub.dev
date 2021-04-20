# Copyright 2021 The TensorFlow Hub Authors. All Rights Reserved.
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
"""Tests for correctly formatted tag config files."""

import pathlib

import tensorflow as tf
import tags_validator


class TagValidationTest(tf.test.TestCase):

  def test_tag_configs(self):
    tools_dir = pathlib.Path(__file__).parent
    tags_dir = tools_dir.parent.joinpath("tags")
    self.assertEmpty(tags_validator.validate_tag_dir(str(tags_dir)))


if __name__ == "__main__":
  tf.test.main()
