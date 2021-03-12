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
"""Tests for tfhub_dev.tools.filesystem_utils."""

import os
import pathlib
import tensorflow as tf
from tools import filesystem_utils


class FilesystemUtilsTest(tf.test.TestCase):

  def test_recursive_list_dir_with_empty_directory(self):
    root_dir = self.create_tempdir()
    files = list(filesystem_utils.recursive_list_dir(root_dir))
    self.assertEmpty(files)

  def test_recursive_list_dir_with_empty_nested_directories(self):
    root_dir = self.create_tempdir()
    pathlib.Path(os.path.join(root_dir, "a", "model1")).mkdir(parents=True)
    pathlib.Path(os.path.join(root_dir, "a", "model2")).mkdir(parents=True)
    files = list(filesystem_utils.recursive_list_dir(root_dir))
    self.assertEmpty(files)

  def test_recursive_list_dir_with_nested_files(self):
    root_dir = self.create_tempdir()
    file_paths = [
        os.path.join(root_dir, *p)
        for p in [("a", "1.md"), ("a", "2.md"), ("a", "b", "1.md")]
    ]
    for path in file_paths:
      self.create_tempfile(path)
    actual_files = list(filesystem_utils.recursive_list_dir(root_dir))
    self.assertCountEqual(actual_files, file_paths)

  def test_get_content_empty_file(self):
    temp_file = self.create_tempfile()
    self.assertEmpty(filesystem_utils.get_content(temp_file))

  def test_get_content_populated_file(self):
    content = "# Module"
    temp_file = self.create_tempfile(content=content)
    self.assertEqual(content, filesystem_utils.get_content(temp_file))


if __name__ == "__main__":
  tf.test.main()
