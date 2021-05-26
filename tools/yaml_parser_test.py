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
"""Tests for tensorflow_hub.tfhub_dev.tools.yaml_parser."""

import os

import tensorflow as tf
import yaml_parser
import yaml


class YamlParserTest(tf.test.TestCase):

  def setUp(self):
    super(tf.test.TestCase, self).setUp()
    self.tmp_dir = self.create_tempdir()
    self.language_key = "language"
    self.dataset_key = "dataset"

  def _create_tag_files(self, dataset_content=None, language_content=None):
    if dataset_content is None:
      dataset_content = """
      values:
        - id: mnist
          display_name: MNIST
        - id: imagenet
          display_name: ImageNet"""
    if language_content is None:
      language_content = """
      values:
        - id: en
          display_name: English"""
    self.create_tempfile(
        file_path=os.path.join(self.tmp_dir,
                               yaml_parser.TAG_TO_YAML_MAP[self.language_key]),
        content=language_content)
    self.create_tempfile(
        file_path=os.path.join(self.tmp_dir,
                               yaml_parser.TAG_TO_YAML_MAP[self.dataset_key]),
        content=dataset_content)

  def test_get_supported_values(self):
    self._create_tag_files()
    parser = yaml_parser.YamlParser(self.tmp_dir)

    self.assertEqual(parser.get_supported_values(self.language_key), {"en"})
    self.assertEqual(
        parser.get_supported_values(self.dataset_key), {"mnist", "imagenet"})

  def test_invalid_yaml_file(self):
    self._create_tag_files(dataset_content="foo\n:", language_content="foo\n:")
    parser = yaml_parser.YamlParser(self.tmp_dir)

    with self.assertRaises(yaml.parser.ParserError):
      parser.get_supported_values(self.language_key)

  def test_non_existent_yaml_file(self):
    parser = yaml_parser.YamlParser(self.create_tempdir())

    with self.assertRaises(FileNotFoundError):
      parser.get_supported_values(self.language_key)

  def test_non_existent_yaml_tag(self):
    self._create_tag_files()
    parser = yaml_parser.YamlParser(self.tmp_dir)

    with self.assertRaisesWithLiteralMatch(
        ValueError, "No supported ids found for tag non-existent-tag."):
      parser.get_supported_values("non-existent-tag")


if __name__ == "__main__":
  tf.test.main()
