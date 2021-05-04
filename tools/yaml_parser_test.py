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

  def test_get_supported_languages(self):
    yaml_content = """
    values:
      - id: en
        display_name: English"""
    self.create_tempfile(
        file_path=os.path.join(self.tmp_dir, yaml_parser.LANGUAGE_YAML),
        content=yaml_content)
    parser = yaml_parser.YamlParser(self.tmp_dir)
    self.assertEqual(parser.get_supported_languages(), {"en"})

  def test_invalid_yaml_file(self):
    self.create_tempfile(
        file_path=os.path.join(self.tmp_dir, yaml_parser.LANGUAGE_YAML),
        content="foo\n:")
    parser = yaml_parser.YamlParser(self.tmp_dir)
    with self.assertRaises(yaml.parser.ParserError):
      parser.get_supported_languages()

  def test_non_existent_yaml_file(self):
    parser = yaml_parser.YamlParser(self.create_tempdir())
    with self.assertRaises(FileNotFoundError):
      parser.get_supported_languages()


if __name__ == "__main__":
  tf.test.main()
