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
import textwrap

from absl.testing import parameterized
import tensorflow as tf
import yaml_parser
import yaml


DEFAULT_ARCHITECTURE_CONTENT = """
values:
  - id: bert
    display_name: BERT
  - id: transformer
    display_name: Transformer"""

DEFAULT_DATASET_CONTENT = """
values:
  - id: mnist
    display_name: MNIST
  - id: imagenet
    display_name: ImageNet"""

DEFAULT_LANGUAGE_CONTENT = """
values:
  - id: en
    display_name: English
  - id: fr
    display_name: French"""

DEFAULT_LICENSE_YAML = """
values:
  - id: apache-2.0
    display_name: Apache-2.0
    url: https://opensource.org/licenses/Apache-2.0"""

DEFAULT_TASK_CONTENT = """
values:
  - id: text-embedding
    display_name: Text embedding
    domains:
      - text
  - id: image-transfer
    display_name: Image transfer
    domains:
      - image"""

DEFAULT_VISUALIZER_CONTENT = """
values:
  - id: spice
    url_template: https://www.gstatic.com/aihub/tfhub/demos/spice.html"""

SIMPLE_DATASET_CONTENT = """
values:
  - id: mnist
    display_name: MNIST"""

SIMPLE_ARCHITECTURE_CONTENT = """
values:
  - id: bert
    display_name: BERT"""

SIMPLE_LANGUAGE_CONTENT = """
values:
  - id: en
    display_name: English"""

SIMPLE_LICENSE_YAML = """
values:
  - id: apache-2.0
    display_name: Apache-2.0
    url: https://opensource.org/licenses/Apache-2.0
  - id: custom
    display_name: custom"""

SIMPLE_TASK_CONTENT = """
values:
  - id: text-embedding
    display_name: Text embedding
    domains:
      - text"""

SIMPLE_VISUALIZER_CONTENT = """
values:
  - id: spice
    url_template: https://www.gstatic.com/aihub/tfhub/demos/spice.html
  - id: vision
    url_template: "https://storage.googleapis.com/tfhub-visualizers.html"
"""


class AbstractYamlParserTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(tf.test.TestCase, self).setUp()
    self.tmp_dir = self.create_tempdir()
    self.architecture_key = "network-architecture"
    self.dataset_key = "dataset"
    self.language_key = "language"
    self.license_key = "license"
    self.task_key = "task"
    self.visualizer_key = "interactive-model-name"

  def _create_tag_files(self,
                        architecture_content=DEFAULT_ARCHITECTURE_CONTENT,
                        dataset_content=DEFAULT_DATASET_CONTENT,
                        language_content=DEFAULT_LANGUAGE_CONTENT,
                        license_content=DEFAULT_LICENSE_YAML,
                        task_content=DEFAULT_TASK_CONTENT,
                        visualizer_content=DEFAULT_VISUALIZER_CONTENT):
    for tag_key, content in zip([
        self.architecture_key, self.dataset_key, self.language_key,
        self.license_key, self.task_key, self.visualizer_key
    ], [
        architecture_content, dataset_content, language_content,
        license_content, task_content, visualizer_content
    ]):
      self.create_tempfile(
          os.path.join(self.tmp_dir, yaml_parser.TAG_TO_YAML_MAP[tag_key]),
          content)

  def test_from_tag_name_with_unknown_tag_fails(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, "No supported parser found for tag unknown-tag."):
      yaml_parser.AbstractYamlParser.from_tag_name(self.tmp_dir, "unknown-tag")


class EnumerableYamlParserTest(AbstractYamlParserTest):

  @parameterized.parameters(("dataset", "dataset_content"),
                            ("network-architecture", "architecture_content"),
                            ("language", "language_content"),
                            ("license", "license_content"),
                            ("task", "task_content"),
                            ("interactive-model-name", "visualizer_content"))
  def test_invalid_yaml_file_raises_parser_error(self, tag_name,
                                                 parameter_name):
    self._create_tag_files(**{parameter_name: "foo\n:"})
    parser = yaml_parser.EnumerableYamlParser(self.tmp_dir, tag_name)

    with self.assertRaises(yaml.parser.ParserError):
      parser._get_supported_values()

  @parameterized.parameters("dataset", "network-architecture", "language",
                            "license", "task", "interactive-model-name")
  def test_non_existent_yaml_file(self, tag_name):
    parser = yaml_parser.EnumerableYamlParser(self.create_tempdir(), tag_name)

    with self.assertRaises(FileNotFoundError):
      parser._get_supported_values()

  @parameterized.parameters(
      (DEFAULT_ARCHITECTURE_CONTENT, {"bert", "transformer"}),
      (DEFAULT_DATASET_CONTENT, {"mnist", "imagenet"}),
      (DEFAULT_LANGUAGE_CONTENT, {"en", "fr"}),
      (DEFAULT_LICENSE_YAML, {"apache-2.0"}),
      (DEFAULT_TASK_CONTENT, {"text-embedding", "image-transfer"}),
      (DEFAULT_VISUALIZER_CONTENT, {"spice"}))
  def test_build_tag_config_from_yaml(self, content, expected_ids):
    yaml_config = yaml.safe_load(content)
    tag_values_validator = yaml_parser.EnumerableTagValuesValidator.from_yaml(
        yaml_config)

    self.assertCountEqual(tag_values_validator.values, [
        yaml_parser.EnumerableTagValue(id=expected_id)
        for expected_id in expected_ids
    ])

  def test_build_tag_config_with_missing_values_key(self):
    yaml_config = yaml.safe_load("""\
      value:
        - id: bert""")

    with self.assertRaisesWithLiteralMatch(
        ValueError, "YAML config should contain `values` key "
        "but was {'value': [{'id': 'bert'}]}."):
      yaml_parser.EnumerableTagValuesValidator.from_yaml(yaml_config)

  @parameterized.parameters("mnist", "mobilenetv2", "mobilenet-v2", "squad-2.0")
  def test_valid_item_id(self, id_value):
    yaml_content = textwrap.dedent("""\
      values:
        - id: %s
          display_name: The name""" % id_value)
    yaml_config = yaml.safe_load(yaml_content)
    tag_values_validator = yaml_parser.EnumerableTagValuesValidator.from_yaml(
        yaml_config)

    self.assertCountEqual(tag_values_validator.values,
                          [yaml_parser.EnumerableTagValue(id=id_value)])

  @parameterized.parameters(
      ("dataset", {"dataset_content": SIMPLE_DATASET_CONTENT}, {"mnist"}),
      ("interactive-model-name", {
          "visualizer_content": SIMPLE_VISUALIZER_CONTENT},
       {"spice", "vision"}),
      ("language", {"language_content": SIMPLE_LANGUAGE_CONTENT}, {"en"}),
      ("license", {
          "license_content": SIMPLE_LICENSE_YAML}, {"apache-2.0", "custom"}),
      ("network-architecture", {
          "architecture_content": SIMPLE_ARCHITECTURE_CONTENT}, {"bert"}),
      ("task", {"task_content": SIMPLE_TASK_CONTENT}, {"text-embedding"}))
  def test_get_supported_values(self, tag_key, content_params, expected_values):
    self._create_tag_files(**content_params)

    parser = yaml_parser.EnumerableYamlParser(self.tmp_dir, tag_key)

    self.assertEqual(parser._get_supported_values(), expected_values)


if __name__ == "__main__":
  tf.test.main()
