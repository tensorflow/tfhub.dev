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
"""Tests for tensorflow_hub.tfhub_dev.tools.tags_validator."""

import os

from absl.testing import parameterized
import tensorflow as tf
import tags_validator

INVALID_FILE = """# Module"""

MINIMAL_TAGS_FILE = """
values:
  - id: mnist
    display_name: MNIST"""

EXTRA_TOP_LEVEL_FIELD = """
name: dataset
values:
  - id: mnist
    display_name: MNIST"""

WRONG_TOP_LEVEL_FIELD = """
items:
  - id: mnist
    display_name: MNIST"""

MISSING_REQUIRED_ITEM_LEVEL_FIELD = """
values:
  - id: mnist"""

WRONG_ITEM_LEVEL_FIELD = """
values:
  - id: mnist
    display_name: MNIST
    extra_field: something"""

WRONG_DUPLICATED_FIELD = """
values:
  - id: mnist
    display_name: dup
    id: another_dataset"""

WRONG_ITEM_LEVEL_FIELD_TYPE = """
values:
  - id: no
    display_name: Norwegian data"""


class ValidatorTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(tf.test.TestCase, self).setUp()
    self.tmp_dir = self.create_tempdir()

  def assert_validation_returns_correct_dict(self, expected_dict):
    """Validating self.tmp_dir should return `expected_dict`."""
    files_to_errors = tags_validator.validate_tag_dir(self.tmp_dir.full_path)
    for v in files_to_errors.values():
      self.assertIsInstance(v, tags_validator.TagDefinitionError)
    self.assertEqual(expected_dict,
                     {k: str(v) for k, v in files_to_errors.items()})

  def get_full_path(self, file_path):
    return os.path.join(self.tmp_dir, file_path)

  def set_content(self, file_path, content):
    full_path = self.get_full_path(file_path)
    tf.io.gfile.makedirs(os.path.dirname(full_path))
    with tf.io.gfile.GFile(full_path, "w") as output_file:
      output_file.write(content)

  @parameterized.parameters(
      ("dataset.yaml", ("id", "display_name")),
      ("language.yaml", ("id", "display_name")),
      ("network_architecture.yaml", ("id", "display_name")),
      ("task.yaml", ("id", "display_name", "domains")))
  def test_get_required_tem_level_keys(self, file_name, expected_keys):
    self.assertCountEqual(
        tags_validator.get_required_item_level_keys(file_name), expected_keys)

  @parameterized.parameters(
      ("dataset.yaml",
       ("id", "display_name", "url", "description", "aggregation_rule")),
      ("language.yaml",
       ("id", "display_name", "url", "description", "aggregation_rule")),
      ("network_architecture.yaml",
       ("id", "display_name", "url", "description", "aggregation_rule")),
      ("task.yaml", ("id", "display_name", "url", "description",
                     "aggregation_rule", "domains")))
  def test_get_supported_item_level_keys(self, file_name, expected_keys):
    self.assertCountEqual(
        tags_validator.get_supported_item_level_keys(file_name), expected_keys)

  def test_parse_good_files(self):
    self.set_content("tags/tag.yml", MINIMAL_TAGS_FILE)
    self.assertEmpty(tags_validator.validate_tag_dir(self.tmp_dir.full_path))

  def test_fail_on_invalid_yaml(self):
    self.set_content("tags/1.md", INVALID_FILE)
    self.assert_validation_returns_correct_dict(
        {f"{self.tmp_dir.full_path}/tags/1.md": "Cannot parse file to YAML."})

  def test_extra_top_level_field(self):
    self.set_content("tags/extra_top_level.yml", EXTRA_TOP_LEVEL_FIELD)
    self.assert_validation_returns_correct_dict({
        f"{self.tmp_dir.full_path}/tags/extra_top_level.yml":
            "Expected top-level keys {'values'} but got ['name', 'values']."
    })

  def test_wrong_top_level_field(self):
    self.set_content("tags/wrong_top_level.yml", WRONG_TOP_LEVEL_FIELD)
    self.assert_validation_returns_correct_dict({
        f"{self.tmp_dir.full_path}/tags/wrong_top_level.yml":
            "Expected top-level keys {'values'} but got ['items']."
    })

  def test_missing_required_item_level_field(self):
    self.set_content("tags/missing_item_level.yml",
                     MISSING_REQUIRED_ITEM_LEVEL_FIELD)
    self.assert_validation_returns_correct_dict({
        f"{self.tmp_dir.full_path}/tags/missing_item_level.yml":
            "Missing required item-level keys: {'display_name'}."
    })

  def test_wrong_item_level_field(self):
    self.set_content("tags/wrong_item_level.yml", WRONG_ITEM_LEVEL_FIELD)
    self.assert_validation_returns_correct_dict({
        f"{self.tmp_dir.full_path}/tags/wrong_item_level.yml":
            "Unsupported item-level keys: {'extra_field'}."
    })

  def test_duplicate_item_level_field(self):
    self.set_content("tags/duplicate_item_level.yml", WRONG_DUPLICATED_FIELD)
    self.assert_validation_returns_correct_dict({
        f"{self.tmp_dir.full_path}/tags/duplicate_item_level.yml":
            "Found duplicate key: id"
    })

  def test_wrong_item_level_field_type(self):
    self.set_content("tags/wrong_item_field_type.yml",
                     WRONG_ITEM_LEVEL_FIELD_TYPE)
    self.assert_validation_returns_correct_dict({
        f"{self.tmp_dir.full_path}/tags/wrong_item_field_type.yml":
            "Found non-string value: "
            "ScalarNode(tag='tag:yaml.org,2002:bool', value='no')"
    })


if __name__ == "__main__":
  tf.test.main()
