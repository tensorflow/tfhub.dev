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
import textwrap

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


class TagDefinitionFileParserTestUtils(tf.test.TestCase):

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


class TagDefinitionFileParserTest(parameterized.TestCase,
                                  TagDefinitionFileParserTestUtils):

  @parameterized.parameters(
      ("language.yaml", tags_validator.EnumerableTagParser),
      ("dataset.yaml", tags_validator.EnumerableTagParser),
      ("network_architecture.yaml", tags_validator.EnumerableTagParser),
      ("task.yaml", tags_validator.TaskTagParser),
      ("license.yaml", tags_validator.LicenseTagParser),
      ("interactive_visualizer.yaml",
       tags_validator.InteractiveVisualizerTagParser),
      ("colab.yaml", tags_validator.UrlTagParser),
      ("demo.yaml", tags_validator.UrlTagParser))
  def test_create_tag_parser(self, yaml_name, expected_type):
    self.assertIsInstance(
        tags_validator.TagDefinitionFileParser.create_tag_parser(
            f"tags/{yaml_name}"), expected_type)

  def test_create_tag_parser_with_unknown_yaml_file(self):
    self.set_content("tags/unknown.yaml", "")

    self.assert_validation_returns_correct_dict({
        f"{self.tmp_dir.full_path}/tags/unknown.yaml":
            "No parser is registered for unknown.yaml."
    })

  @parameterized.parameters("dataset.yaml", "language.yaml", "task.yaml",
                            "network_architecture.yaml", "license.yaml")
  def test_fail_on_invalid_yaml(self, yaml_name):
    self.set_content(f"tags/{yaml_name}", INVALID_FILE)

    self.assert_validation_returns_correct_dict({
        f"{self.tmp_dir.full_path}/tags/{yaml_name}":
            "Cannot parse file to YAML."
    })

  @parameterized.parameters("dataset.yaml", "language.yaml", "task.yaml",
                            "network_architecture.yaml", "license.yaml")
  def test_duplicate_item_level_field(self, yaml_name):
    self.set_content(f"tags/{yaml_name}", WRONG_DUPLICATED_FIELD)

    self.assert_validation_returns_correct_dict({
        f"{self.tmp_dir.full_path}/tags/{yaml_name}": "Found duplicate key: id"
    })

  @parameterized.parameters("dataset.yaml", "language.yaml", "task.yaml",
                            "network_architecture.yaml", "license.yaml")
  def test_wrong_item_level_field_type(self, yaml_name):
    self.set_content(f"tags/{yaml_name}", WRONG_ITEM_LEVEL_FIELD_TYPE)

    self.assert_validation_returns_correct_dict({
        f"{self.tmp_dir.full_path}/tags/{yaml_name}":
            "Found non-string value: "
            "ScalarNode(tag='tag:yaml.org,2002:bool', value='no')"
    })


class EnumerableTagParserTest(TagDefinitionFileParserTest):

  @parameterized.parameters("dataset.yaml", "language.yaml",
                            "network_architecture.yaml")
  def test_parse_good_enumerable_tag_files(self, yaml_name):
    self.set_content(f"tags/{yaml_name}", MINIMAL_TAGS_FILE)

    self.assertEmpty(tags_validator.validate_tag_dir(self.tmp_dir.full_path))

  @parameterized.parameters("dataset.yaml", "language.yaml", "task.yaml",
                            "network_architecture.yaml", "license.yaml",
                            "interactive_visualizer.yaml")
  def test_extra_top_level_field(self, yaml_name):
    self.set_content(f"tags/{yaml_name}", EXTRA_TOP_LEVEL_FIELD)

    self.assert_validation_returns_correct_dict({
        f"{self.tmp_dir.full_path}/tags/{yaml_name}":
            "Expected top-level keys {'values'} but got ['name', 'values']."
    })

  @parameterized.parameters("dataset.yaml", "language.yaml", "task.yaml",
                            "network_architecture.yaml", "license.yaml",
                            "interactive_visualizer.yaml")
  def test_wrong_top_level_field(self, yaml_name):
    self.set_content(f"tags/{yaml_name}", WRONG_TOP_LEVEL_FIELD)

    self.assert_validation_returns_correct_dict({
        f"{self.tmp_dir.full_path}/tags/{yaml_name}":
            "Expected top-level keys {'values'} but got ['items']."
    })

  @parameterized.parameters(("dataset.yaml", ["display_name"]),
                            ("language.yaml", ["display_name"]),
                            ("network_architecture.yaml", ["display_name"]),
                            ("task.yaml", ["display_name", "domains"]),
                            ("license.yaml", ["display_name"]),
                            ("interactive_visualizer.yaml", ["url_template"]))
  def test_missing_required_item_level_field(self, yaml_name, expected_keys):
    self.set_content(f"tags/{yaml_name}", MISSING_REQUIRED_ITEM_LEVEL_FIELD)

    self.assert_validation_returns_correct_dict({
        f"{self.tmp_dir.full_path}/tags/{yaml_name}":
            f"Missing required item-level keys: {expected_keys}."
    })

  @parameterized.parameters("dataset.yaml", "language.yaml",
                            "network_architecture.yaml")
  def test_wrong_item_level_field(self, yaml_name):
    self.set_content(f"tags/{yaml_name}", WRONG_ITEM_LEVEL_FIELD)

    self.assert_validation_returns_correct_dict({
        f"{self.tmp_dir.full_path}/tags/{yaml_name}":
            "Unsupported item-level keys: {'extra_field'}."
    })

  @parameterized.parameters("dataset.yaml", "language.yaml", "license.yaml",
                            "network_architecture.yaml")
  def test_invalid_item_id(self, yaml_name):
    yaml_content = textwrap.dedent("""\
      values:
        - id: Bad_id
          display_name: The human-readable name""")
    self.set_content(f"tags/{yaml_name}", yaml_content)

    self.assert_validation_returns_correct_dict({
        f"{self.tmp_dir.full_path}/tags/{yaml_name}":
            r"The value of an id must match [a-z-_\d\.]+ but was Bad_id."
    })


class TaskFileParserTest(TagDefinitionFileParserTest):

  def test_parse_good_task_yaml_file(self):
    content = textwrap.dedent("""\
      values:
        - id: image-detection
          display_name: Image detection
          domains:
            - image
    """)
    self.set_content("tags/task.yaml", content)

    self.assertEmpty(tags_validator.validate_tag_dir(self.tmp_dir.full_path))


class LicenseFileParserTest(TagDefinitionFileParserTest):

  def test_parse_license_with_all_fields(self):
    content = textwrap.dedent("""\
      values:
        - id: apache-2.0
          display_name: Apache-2.0
          url: https://opensource.org/licenses/Apache-2.0
    """)
    self.set_content("tags/license.yaml", content)

    self.assertEmpty(tags_validator.validate_tag_dir(self.tmp_dir.full_path))

  def test_parse_license_without_url_field(self):
    content = textwrap.dedent("""\
      values:
        - id: apache-2.0
          display_name: Apache-2.0
    """)
    self.set_content("tags/license.yaml", content)

    self.assertEmpty(tags_validator.validate_tag_dir(self.tmp_dir.full_path))


class InteractiveVisualizerTagParserTest(TagDefinitionFileParserTest):

  def test_parse_visualizer_from_valid_yaml(self):
    content = textwrap.dedent("""\
      values:
        - id: vision
          url_template: "https://www.gstatic.com/index.html?\\
            model={MODEL_HANDLE}"
    """)
    self.set_content("tags/interactive_visualizer.yaml", content)

    self.assertEmpty(tags_validator.validate_tag_dir(self.tmp_dir.full_path))

  def test_fail_on_unsupported_variable(self):
    content = textwrap.dedent("""\
      values:
        - id: vision
          url_template: "https://www.gstatic.com/index.html?\\
            model={UNKNOWN_VARIABLE}"
    """)
    self.set_content("tags/interactive_visualizer.yaml", content)

    self.assert_validation_returns_correct_dict({
        f"{self.tmp_dir.full_path}/tags/interactive_visualizer.yaml":
            "Only substituting ['MODEL_HANDLE', 'MODEL_NAME', 'MODEL_URL', "
            "'PUBLISHER_ICON_URL', 'PUBLISHER_NAME'] is allowed: "
            "https://www.gstatic.com/index.html?model={UNKNOWN_VARIABLE}."
    })

  def test_fail_on_invalid_url(self):
    content = textwrap.dedent("""\
      values:
        - id: vision
          url_template: "https://www.gstatic.com]index.html?\\
            model={MODEL_HANDLE}"
    """)
    self.set_content("tags/interactive_visualizer.yaml", content)

    self.assert_validation_returns_correct_dict({
        f"{self.tmp_dir.full_path}/tags/interactive_visualizer.yaml":
            "https://www.gstatic.com]index.html?model=placeholder is not a "
            "valid URL."
    })

  def test_fail_on_spaces_in_url(self):
    content = textwrap.dedent("""\
      values:
        - id: vision
          url_template: "https://www.gstatic.com/index.html?
            model={MODEL_HANDLE}"
    """)
    self.set_content("tags/interactive_visualizer.yaml", content)

    self.assert_validation_returns_correct_dict({
        f"{self.tmp_dir.full_path}/tags/interactive_visualizer.yaml":
            "https://www.gstatic.com/index.html? model=placeholder must not "
            "contain spaces."
    })

  def test_fail_on_unsecure_url(self):
    content = textwrap.dedent("""\
      values:
        - id: vision
          url_template: "http://page.com/index.html?\\
            model={MODEL_HANDLE}"
    """)
    self.set_content("tags/interactive_visualizer.yaml", content)

    self.assert_validation_returns_correct_dict({
        f"{self.tmp_dir.full_path}/tags/interactive_visualizer.yaml":
            "http://page.com/index.html?model=placeholder is not a HTTPS URL."
    })

  @parameterized.parameters("index.html", "page.com")
  def test_fail_on_missing_domain_or_path_in_url(self, url):
    content = textwrap.dedent(f"""\
      values:
        - id: vision
          url_template: https://{url}
    """)
    self.set_content("tags/interactive_visualizer.yaml", content)

    self.assert_validation_returns_correct_dict({
        f"{self.tmp_dir.full_path}/tags/interactive_visualizer.yaml":
            f"https://{url} must specify a domain and a path."
    })

  @parameterized.parameters(
      "https://storage.googleapis.com/interactive_visualizer/",
      "https://storage.googleapis.com/tfhub-visualizers/",
      "https://www.gstatic.com/")
  def test_pass_on_allowed_url_prefix(self, url_prefix):
    content = textwrap.dedent(f"""\
      values:
        - id: vision
          url_template: {url_prefix}index.html
    """)
    self.set_content("tags/interactive_visualizer.yaml", content)

    self.assertEmpty(tags_validator.validate_tag_dir(self.tmp_dir.full_path))

  def test_fail_on_disallowed_url_prefix(self):
    content = textwrap.dedent("""\
      values:
        - id: vision
          url_template: https://mypage.com/index.html
    """)
    self.set_content("tags/interactive_visualizer.yaml", content)

    self.assert_validation_returns_correct_dict(
        {
            f"{self.tmp_dir.full_path}/tags/interactive_visualizer.yaml": (
                "URL needs to start with any of"
                " ['https://storage.googleapis.com/interactive_visualizer/',"
                " 'https://storage.googleapis.com/tfhub-visualizers/',"
                " 'https://www.gstatic.com/'] but was"
                " https://mypage.com/index.html."
            )
        }
    )


@parameterized.parameters("colab.yaml", "demo.yaml")
class UrlTagParserTest(TagDefinitionFileParserTestUtils):

  def test_pass_on_good_content(self, yaml_name):
    content = textwrap.dedent("""\
    format: url
    required_domain: page.com
    """)
    self.set_content(f"tags/{yaml_name}", content)

    self.assertEmpty(tags_validator.validate_tag_dir(self.tmp_dir.full_path))

  def test_fail_on_wrong_top_level_key(self, yaml_name):
    self.set_content(f"tags/{yaml_name}", "regex: url")

    self.assert_validation_returns_correct_dict({
        f"{self.tmp_dir.full_path}/tags/{yaml_name}":
            "Missing required top-level keys: {'format'}."
    })

  def test_fail_on_unsupported_format(self, yaml_name):
    self.set_content(f"tags/{yaml_name}", "format: website")

    self.assert_validation_returns_correct_dict({
        f"{self.tmp_dir.full_path}/tags/{yaml_name}":
            "Expected 'format' value to be one of {'url'} but was 'website'."
    })

  def test_fail_on_unsupported_top_level_key(self, yaml_name):
    self.set_content(f"tags/{yaml_name}", "format: url\nregex: .*")

    self.assert_validation_returns_correct_dict({
        f"{self.tmp_dir.full_path}/tags/{yaml_name}":
            "Unsupported top-level keys: {'regex'}."
    })


if __name__ == "__main__":
  tf.test.main()
