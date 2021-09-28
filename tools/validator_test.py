# Copyright 2020 The TensorFlow Hub Authors. All Rights Reserved.
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
"""Tests for tensorflow_hub.tfhub_dev.tools.validator."""

import contextlib
import os
import textwrap
from typing import Optional
from unittest import mock
import urllib.request

from absl.testing import parameterized
import tensorflow as tf
import filesystem_utils
import validator
import yaml_parser

_GOOGLE_PUBLISHER = "google"
_MODEL_HANDLE = f"{_GOOGLE_PUBLISHER}/text-embedding-model/1"
_RELATIVE_SAVED_MODEL_PATH = f"{_MODEL_HANDLE}.md"
_RELATIVE_TFJS_PATH = f"{_GOOGLE_PUBLISHER}/text-embedding-model/tfjs/1.md"
_RELATIVE_LITE_PATH = f"{_GOOGLE_PUBLISHER}/text-embedding-model/lite/1.md"
_RELATIVE_CORAL_PATH = f"{_GOOGLE_PUBLISHER}/text-embedding-model/coral/1.md"
_ABSOLUTE_COLLECTION_PATH = (
    "root/assets/docs/google/collections/text-embedding-collection/1.md")

ARCHITECTURE_YAML = """
values:
  - id: bert
    display_name: BERT
  - id: transformer
    display_name: Transformer"""

COLAB_YAML = """
format: url
required_domain: colab.research.google.com"""

DATASET_YAML = """
values:
  - id: mnist
    display_name: MNIST
  - id: wikipedia
    display_name: Wikipedia"""

DEMO_YAML = """
format: url"""

LANGUAGE_YAML = """
values:
  - id: en
    display_name: English
  - id: fr
    display_name: French"""

LICENSE_YAML = """
values:
  - id: apache-2.0
    display_name: Apache-2.0
    url: https://opensource.org/licenses/Apache-2.0"""

TASK_YAML = """
values:
  - id: text-embedding
    display_name: Text embedding
    domains:
      - text
  - id: text-classification
    display_name: Text classification
    domains:
      - text"""

VISUALIZER_YAML = """
values:
  - id: spice
    url_template: https://www.gstatic.com/aihub/tfhub/demos/spice.html
  - id: vision
    url_template: "https://storage.googleapis.com/tfhub-visualizers.html"
"""

TAG_FILE_NAME_TO_CONTENT_MAP = {
    "colab.yaml": COLAB_YAML,
    "dataset.yaml": DATASET_YAML,
    "demo.yaml": DEMO_YAML,
    "interactive_visualizer.yaml": VISUALIZER_YAML,
    "language.yaml": LANGUAGE_YAML,
    "license.yaml": LICENSE_YAML,
    "network_architecture.yaml": ARCHITECTURE_YAML,
    "task.yaml": TASK_YAML
}

LEGACY_VALUE = "legacy"

MINIMAL_SAVED_MODEL_TEMPLATE = f"""# Module {_MODEL_HANDLE}
Simple description spanning
multiple lines.

<!-- asset-path: %s -->
<!-- task:   text-embedding   -->
<!-- fine-tunable:true -->
<!-- format: saved_model_2 -->

## Overview
"""

SAVED_MODEL_OPTIONAL_TAG_TEMPLATE = """# Module google/text-embedding-model/1
Simple description spanning
multiple lines.

<!-- asset-path: /path/to/model.tar.gz -->
<!-- task:   text-embedding   -->
<!-- fine-tunable:true -->
<!-- format: saved_model_2 -->
<!-- {tag_key}: {tag_value} -->
"""

SAVED_MODEL_WITHOUT_DESCRIPTION = f"""# Module {_MODEL_HANDLE}

<!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
<!-- format: saved_model_2 -->

## Overview
"""

SAVED_MODEL_WITHOUT_DESCRIPTION_WITHOUT_LINEBREAK = textwrap.dedent(f"""\
# Module {_MODEL_HANDLE}
<!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
<!-- format: saved_model_2 -->

## Overview
""")

SAVED_MODEL_OPTIONAL_TAGS_TEMPLATE = """# Module google/text-embedding-model/1
One line description.
<!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
<!-- task: text-classification -->
<!-- task: text-embedding -->
<!-- {tag_key_1}: {tag_value_1} -->
<!-- {tag_key_2}: {tag_value_2} -->
<!-- fine-tunable: true -->
<!-- format: saved_model_2 -->

## Overview
"""

MAXIMAL_SAVED_MODEL_TEMPLATE = f"""# Module {_MODEL_HANDLE}
Simple description spanning
multiple lines.

<!-- asset-path: %s -->
<!-- task:   text-embedding   -->
<!-- fine-tunable:true -->
<!-- format: saved_model_2 -->
<!-- dataset: mnist -->
<!-- interactive-visualizer: vision -->
<!-- language: en -->
<!-- network-architecture: bert -->
<!-- license: apache-2.0 -->
<!-- colab: https://colab.research.google.com/mycolab.ipynb -->

## Overview
"""

MINIMAL_PLACEHOLDER = f"""# Placeholder {_MODEL_HANDLE}
Simple description spanning
multiple lines.

<!-- task:   text-embedding   -->
"""

PLACEHOLDER_OPTIONAL_TAG_TEMPLATE = """# Placeholder google/text-embedding-model/1
Simple description spanning
multiple lines.

<!-- task:   text-embedding   -->
<!-- {tag_key}: {tag_value} -->
"""

MAXIMAL_PLACEHOLDER = f"""# Placeholder {_MODEL_HANDLE}
Simple description spanning
multiple lines.

<!-- dataset: mnist -->
<!-- fine-tunable:true -->
<!-- interactive-visualizer: vision -->
<!-- language: en -->
<!-- task:   text-embedding   -->
<!-- network-architecture: bert -->
<!-- license: apache-2.0 -->
"""

LITE_OPTIONAL_TAG_TEMPLATE = """# Lite google/text-embedding-model/1
Simple description spanning
multiple lines.

<!-- asset-path: /path/to/model.tflite -->
<!-- parent-model: google/text-embedding-model/1 -->
<!-- {tag_key}: {tag_value} -->

## Overview
"""

CORAL_OPTIONAL_TAG_TEMPLATE = """# Coral google/text-embedding-model/1
Simple description spanning
multiple lines.

<!-- asset-path: /path/to/model.tflite -->
<!-- parent-model: google/text-embedding-model/1 -->
<!-- {tag_key}: {tag_value} -->

## Overview
"""

TFJS_OPTIONAL_TAG_TEMPLATE = """# Tfjs google/text-embedding-model/1
Simple description spanning
multiple lines.

<!-- asset-path: https://page.com/model.tar.gz -->
<!-- parent-model:   google/text-embedding-model/1   -->
<!-- {tag_key}: {tag_value} -->

## Overview"""

MINIMAL_COLLECTION = """# Collection google/text-embedding-collection/1
Simple description spanning
multiple lines.

<!-- task: text-embedding -->

## Overview

Add links to collections to reference models:
https://tfhub.dev/google/bert/1
"""

COLLECTION_OPTIONAL_TAG_TEMPLATE = """# Collection google/text-embedding-collection/1
Simple description spanning
multiple lines.

<!-- task: text-embedding -->
<!-- {tag_key}: {tag_value} -->

## Overview
"""

COLLECTION_CONTENT_TEMPLATE = """# Collection google/text-embedding-collection/1
Simple description spanning
multiple lines.

<!-- task: text-embedding -->

## Overview
{content}
"""

MAXIMAL_COLLECTION = """# Collection google/text-embedding-collection/1
Simple description spanning
multiple lines.

<!-- task: text-embedding -->
<!-- dataset: mnist -->
<!-- language: en -->
<!-- network-architecture: bert -->

## Overview

Add links to collections to reference models:
https://tfhub.dev/google/bert/1
"""

PUBLISHER_HANDLE_TEMPLATE = """# Publisher %s
The publisher name.

[![Icon URL]](https://path/to/icon.png)

## Overview
"""


class MockUrlOpen(contextlib.AbstractContextManager):
  """Mock to replace the urlopen context manager with a GFile file stream."""

  def __init__(self, url: str) -> None:
    self.file_obj = tf.io.gfile.GFile(url, "rb")

  def __enter__(self) -> tf.io.gfile.GFile:
    return self.file_obj

  def __exit__(self, *args) -> None:
    self.file_obj.close()


class ValidatorTest(parameterized.TestCase, tf.test.TestCase):

  def _get_parser_for_validating_saved_model_file(
      self, extra_file_name: str,
      markdown_file_path: str) -> validator.DocumentationParser:
    """Returns a parser for checking a Markdown file pointing to an archive."""
    self.save_dummy_model_archive(self.model_path, extra_file_name)
    self.minimal_markdown = MINIMAL_SAVED_MODEL_TEMPLATE % self.model_path
    self.set_content(markdown_file_path, self.minimal_markdown)
    return validator.DocumentationParser(self.tmp_root_dir, self.tmp_docs_dir,
                                         self.parser_by_tag)

  def setUp(self):
    super(tf.test.TestCase, self).setUp()
    self.validation_config = validator.ValidationConfig()
    self.tmp_dir = self.create_tempdir()
    self.tmp_root_dir = os.path.join(self.tmp_dir, "root")
    self.tmp_docs_dir = os.path.join(self.tmp_root_dir, "assets", "docs")
    self.model_path = os.path.join(self.tmp_dir, "model_1.tar.gz")
    self.save_dummy_model_archive(self.model_path)
    self.minimal_markdown = MINIMAL_SAVED_MODEL_TEMPLATE % self.model_path
    self.maximal_markdown = MAXIMAL_SAVED_MODEL_TEMPLATE % self.model_path
    self.markdown_file_path = "root/assets/docs/google/models/text-embedding-model/1.md"
    for file_name, content in TAG_FILE_NAME_TO_CONTENT_MAP.items():
      self.set_content(f"root/tags/{file_name}", content)
    self.set_up_publisher_page(_GOOGLE_PUBLISHER)
    self.asset_path_modified = mock.patch.object(
        validator, "_is_asset_path_modified", return_value=True)
    self.enumerable_parser = yaml_parser.EnumerableYamlParser(
        self.tmp_root_dir, "language")
    self.parser_by_tag = {"language": self.enumerable_parser}
    self.asset_path_modified.start()
    self.addCleanup(self.asset_path_modified.stop)

  def get_full_path(self, file_path):
    return os.path.join(self.tmp_dir, file_path)

  def set_content(self, file_path, content):
    full_path = self.get_full_path(file_path)
    tf.io.gfile.makedirs(os.path.dirname(full_path))
    with tf.io.gfile.GFile(full_path, "w") as output_file:
      output_file.write(content)

  def set_up_publisher_page(self, publisher):
    self.set_content(f"root/assets/docs/{publisher}/{publisher}.md",
                     PUBLISHER_HANDLE_TEMPLATE % publisher)

  def save_dummy_model_archive(self,
                               path: str,
                               extra_file: Optional[str] = None) -> None:

    class MultiplyTimesTwoModel(tf.train.Checkpoint):
      """Callable model that multiplies by two."""

      @tf.function(
          input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
      def __call__(self, x):
        return x * 2

    model = MultiplyTimesTwoModel()
    temp_dir = self.create_tempdir().full_path
    tf.saved_model.save(model, temp_dir)
    if extra_file is not None:
      os.makedirs(
          os.path.join(temp_dir, os.path.dirname(extra_file)), exist_ok=True)
      with open(os.path.join(temp_dir, extra_file), "wt") as f:
        f.write("This file is not necessary.")
    filesystem_utils.compress_local_directory_to_archive(temp_dir, path)

  @parameterized.parameters(
      ("# Module google/ALBERT/1", validator.SavedModelParsingPolicy),
      ("# Placeholder google/ALBERT/1", validator.PlaceholderParsingPolicy),
      ("# Lite google/ALBERT/1", validator.LiteParsingPolicy),
      ("# Tfjs google/ALBERT/1", validator.TfjsParsingPolicy),
      ("# Coral google/ALBERT/1", validator.CoralParsingPolicy),
      ("# Publisher google", validator.PublisherParsingPolicy),
      ("# Collection google/experts/1", validator.CollectionParsingPolicy))
  def test_get_policy_from_string(self, document_string, expected_policy):
    self.assertIsInstance(
        validator.ParsingPolicy.from_string(document_string,
                                            self.parser_by_tag),
        expected_policy)

  def test_fail_getting_policy_from_unknown_string(self):
    with self.assertRaisesRegex(
        validator.MarkdownDocumentationError,
        ".*Instead the first line is '# Newmodel google/ALBERT/1'"):
      validator.ParsingPolicy.from_string("# Newmodel google/ALBERT/1",
                                          self.parser_by_tag)

  @parameterized.parameters(
      (validator.SavedModelParsingPolicy,
       ["ROOT/google/bert/1.md", "ROOT/google/models/bert/1.md"]),
      (validator.PlaceholderParsingPolicy,
       ["ROOT/google/bert/1.md", "ROOT/google/models/bert/1.md"]),
      (validator.LiteParsingPolicy,
       ["ROOT/google/bert/lite/1.md", "ROOT/google/models/bert/lite/1.md"]),
      (validator.TfjsParsingPolicy,
       ["ROOT/google/bert/tfjs/1.md", "ROOT/google/models/bert/tfjs/1.md"]),
      (validator.CoralParsingPolicy,
       ["ROOT/google/bert/coral/1.md", "ROOT/google/models/bert/coral/1.md"]),
      (validator.PublisherParsingPolicy, ["ROOT/google/google.md"]))
  def test_allowed_paths_for_fixed_version(self, policy_class, expected_paths):
    policy = policy_class({}, "google", "bert", "1")
    self.assertCountEqual(policy.get_allowed_file_paths("ROOT"), expected_paths)

  @parameterized.parameters(
      ("https://tfhub.dev/google/albert_base",
       validator.SavedModelParsingPolicy({}, "google", "albert_base", "*")),
      ("https://tfhub.dev/google/albert_base/3",
       validator.SavedModelParsingPolicy({}, "google", "albert_base", "3")),
      ("https://tfhub.dev/tensorflow/lite-model/densenet/1/metadata",
       validator.LiteParsingPolicy({}, "tensorflow", "densenet/1/metadata",
                                   "*")),
      ("https://tfhub.dev/tensorflow/lite-model/densenet/1/metadata/2",
       validator.LiteParsingPolicy({}, "tensorflow", "densenet/1/metadata",
                                   "2")),
      ("https://tfhub.dev/google/tfjs-model/spice/1/default",
       validator.TfjsParsingPolicy({}, "google", "spice/1/default", "*")),
      ("https://tfhub.dev/google/tfjs-model/spice/1/default/2",
       validator.TfjsParsingPolicy({}, "google", "spice/1/default", "2")),
      ("https://tfhub.dev/google/coral-model/yamnet/classification/coral",
       validator.CoralParsingPolicy({}, "google", "yamnet/classification/coral",
                                    "*")),
      ("https://tfhub.dev/google/coral-model/yamnet/classification/coral/2",
       validator.CoralParsingPolicy({}, "google", "yamnet/classification/coral",
                                    "2")))
  def test_tfhubdev_regex_yields_correct_groupdict(self, model_url,
                                                   expected_policy):
    policies = list(validator._get_policies_for_line_with_model_urls(model_url))
    self.assertLen(policies, 1)
    actual_policy = policies[0]
    self.assertEqual(actual_policy._publisher, expected_policy._publisher)
    self.assertEqual(actual_policy._model_name, expected_policy._model_name)
    self.assertEqual(actual_policy._model_version,
                     expected_policy._model_version)

  def test_markdown_parsed_saved_model(self):
    empty_second_line = textwrap.dedent(f"""\
       # Module {_MODEL_HANDLE}

       Simple description spanning
       multiple lines.

       <!-- asset-path: {self.model_path} -->
       <!-- task:   text-embedding   -->
       <!-- fine-tunable:true -->
       <!-- format: saved_model_2 -->

       ## Overview""")
    for markdown in [
        self.minimal_markdown, self.maximal_markdown, empty_second_line
    ]:
      self.set_content(self.markdown_file_path, markdown)

      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  @parameterized.parameters(MINIMAL_PLACEHOLDER, MAXIMAL_PLACEHOLDER)
  def test_markdown_parsed_placeholder(self, markdown):
    self.set_content(self.markdown_file_path, markdown)

    validator.validate_documentation_dir(
        validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_minimal_markdown_parsed_lite(self):
    lite_model = os.path.join(self.tmp_dir, "model.tflite")
    content = textwrap.dedent(f"""\
      # Lite {_MODEL_HANDLE}
      Simple description spanning
      multiple lines.

      <!-- asset-path: {lite_model} -->
      <!-- parent-model: google/text-embedding-model/1 -->

      ## Overview""")
    self.set_content(
        os.path.join(self.tmp_docs_dir, _RELATIVE_LITE_PATH), content)

    validator.validate_documentation_dir(
        validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_minimal_markdown_parsed_tfjs(self):
    content = textwrap.dedent(f"""\
      # Tfjs {_MODEL_HANDLE}
      Simple description spanning
      multiple lines.

      <!-- asset-path: {self.model_path} -->
      <!-- parent-model:   google/text-embedding-model/1   -->

      ## Overview""")
    self.set_content(
        os.path.join(self.tmp_docs_dir, _RELATIVE_TFJS_PATH), content)

    validator.validate_documentation_dir(
        validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_minimal_markdown_parsed_coral(self):
    lite_model = os.path.join(self.tmp_dir, "model.tflite")
    content = textwrap.dedent(f"""\
      # Coral {_MODEL_HANDLE}
      Simple description spanning
      multiple lines.

      <!-- asset-path: {lite_model} -->
      <!-- parent-model:   google/text-embedding-model/1   -->

      ## Overview""")
    self.set_content(
        os.path.join(self.tmp_docs_dir, _RELATIVE_CORAL_PATH), content)

    validator.validate_documentation_dir(
        validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_minimal_markdown_parsed_with_selected_files(self):
    self.set_content(self.markdown_file_path, self.minimal_markdown)

    validator.validate_documentation_files(
        validation_config=self.validation_config,
        root_dir=self.tmp_root_dir,
        files_to_validate=["google/models/text-embedding-model/1.md"])

  @parameterized.parameters(MINIMAL_COLLECTION, MAXIMAL_COLLECTION)
  def test_collection_markdown_parsed(self, markdown):
    google_path = "root/assets/docs/google"
    model_content = textwrap.dedent("""\
      # Module google/bert/1
      One line description.
      <!-- asset-path: https://domain.test/path/to/bert/1.tar.gz -->
      <!-- format: saved_model_2 -->
      <!-- task: text-embedding -->
      <!-- fine-tunable: false -->

      ## Overview""")
    self.set_content(f"{google_path}/models/bert/1.md", model_content)
    self.set_content(_ABSOLUTE_COLLECTION_PATH, markdown)

    validator.validate_documentation_dir(
        validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_collection_with_url_but_no_version_passes(self):
    google_path = "root/assets/docs/google"
    model_content = textwrap.dedent("""\
      # Module google/bert/3
      One line description.
      <!-- asset-path: https://domain.test/path/to/bert/3.tar.gz -->
      <!-- format: saved_model_2 -->
      <!-- task: text-embedding -->
      <!-- fine-tunable: false -->

      ## Overview""")
    self.set_content(f"{google_path}/models/bert/3.md", model_content)
    markdown = COLLECTION_CONTENT_TEMPLATE.format(
        content="https://tfhub.dev/google/bert")
    self.set_content(_ABSOLUTE_COLLECTION_PATH, markdown)

    validator.validate_documentation_dir(
        validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  @parameterized.parameters(
      ("No model url."), ("http://tfhub.dev/google/bert"),
      ("https://tfhub.dev/google"), ("https://tfhub.dev/google/"))
  def test_collection_fails_with_missing_model_url(self, invalid_url):
    markdown = COLLECTION_CONTENT_TEMPLATE.format(content=invalid_url)
    self.set_content(_ABSOLUTE_COLLECTION_PATH, markdown)

    with self.assertRaisesWithLiteralMatch(
        validator.MarkdownDocumentationError,
        "Found the following errors: {'google/collections/text-embedding-collec"
        "tion/1.md': 'A collection needs to contain at least one model URL.'}"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_collection_fails_if_model_does_not_exist(self):
    markdown = COLLECTION_CONTENT_TEMPLATE.format(
        content="https://tfhub.dev/google/non-existent/1")
    self.set_content(_ABSOLUTE_COLLECTION_PATH, markdown)

    with self.assertRaisesRegex(
        validator.MarkdownDocumentationError,
        ".*No documentation file found in "
        r"\['.*root/assets/docs/google/non-existent/1.md', "
        ".*root/assets/docs/google/models/non-existent/1.md.*"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_minimal_publisher_markdown_parsed(self):
    self.set_up_publisher_page("some-publisher")

    validator.validate_documentation_dir(
        validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_invalid_markdown_fails(self):
    self.set_content("root/assets/docs/google/model/1.md",
                     "INVALID MARKDOWN")

    with self.assertRaisesRegex(validator.MarkdownDocumentationError,
                                ".*First line.*"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_fails_if_publisher_page_does_not_exist(self):
    model_content = textwrap.dedent("""\
      # Module deepmind/model/1
      One line description.
      <!-- asset-path: https://domain.test/path/to/model/1.tar.gz -->
      <!-- format: saved_model_2 -->
      <!-- task: text-embedding -->
      <!-- fine-tunable: false -->

      ## Overview""")
    self.set_content(
        os.path.join(self.tmp_docs_dir, "deepmind/model/1"), model_content)

    with self.assertRaisesRegex(validator.MarkdownDocumentationError,
                                ".*Publisher documentation does not.*"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  @parameterized.parameters(
      ("Module", "google/model/1",
       r"\[.*google/model/1.md', '.*google/models/model/1.md'\]"),
      ("Placeholder", "google/model/1",
       r"\[.*google/model/1.md', '.*google/models/model/1.md'\]"),
      ("Tfjs", "google/model/1",
       r"\[.*google/model/tfjs/1.md', '.*google/models/model/tfjs/1.md'\]"),
      ("Lite", "google/model/1",
       r"\[.*google/model/lite/1.md', '.*google/models/model/lite/1.md'\]"),
      ("Coral", "google/model/1",
       r"\[.*google/model/coral/1.md', '.*google/models/model/coral/1.md'\]"),
      ("Collection", "google/model", r"\['.*google/collections/model/1.md'\]"))
  def test_fails_if_documentation_is_stored_in_wrong_location(
      self, model_type, handle, allowed_path_regexs):
    self.set_content("root/assets/docs/google/models/wrong-location/1.md",
                     f"# {model_type} google/model/1")

    with self.assertRaisesRegex(
        validator.MarkdownDocumentationError,
        fr".*Expected {handle} to have documentation stored in one of "
        f"{allowed_path_regexs} but was .*google/models/wrong-location/1.md."):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_fails_if_publisher_is_stored_in_wrong_location(self):
    self.set_content("root/assets/docs/google/1.md", "# Publisher google")

    with self.assertRaisesRegex(
        validator.MarkdownDocumentationError,
        ".*Expected google to have documentation stored in one of "
        r"\[.*google/google.md'\] but was .*google/1.md."):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  @parameterized.parameters("Google", "google/tf2", "google:tf2")
  def test_bad_publisher_id_fails(self, bad_id):
    self.set_content("root/assets/docs/google/google.md",
                     PUBLISHER_HANDLE_TEMPLATE % bad_id)

    with self.assertRaisesRegex(
        validator.MarkdownDocumentationError,
        f".*Instead the first line is '# Publisher {bad_id}'"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_publisher_markdown_at_incorrect_location_fails(self):
    self.set_content("root/assets/docs/google/publisher.md",
                     PUBLISHER_HANDLE_TEMPLATE % "some-publisher")

    with self.assertRaisesRegex(validator.MarkdownDocumentationError,
                                r".*some-publisher\.md.*"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_publisher_markdown_at_correct_location(self):
    self.set_up_publisher_page("some-publisher")

    validator.validate_documentation_dir(
        validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  @parameterized.parameters("google/model//1", "google/text-embedding&nbsp;/1",
                            "google/1", "google/model")
  def test_markdown_with_bad_handle(self, handle):
    content = textwrap.dedent("""\
      # Module %s
      Simple description.""" % handle)
    self.set_content(self.markdown_file_path, content)

    with self.assertRaisesRegex(validator.MarkdownDocumentationError,
                                ".*First line of the documentation*"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  @parameterized.parameters(SAVED_MODEL_WITHOUT_DESCRIPTION,
                            SAVED_MODEL_WITHOUT_DESCRIPTION_WITHOUT_LINEBREAK)
  def test_markdown_without_description(self, markdown):
    self.set_content(self.markdown_file_path, markdown)

    with self.assertRaisesRegex(validator.MarkdownDocumentationError,
                                ".*has to contain a short description.*"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_markdown_with_missing_metadata(self):
    content = textwrap.dedent(f"""\
      # Module {_MODEL_HANDLE}
      One line description.
      <!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
      <!-- format: saved_model_2 -->

      ## Overview""")
    self.set_content(self.markdown_file_path, content)

    with self.assertRaisesRegex(validator.MarkdownDocumentationError,
                                ".*missing.*fine-tunable.*task.*"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_markdown_with_unsupported_format_metadata(self):
    content = textwrap.dedent(f"""\
      # Module {_MODEL_HANDLE}
      Simple description.

      <!-- asset-path: /path/to/model.tar.gz -->
      <!-- task: text-embedding -->
      <!-- fine-tunable: true -->
      <!-- format: unsupported -->
      <!-- license: apache-2.0 -->

      ## Overview""")
    self.set_content(self.markdown_file_path, content)

    with self.assertRaisesRegex(
        validator.MarkdownDocumentationError, "The 'format' metadata.*but "
        "was 'unsupported'."):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_markdown_with_forbidden_duplicate_metadata(self):
    content = textwrap.dedent(f"""\
      # Module {_MODEL_HANDLE}
      One line description.
      <!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
      <!-- asset-path: https://path/to/text-embedding-model/model2.tar.gz -->
      <!-- task: text-embedding -->
      <!-- fine-tunable: true -->
      <!-- format: saved_model_2 -->

      ## Overview""")
    self.set_content(self.markdown_file_path, content)

    with self.assertRaisesRegex(validator.MarkdownDocumentationError,
                                ".*duplicate.*asset-path.*"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  @parameterized.parameters(
      ("dataset", ["mnist", "wikipedia"]), ("language", ["en", "fr"]),
      ("network-architecture", ["bert", "transformer"]))
  def test_markdown_with_allowed_duplicate_metadata(self, tag_key, tag_values):
    content = SAVED_MODEL_OPTIONAL_TAGS_TEMPLATE.format(
        tag_key_1=tag_key,
        tag_key_2=tag_key,
        tag_value_1=tag_values[0],
        tag_value_2=tag_values[1])
    self.set_content(self.markdown_file_path, content)

    validator.validate_documentation_dir(
        validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_markdown_with_unexpected_lines(self):
    content = textwrap.dedent(f"""\
      # Module {_MODEL_HANDLE}
      One line description.

      This should not be here.
      <!-- format: saved_model_2 -->

      ## Overview""")
    self.set_content(self.markdown_file_path, content)

    with self.assertRaisesRegex(validator.MarkdownDocumentationError,
                                ".*Unexpected line.*"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_minimal_markdown_parsed_full(self):
    self.set_content(self.markdown_file_path, self.minimal_markdown)
    documentation_parser = validator.DocumentationParser(
        self.tmp_root_dir, self.tmp_docs_dir, self.parser_by_tag)

    documentation_parser.validate(
        validation_config=self.validation_config,
        file_path=self.get_full_path(self.markdown_file_path))

    self.assertEqual("Simple description spanning multiple lines.",
                     documentation_parser.parsed_description)
    expected_metadata = {
        "asset-path": {self.model_path},
        "task": {"text-embedding"},
        "fine-tunable": {"true"},
        "format": {"saved_model_2"},
    }
    self.assertAllEqual(expected_metadata, documentation_parser.parsed_metadata)

  def test_asset_path_is_github_download_url_test(self):
    self.set_content(
        self.markdown_file_path, MINIMAL_SAVED_MODEL_TEMPLATE %
        "https://github.com/some_repo/releases/download/some_path.tar.gz")

    with self.assertRaisesRegex(validator.MarkdownDocumentationError,
                                ".*cannot be automatically fetched.*"):
      validator.validate_documentation_files(
          validation_config=self.validation_config,
          root_dir=self.tmp_root_dir,
          files_to_validate=["google/models/text-embedding-model/1.md"])

  def test_asset_path_is_legacy_and_modified(self):
    self.set_content(self.markdown_file_path,
                     MINIMAL_SAVED_MODEL_TEMPLATE % LEGACY_VALUE)

    with self.assertRaisesRegex(validator.MarkdownDocumentationError,
                                ".*end with .tar.gz but was legacy."):
      validator.validate_documentation_files(
          validation_config=self.validation_config,
          root_dir=self.tmp_root_dir,
          files_to_validate=["google/models/text-embedding-model/1.md"])

  def test_asset_path_is_legacy_and_unmodified(self):
    self.asset_path_modified = mock.patch.object(
        validator, "_is_asset_path_modified", return_value=False)
    self.asset_path_modified.start()
    self.set_content(self.markdown_file_path,
                     MINIMAL_SAVED_MODEL_TEMPLATE % LEGACY_VALUE)

    validator.validate_documentation_files(
        validation_config=self.validation_config,
        root_dir=self.tmp_root_dir,
        files_to_validate=["google/models/text-embedding-model/1.md"])

  @mock.patch.object(urllib.request, "urlopen", new=MockUrlOpen)
  def test_missing_saved_model_file_does_not_pass_smoke_test(self):
    not_a_model_path = os.path.join(self.tmp_dir, "not_a_model.tar.gz")
    temp_file = self.create_tempfile("keras_metadata.pb", "No SavedModel file.")
    filesystem_utils.create_archive(not_a_model_path, temp_file.full_path)
    self.minimal_markdown_with_bad_model = (
        MINIMAL_SAVED_MODEL_TEMPLATE % not_a_model_path)
    self.set_content(self.markdown_file_path,
                     self.minimal_markdown_with_bad_model)

    with self.assertRaisesRegex(validator.MarkdownDocumentationError,
                                ".*not contain a valid saved_model.pb.*"):
      validator.validate_documentation_files(
          validation_config=validator.ValidationConfig(do_smoke_test=True),
          root_dir=self.tmp_root_dir,
          files_to_validate=["google/models/text-embedding-model/1.md"])

  @mock.patch.object(urllib.request, "urlopen", new=MockUrlOpen)
  def test_invalid_asset_archive(self):
    not_an_archive_path = os.path.join(self.tmp_dir, "no_archive.tar.gz")
    temp_file = self.create_tempfile(not_an_archive_path, "No tar.gz archive.")
    self.minimal_markdown_with_bad_model = (
        MINIMAL_SAVED_MODEL_TEMPLATE % temp_file.full_path)
    self.set_content(self.markdown_file_path,
                     self.minimal_markdown_with_bad_model)

    with self.assertRaisesRegex(validator.MarkdownDocumentationError,
                                ".*Could not read tarfile: not a gzip file"):
      validator.validate_documentation_files(
          validation_config=validator.ValidationConfig(do_smoke_test=True),
          root_dir=self.tmp_root_dir,
          files_to_validate=["google/models/text-embedding-model/1.md"])

  @parameterized.parameters("saved_model.pb", "saved_model.pbtxt")
  @mock.patch.object(urllib.request, "urlopen", new=MockUrlOpen)
  def test_invalid_saved_model_file_does_not_pass_smoke_test(
      self, saved_model_name):
    not_a_model_path = os.path.join(self.tmp_dir, "not_a_model.tar.gz")
    temp_file = self.create_tempfile(saved_model_name, "No SavedModel file.")
    filesystem_utils.create_archive(not_a_model_path, temp_file.full_path)
    self.minimal_markdown_with_bad_model = (
        MINIMAL_SAVED_MODEL_TEMPLATE % not_a_model_path)
    self.set_content(self.markdown_file_path,
                     self.minimal_markdown_with_bad_model)

    with self.assertRaisesRegex(validator.MarkdownDocumentationError,
                                f".*Could not parse {saved_model_name}.*"):
      validator.validate_documentation_files(
          validation_config=validator.ValidationConfig(do_smoke_test=True),
          root_dir=self.tmp_root_dir,
          files_to_validate=["google/models/text-embedding-model/1.md"])

  @parameterized.parameters(
      ("Open Colab notebook", "https://colab.research.google.com"),
      ("Open Demo", "https://teachablemachine.withgoogle.com/train/pose"))
  def test_fail_on_deprecated_markdown_buttons(self, button_text, button_value):
    content = textwrap.dedent(f"""\
      # Module {_MODEL_HANDLE}
      One line description.
      <!-- asset-path: https://path/to/model.tar.gz -->
      <!-- task: text-embedding -->
      <!-- fine-tunable: true -->
      <!-- format: saved_model_2 -->

      [![{button_text}]]({button_value})

      ## Overview""")
    self.set_content(self.markdown_file_path, content)

    with self.assertRaisesRegex(
        validator.MarkdownDocumentationError,
        rf".*Unexpected line found: '\[!\[{button_text}"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_markdown_with_bad_module_type(self):
    content = textwrap.dedent(f"""\
      # Module {_MODEL_HANDLE}
      Simple description spanning
      multiple lines.

      <!-- asset-path: /path/to/model.tar.gz -->
      <!-- task: something-embedding -->
      <!-- fine-tunable:true -->
      <!-- format: saved_model_2 -->

      # Overview""")
    self.set_content(self.markdown_file_path, content)

    with self.assertRaisesRegex(
        validator.MarkdownDocumentationError,
        "The 'task' metadata has to start with any of 'image-', 'text', "
        "'audio-', 'video-', but is: 'something-embedding'"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_markdown_with_forbidden_format_metadata(self):
    self.set_content(
        os.path.join(self.tmp_docs_dir, _RELATIVE_LITE_PATH),
        LITE_OPTIONAL_TAG_TEMPLATE.format(
            tag_key="format", tag_value="saved_model"))

    with self.assertRaisesRegex(
        validator.MarkdownDocumentationError,
        r".*contains unsupported metadata properties: \['format'\].*"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  @parameterized.parameters(
      (PLACEHOLDER_OPTIONAL_TAG_TEMPLATE, _RELATIVE_SAVED_MODEL_PATH),
      (SAVED_MODEL_OPTIONAL_TAG_TEMPLATE, _RELATIVE_SAVED_MODEL_PATH),
      (LITE_OPTIONAL_TAG_TEMPLATE, _RELATIVE_LITE_PATH),
      (CORAL_OPTIONAL_TAG_TEMPLATE, _RELATIVE_CORAL_PATH),
      (TFJS_OPTIONAL_TAG_TEMPLATE, _RELATIVE_TFJS_PATH))
  def test_markdown_with_unsupported_metadata(self, markdown_template,
                                              rel_file_path):
    content = markdown_template.format(
        tag_key="unsupported_tag", tag_value="value")
    self.set_content(os.path.join(self.tmp_docs_dir, rel_file_path), content)

    with self.assertRaisesRegex(
        validator.MarkdownDocumentationError,
        r".*contains unsupported metadata properties: \['unsupported_tag'\].*"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  @parameterized.parameters(
      (SAVED_MODEL_OPTIONAL_TAG_TEMPLATE, _RELATIVE_SAVED_MODEL_PATH),
      (TFJS_OPTIONAL_TAG_TEMPLATE, _RELATIVE_TFJS_PATH),
      (LITE_OPTIONAL_TAG_TEMPLATE, _RELATIVE_LITE_PATH),
      (CORAL_OPTIONAL_TAG_TEMPLATE, _RELATIVE_CORAL_PATH))
  def test_markdown_with_valid_colab_url(self, template, relative_path):
    content = template.format(
        tag_key="colab",
        tag_value="https://colab.research.google.com/mycolab.ipynb")
    self.set_content(os.path.join(self.tmp_docs_dir, relative_path), content)

    validator.validate_documentation_dir(
        validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  @parameterized.parameters(
      (SAVED_MODEL_OPTIONAL_TAG_TEMPLATE, _RELATIVE_SAVED_MODEL_PATH),
      (TFJS_OPTIONAL_TAG_TEMPLATE, _RELATIVE_TFJS_PATH),
      (LITE_OPTIONAL_TAG_TEMPLATE, _RELATIVE_LITE_PATH),
      (CORAL_OPTIONAL_TAG_TEMPLATE, _RELATIVE_CORAL_PATH))
  def test_markdown_with_bad_colab_url_fails(self, template, relative_path):
    content = template.format(
        tag_key="colab",
        tag_value="https://github.com/mycolab.ipynb")
    self.set_content(os.path.join(self.tmp_docs_dir, relative_path), content)

    with self.assertRaisesRegex(
        validator.MarkdownDocumentationError,
        "URL must lead to domain colab.research.google.com but is github.com"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_demo_tag_on_tfjs_model(self):
    content = TFJS_OPTIONAL_TAG_TEMPLATE.format(
        tag_key="demo", tag_value="https://mydemo.com")
    self.set_content(
        os.path.join(self.tmp_docs_dir, _RELATIVE_TFJS_PATH), content)

    validator.validate_documentation_dir(
        validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_demo_tag_on_tfjs_model_with_unsecure_url_fails(self):
    content = TFJS_OPTIONAL_TAG_TEMPLATE.format(
        tag_key="demo", tag_value="http://the-unsecure-page.com")
    self.set_content(
        os.path.join(self.tmp_docs_dir, _RELATIVE_TFJS_PATH), content)

    with self.assertRaisesRegex(
        validator.MarkdownDocumentationError,
        "http://the-unsecure-page.com is not an HTTPS URL."):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  @parameterized.parameters(
      ("dataset", "dataset"),

      ("interactive-visualizer", "interactive_visualizer"),
      ("language", "language"),
      ("license", "license"),
      ("network-architecture", "network_architecture"),
  )
  def test_saved_model_markdown_with_unsupported_tag_value(
      self, tag_key, yaml_file_name):
    content = SAVED_MODEL_OPTIONAL_TAG_TEMPLATE.format(
        tag_key=tag_key, tag_value="n/a")
    self.set_content(self.markdown_file_path, content)

    with self.assertRaisesRegex(
        validator.MarkdownDocumentationError,
        f"Validating {tag_key} failed: "
        f"Unsupported values for {tag_key} tag were found: "
        rf"\['n/a'\]. Please add them to tags/{yaml_file_name}.yaml."):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  @parameterized.parameters(
      ("dataset", "n/a", "dataset"), ("language", "n/a", "language"),
      ("network-architecture", "n/a", "network_architecture"))
  def test_collection_markdown_with_unsupported_tag_value(
      self, tag_key, tag_value, yaml_file_name):
    content = COLLECTION_OPTIONAL_TAG_TEMPLATE.format(
        tag_key=tag_key, tag_value=tag_value)
    self.set_content(_ABSOLUTE_COLLECTION_PATH, content)

    with self.assertRaisesRegex(
        validator.MarkdownDocumentationError,
        f"Unsupported values for {tag_key} tag were found: "
        rf"\['n/a'\]. Please add them to tags/{yaml_file_name}.yaml."):
      validator.validate_documentation_dir(
          validation_config=self.validation_config,
          root_dir=self.tmp_root_dir)

  def test_collection_without_contained_model_url_fails(self):
    content = COLLECTION_OPTIONAL_TAG_TEMPLATE.format(
        tag_key="language", tag_value="en")
    self.set_content(_ABSOLUTE_COLLECTION_PATH, content)

    with self.assertRaisesRegex(
        validator.MarkdownDocumentationError,
        ".*'google/collections/text-embedding-collection/1.md': "
        "'A collection needs to contain at least one model URL.'}"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  @parameterized.parameters(
      ("dataset", "dataset"),

      ("interactive-visualizer", "interactive_visualizer"),
      ("language", "language"), ("license", "license"),
      ("network-architecture", "network_architecture"))
  def test_placeholder_markdown_with_unsupported_tag_value(
      self, tag_key, yaml_file_name):
    content = PLACEHOLDER_OPTIONAL_TAG_TEMPLATE.format(
        tag_key=tag_key, tag_value="n/a")
    self.set_content(self.markdown_file_path, content)

    with self.assertRaisesRegex(
        validator.MarkdownDocumentationError,
        f"Unsupported values for {tag_key} tag were found: "
        rf"\['n/a'\]. Please add them to tags/{yaml_file_name}.yaml."):
      validator.validate_documentation_dir(
          validation_config=self.validation_config,
          root_dir=self.tmp_root_dir)

  @mock.patch.object(urllib.request, "urlopen", new=MockUrlOpen)
  def test_model_with_invalid_filenames_fails_smoke_test(self):
    invalid_file_name = ".invalid_file"
    documentation_parser = self._get_parser_for_validating_saved_model_file(
        invalid_file_name, self.markdown_file_path)

    with self.assertRaisesRegex(validator.MarkdownDocumentationError,
                                rf"Invalid filepath.*{invalid_file_name}"):
      documentation_parser.validate(
          validation_config=validator.ValidationConfig(do_smoke_test=True),
          file_path=self.get_full_path(self.markdown_file_path))

  @parameterized.parameters("assets/asset1", "assets.extra/asset1",
                            "assets/somedir/asset1",
                            "variables/variables.index")
  @mock.patch.object(urllib.request, "urlopen", new=MockUrlOpen)
  def test_saved_model_with_custom_asset_file_succeeds(self, allowed_file):
    """Tests that validating an archive with usable files succeeds."""
    documentation_parser = self._get_parser_for_validating_saved_model_file(
        allowed_file, self.markdown_file_path)

    try:
      documentation_parser.validate(
          validation_config=validator.ValidationConfig(do_smoke_test=True),
          file_path=self.get_full_path(self.markdown_file_path))
    except validator.MarkdownDocumentationError as e:
      self.fail("We should allow custom files that can be used by the "
                "SavedModel but the test raised a MarkdownDocumentationError: "
                f"{e}.")

  @parameterized.parameters("other_file.txt", "variables/vocab.txt")
  @mock.patch.object(urllib.request, "urlopen", new=MockUrlOpen)
  def test_saved_model_with_unused_file_fails(self, forbidden_file):
    """Tests that validating an archive with unusuable files fails."""
    documentation_parser = self._get_parser_for_validating_saved_model_file(
        forbidden_file, self.markdown_file_path)

    with self.assertRaisesRegex(
        validator.MarkdownDocumentationError,
        rf"File cannot be used by SavedModel: .*{forbidden_file}"):
      documentation_parser.validate(
          validation_config=validator.ValidationConfig(do_smoke_test=True),
          file_path=self.get_full_path(self.markdown_file_path))


if __name__ == "__main__":
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
