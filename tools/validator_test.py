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

import os
import textwrap
from unittest import mock

from absl.testing import parameterized
import tensorflow as tf
import validator
import yaml_parser

DATASET_YAML = """
values:
  - id: mnist
    display_name: MNIST
  - id: wikipedia
    display_name: Wikipedia"""

ARCHITECTURE_YAML = """
values:
  - id: bert
    display_name: BERT
  - id: transformer
    display_name: Transformer"""

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
    "network_architecture.yaml": ARCHITECTURE_YAML,
    "dataset.yaml": DATASET_YAML,
    "language.yaml": LANGUAGE_YAML,
    "license.yaml": LICENSE_YAML,
    "task.yaml": TASK_YAML,
    "interactive_visualizer.yaml": VISUALIZER_YAML
}

LEGACY_VALUE = "legacy"

MINIMAL_SAVED_MODEL_TEMPLATE = """# Module google/text-embedding-model/1
Simple description spanning
multiple lines.

<!-- asset-path: %s -->
<!-- task:   text-embedding   -->
<!-- fine-tunable:true -->
<!-- format: saved_model_2 -->

## Overview
"""

SAVED_MODEL_OPTIONAL_TAG_TEMPLATE = """# Module google/model/1
Simple description spanning
multiple lines.

<!-- asset-path: /path/to/model.tar.gz -->
<!-- task:   text-embedding   -->
<!-- fine-tunable:true -->
<!-- format: saved_model_2 -->
<!-- {tag_key}: {tag_value} -->
"""

SAVED_MODEL_WITHOUT_DESCRIPTION = """# Module google/text-embedding-model/1

<!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
<!-- format: saved_model_2 -->

## Overview
"""

SAVED_MODEL_WITHOUT_DESCRIPTION_WITHOUT_LINEBREAK = """# Module google/text-embedding-model/1
<!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
<!-- format: saved_model_2 -->

## Overview
"""

SAVED_MODEL_OPTIONAL_TAGS_TEMPLATE = """# Module google/model/1
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

MAXIMAL_SAVED_MODEL_TEMPLATE = """# Module google/text-embedding-model/1
Simple description spanning
multiple lines.

<!-- asset-path: %s -->
<!-- task:   text-embedding   -->
<!-- fine-tunable:true -->
<!-- format: saved_model_2 -->
<!-- dataset: mnist -->
<!-- interactive-model-name: vision -->
<!-- language: en -->
<!-- network-architecture: bert -->
<!-- license: apache-2.0 -->
<!-- colab: https://colab.research.google.com/mycolab.ipynb -->

## Overview
"""

MINIMAL_PLACEHOLDER = """# Placeholder google/text-embedding-model/1
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

MAXIMAL_PLACEHOLDER = """# Placeholder google/text-embedding-model/1
Simple description spanning
multiple lines.

<!-- dataset: mnist -->
<!-- fine-tunable:true -->
<!-- interactive-model-name: vision -->
<!-- language: en -->
<!-- task:   text-embedding   -->
<!-- network-architecture: bert -->
<!-- license: apache-2.0 -->
"""

LITE_OPTIONAL_TAG_TEMPLATE = """# Lite google/model/lite/1
Simple description spanning
multiple lines.

<!-- asset-path: /path/to/model.tflite -->
<!-- parent-model: google/text-embedding-model/1 -->
<!-- {tag_key}: {tag_value} -->

## Overview
"""

TFJS_OPTIONAL_TAG_TEMPLATE = """# Tfjs google/text-embedding-model/tfjs/1
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
"""

COLLECTION_OPTIONAL_TAG_TEMPLATE = """# Collection google/model/1
Simple description spanning
multiple lines.

<!-- task: text-embedding -->
<!-- {tag_key}: {tag_value} -->

## Overview
"""

MAXIMAL_COLLECTION = """# Collection google/text-embedding-collection/1
Simple description spanning
multiple lines.

<!-- task: text-embedding -->
<!-- dataset: mnist -->
<!-- language: en -->
<!-- network-architecture: bert -->

## Overview
"""

PUBLISHER_HANDLE_TEMPLATE = """# Publisher %s
Simple description spanning one line.

[![Icon URL]](https://path/to/icon.png)

## Overview
"""


class ValidatorTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(tf.test.TestCase, self).setUp()
    self.validation_config = validator.ValidationConfig()
    self.tmp_dir = self.create_tempdir()
    self.tmp_root_dir = os.path.join(self.tmp_dir, "root")
    self.tmp_docs_dir = os.path.join(self.tmp_root_dir, "assets", "docs")
    self.model_path = os.path.join(self.tmp_dir, "model_1.tar.gz")
    self.not_a_model_path = os.path.join(self.tmp_dir, "not_a_model.tar.gz")
    self.save_dummy_model(self.model_path)
    self.minimal_markdown = MINIMAL_SAVED_MODEL_TEMPLATE % self.model_path
    self.maximal_markdown = MAXIMAL_SAVED_MODEL_TEMPLATE % self.model_path
    self.minimal_markdown_with_bad_model = (
        MINIMAL_SAVED_MODEL_TEMPLATE % self.not_a_model_path)
    for file_name, content in TAG_FILE_NAME_TO_CONTENT_MAP.items():
      self.set_content(f"root/tags/{file_name}", content)
    self.asset_path_modified = mock.patch.object(
        validator, "_is_asset_path_modified", return_value=True)
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

  def save_dummy_model(self, path):

    class MultiplyTimesTwoModel(tf.train.Checkpoint):
      """Callable model that multiplies by two."""

      @tf.function(
          input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
      def __call__(self, x):
        return x * 2

    model = MultiplyTimesTwoModel()
    tf.saved_model.save(model, path)

  def test_markdown_parsed_saved_model(self):
    empty_second_line = textwrap.dedent(f"""\
       # Module google/text-embedding-model/1

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
      self.set_up_publisher_page("google")
      self.set_content(
          "root/assets/docs/google/models/text-embedding-model/1.md", markdown)

      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  @parameterized.parameters(MINIMAL_PLACEHOLDER, MAXIMAL_PLACEHOLDER)
  def test_markdown_parsed_placeholder(self, markdown):
    self.set_up_publisher_page("google")
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     markdown)

    validator.validate_documentation_dir(
        validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_minimal_markdown_parsed_lite(self):
    lite_model = os.path.join(self.tmp_dir, "model.tflite")
    content = textwrap.dedent(f"""\
      # Lite google/text-embedding-model/lite/1
      Simple description spanning
      multiple lines.

      <!-- asset-path: {lite_model} -->
      <!-- parent-model: google/text-embedding-model/1 -->

      ## Overview""")
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     content)
    self.set_up_publisher_page("google")

    validator.validate_documentation_dir(
        validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_minimal_markdown_parsed_tfjs(self):
    content = textwrap.dedent(f"""\
      # Tfjs google/text-embedding-model/tfjs/1
      Simple description spanning
      multiple lines.

      <!-- asset-path: {self.model_path} -->
      <!-- parent-model:   google/text-embedding-model/1   -->

      ## Overview""")
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     content)
    self.set_up_publisher_page("google")

    validator.validate_documentation_dir(
        validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_minimal_markdown_parsed_coral(self):
    lite_model = os.path.join(self.tmp_dir, "model.tflite")
    content = textwrap.dedent(f"""\
      # Coral google/text-embedding-model/coral/1
      Simple description spanning
      multiple lines.

      <!-- asset-path: {lite_model} -->
      <!-- parent-model:   google/text-embedding-model/1   -->

      ## Overview""")
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     content)
    self.set_up_publisher_page("google")

    validator.validate_documentation_dir(
        validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_minimal_markdown_parsed_with_selected_files(self):
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     self.minimal_markdown)
    self.set_up_publisher_page("google")

    validator.validate_documentation_files(
        validation_config=self.validation_config,
        root_dir=self.tmp_root_dir,
        files_to_validate=["google/models/text-embedding-model/1.md"])

  @parameterized.parameters(MINIMAL_COLLECTION, MAXIMAL_COLLECTION)
  def test_collection_markdown_parsed(self, markdown):
    self.set_up_publisher_page("google")
    self.set_content(
        "root/assets/docs/google/collections/text-embedding-collection/1.md",
        markdown)

    validator.validate_documentation_dir(
        validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_minimal_publisher_markdown_parsed(self):
    self.set_up_publisher_page("some-publisher")

    validator.validate_documentation_dir(
        validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_invalid_markdown_fails(self):
    self.set_content("root/assets/docs/publisher/model/1.md",
                     "INVALID MARKDOWN")

    with self.assertRaisesRegex(validator.MarkdownDocumentationError,
                                ".*First line.*"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_minimal_markdown_not_in_publisher_dir(self):
    self.set_content("root/assets/docs/gooogle/models/wrong-location/1.md",
                     self.minimal_markdown)

    with self.assertRaisesRegex(validator.MarkdownDocumentationError,
                                ".*placed in the publisher directory.*"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_fails_if_publisher_page_does_not_exist(self):
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     self.minimal_markdown)

    with self.assertRaisesRegex(validator.MarkdownDocumentationError,
                                ".*Publisher documentation does not.*"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_minimal_markdown_does_not_end_with_md_fails(self):
    self.set_content("root/assets/docs/google/models/wrong-extension/1.mdz",
                     self.minimal_markdown)

    with self.assertRaisesRegex(validator.MarkdownDocumentationError,
                                r".*end with '\.md.'*"):
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
    self.set_content("root/assets/docs/google/models/model/1.md", content)

    with self.assertRaisesRegex(validator.MarkdownDocumentationError,
                                ".*First line of the documentation*"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  @parameterized.parameters(SAVED_MODEL_WITHOUT_DESCRIPTION,
                            SAVED_MODEL_WITHOUT_DESCRIPTION_WITHOUT_LINEBREAK)
  def test_markdown_without_description(self, markdown):
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     markdown)

    with self.assertRaisesRegex(validator.MarkdownDocumentationError,
                                ".*has to contain a short description.*"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_markdown_with_missing_metadata(self):
    content = textwrap.dedent("""\
      # Module google/text-embedding-model/1
      One line description.
      <!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
      <!-- format: saved_model_2 -->

      ## Overview""")
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     content)

    with self.assertRaisesRegex(validator.MarkdownDocumentationError,
                                ".*missing.*fine-tunable.*task.*"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_markdown_with_unsupported_format_metadata(self):
    content = textwrap.dedent("""\
      # Module google/model/1
      Simple description.

      <!-- asset-path: /path/to/model.tar.gz -->
      <!-- task: text-embedding -->
      <!-- fine-tunable: true -->
      <!-- format: unsupported -->
      <!-- license: apache-2.0 -->

      ## Overview""")
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     content)

    with self.assertRaisesRegex(
        validator.MarkdownDocumentationError, "The 'format' metadata.*but "
        "was 'unsupported'."):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_markdown_with_forbidden_duplicate_metadata(self):
    content = textwrap.dedent("""\
      # Module google/model/1
      One line description.
      <!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
      <!-- asset-path: https://path/to/text-embedding-model/model2.tar.gz -->
      <!-- task: text-embedding -->
      <!-- fine-tunable: true -->
      <!-- format: saved_model_2 -->

      ## Overview""")
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     content)

    with self.assertRaisesRegex(validator.MarkdownDocumentationError,
                                ".*duplicate.*asset-path.*"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  @parameterized.parameters(
      ("dataset", ["mnist", "wikipedia"]), ("language", ["en", "fr"]),
      ("network-architecture", ["bert", "transformer"]))
  def test_markdown_with_allowed_duplicate_metadata(self, tag_key, tag_values):
    self.set_up_publisher_page("google")
    content = SAVED_MODEL_OPTIONAL_TAGS_TEMPLATE.format(
        tag_key_1=tag_key,
        tag_key_2=tag_key,
        tag_value_1=tag_values[0],
        tag_value_2=tag_values[1])
    self.set_content("root/assets/docs/google/models/model/1.md", content)

    validator.validate_documentation_dir(
        validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_markdown_with_unexpected_lines(self):
    content = textwrap.dedent("""\
      # Module google/text-embedding-model/1
      One line description.

      This should not be here.
      <!-- format: saved_model_2 -->

      ## Overview""")
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     content)

    with self.assertRaisesRegex(validator.MarkdownDocumentationError,
                                ".*Unexpected line.*"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_minimal_markdown_parsed_full(self):
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     self.minimal_markdown)
    self.set_up_publisher_page("google")
    documentation_parser = validator.DocumentationParser(
        self.tmp_root_dir, self.tmp_docs_dir)
    parser = yaml_parser.YamlParser(self.tmp_root_dir)

    documentation_parser.validate(
        validation_config=self.validation_config,
        yaml_parser=parser,
        file_path=self.get_full_path(
            "root/assets/docs/google/models/text-embedding-model/1.md"))

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
        "root/assets/docs/google/models/text-embedding-model/1.md",
        MINIMAL_SAVED_MODEL_TEMPLATE %
        "https://github.com/some_repo/releases/download/some_path.tar.gz")
    self.set_up_publisher_page("google")

    with self.assertRaisesRegex(validator.MarkdownDocumentationError,
                                ".*cannot be automatically fetched.*"):
      validator.validate_documentation_files(
          validation_config=self.validation_config,
          root_dir=self.tmp_root_dir,
          files_to_validate=["google/models/text-embedding-model/1.md"])

  def test_asset_path_is_legacy_and_modified(self):
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     MINIMAL_SAVED_MODEL_TEMPLATE % LEGACY_VALUE)
    self.set_up_publisher_page("google")

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
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     MINIMAL_SAVED_MODEL_TEMPLATE % LEGACY_VALUE)
    self.set_up_publisher_page("google")

    validator.validate_documentation_files(
        validation_config=self.validation_config,
        root_dir=self.tmp_root_dir,
        files_to_validate=["google/models/text-embedding-model/1.md"])

  def test_bad_model_does_not_pass_smoke_test(self):
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     self.minimal_markdown_with_bad_model)
    self.set_up_publisher_page("google")

    with self.assertRaisesRegex(validator.MarkdownDocumentationError,
                                ".*failed to parse.*"):
      validator.validate_documentation_files(
          validation_config=validator.ValidationConfig(do_smoke_test=True),
          root_dir=self.tmp_root_dir,
          files_to_validate=["google/models/text-embedding-model/1.md"])

  @parameterized.parameters(
      ("Open Colab notebook", "https://colab.research.google.com"),
      ("Open Demo", "https://teachablemachine.withgoogle.com/train/pose"))
  def test_markdown_buttons(self, button_text, button_value):
    content = textwrap.dedent(f"""\
      # Module google/text-embedding-model/1
      One line description.
      <!-- asset-path: https://path/to/model.tar.gz -->
      <!-- task: text-embedding -->
      <!-- fine-tunable: true -->
      <!-- format: saved_model_2 -->

      [![{button_text}]]({button_value})

      ## Overview""")
    self.set_content("root/assets/docs/google/models/model/1.md", content)
    self.set_up_publisher_page("google")

    validator.validate_documentation_dir(
        validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_markdown_with_bad_module_type(self):
    content = textwrap.dedent("""\
      # Module google/model/1
      Simple description spanning
      multiple lines.

      <!-- asset-path: /path/to/model.tar.gz -->
      <!-- task: something-embedding -->
      <!-- fine-tunable:true -->
      <!-- format: saved_model_2 -->

      # Overview""")
    self.set_content("root/assets/docs/google/models/model/1.md", content)
    self.set_up_publisher_page("google")

    with self.assertRaisesRegex(
        validator.MarkdownDocumentationError,
        "The 'task' metadata has to start with any of 'image-', 'text', "
        "'audio-', 'video-', but is: 'something-embedding'"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_markdown_with_forbidden_format_metadata(self):
    self.set_content(
        "root/assets/docs/google/models/model/1.md",
        LITE_OPTIONAL_TAG_TEMPLATE.format(
            tag_key="format", tag_value="saved_model"))
    self.set_up_publisher_page("google")

    with self.assertRaisesRegex(
        validator.MarkdownDocumentationError,
        r".*contains unsupported metadata properties: \['format'\].*"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  @parameterized.parameters(PLACEHOLDER_OPTIONAL_TAG_TEMPLATE,
                            SAVED_MODEL_OPTIONAL_TAG_TEMPLATE,
                            LITE_OPTIONAL_TAG_TEMPLATE,
                            TFJS_OPTIONAL_TAG_TEMPLATE)
  def test_markdown_with_unsupported_metadata(self, markdown_template):
    self.set_up_publisher_page("google")
    content = markdown_template.format(
        tag_key="unsupported_tag", tag_value="value")
    self.set_content("root/assets/docs/google/models/model/1.md", content)

    with self.assertRaisesRegex(
        validator.MarkdownDocumentationError,
        r".*contains unsupported metadata properties: \['unsupported_tag'\].*"):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  @parameterized.parameters(SAVED_MODEL_OPTIONAL_TAG_TEMPLATE,
                            TFJS_OPTIONAL_TAG_TEMPLATE,
                            LITE_OPTIONAL_TAG_TEMPLATE)
  def test_markdown_with_colab_tag(self, template):
    self.set_up_publisher_page("google")
    content = template.format(
        tag_key="colab",
        tag_value="https://colab.research.google.com/mycolab.ipynb")
    self.set_content("root/assets/docs/google/models/model/1.md", content)

    validator.validate_documentation_dir(
        validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  def test_demo_tag_on_tfjs_model(self):
    self.set_up_publisher_page("google")
    content = TFJS_OPTIONAL_TAG_TEMPLATE.format(
        tag_key="demo", tag_value="https://mydemo.com")
    self.set_content("root/assets/docs/google/models/model/1.md", content)

    validator.validate_documentation_dir(
        validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  @parameterized.parameters(
      ("dataset", "dataset"),
      ("interactive-model-name", "interactive_visualizer"),
      ("language", "language"),
      ("license", "license"),
      ("network-architecture", "network_architecture"),
  )
  def test_saved_model_markdown_with_unsupported_tag_value(
      self, tag_key, yaml_file_name):
    self.set_up_publisher_page("google")
    content = SAVED_MODEL_OPTIONAL_TAG_TEMPLATE.format(
        tag_key=tag_key, tag_value="n/a")
    self.set_content("root/assets/docs/google/models/model/1.md", content)

    with self.assertRaisesRegex(
        validator.MarkdownDocumentationError,
        f"Unsupported values for {tag_key} tag were found: "
        rf"\['n/a'\]. Please add them to tags/{yaml_file_name}.yaml."):
      validator.validate_documentation_dir(
          validation_config=self.validation_config, root_dir=self.tmp_root_dir)

  @parameterized.parameters(
      ("dataset", "n/a", "dataset"), ("language", "n/a", "language"),
      ("network-architecture", "n/a", "network_architecture"))
  def test_collection_markdown_with_unsupported_tag_value(
      self, tag_key, tag_value, yaml_file_name):
    self.set_up_publisher_page("google")
    content = COLLECTION_OPTIONAL_TAG_TEMPLATE.format(
        tag_key=tag_key, tag_value=tag_value)
    self.set_content("root/assets/docs/google/models/model/1.md", content)

    with self.assertRaisesRegex(
        validator.MarkdownDocumentationError,
        f"Unsupported values for {tag_key} tag were found: "
        rf"\['n/a'\]. Please add them to tags/{yaml_file_name}.yaml."):
      validator.validate_documentation_dir(
          validation_config=self.validation_config,
          root_dir=self.tmp_root_dir)

  @parameterized.parameters(
      ("dataset", "dataset"),
      ("interactive-model-name", "interactive_visualizer"),
      ("language", "language"), ("license", "license"),
      ("network-architecture", "network_architecture"))
  def test_placeholder_markdown_with_unsupported_tag_value(
      self, tag_key, yaml_file_name):
    self.set_up_publisher_page("google")
    content = PLACEHOLDER_OPTIONAL_TAG_TEMPLATE.format(
        tag_key=tag_key, tag_value="n/a")
    self.set_content("root/assets/docs/google/models/model/1.md", content)

    with self.assertRaisesRegex(
        validator.MarkdownDocumentationError,
        f"Unsupported values for {tag_key} tag were found: "
        rf"\['n/a'\]. Please add them to tags/{yaml_file_name}.yaml."):
      validator.validate_documentation_dir(
          validation_config=self.validation_config,
          root_dir=self.tmp_root_dir)

  def test_model_with_invalid_filenames_fails_smoke_test(self):
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     self.minimal_markdown)
    self.set_up_publisher_page("google")
    parser = yaml_parser.YamlParser(self.tmp_root_dir)
    with open(os.path.join(self.model_path, ".invalid_file"), "w") as bad_file:
      bad_file.write("This file shouldn't be here")
    documentation_parser = validator.DocumentationParser(
        self.tmp_root_dir, self.tmp_docs_dir)

    with self.assertRaisesRegex(validator.MarkdownDocumentationError,
                                r"Invalid filepath.*\.invalid_file"):
      documentation_parser.validate(
          validation_config=validator.ValidationConfig(do_smoke_test=True),
          yaml_parser=parser,
          file_path=self.get_full_path(
              "root/assets/docs/google/models/text-embedding-model/1.md"))


if __name__ == "__main__":
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
