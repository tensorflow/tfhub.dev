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
from unittest import mock

import tensorflow as tf
import validator


MINIMAL_MARKDOWN_SAVED_MODEL_TEMPLATE = """# Module google/text-embedding-model/1
Simple description spanning
multiple lines.

<!-- asset-path: %s -->
<!-- module-type:   text-embedding   -->
<!-- fine-tunable:true -->
<!-- format: saved_model_2 -->

## Overview
"""

MARKDOWN_WITH_EMPTY_SECOND_LINE = """# Module google/text-embedding-model/1

Simple description spanning
multiple lines.

<!-- asset-path: %s -->
<!-- module-type:   text-embedding   -->
<!-- fine-tunable:true -->
<!-- format: saved_model_2 -->

## Overview
"""

MARKDOWN_SAVED_MODEL_UNSUPPORTED_TAG = """# Module google/model/1
Simple description spanning
multiple lines.

<!-- asset-path: /path/to/model -->
<!-- module-type:   text-embedding   -->
<!-- fine-tunable:true -->
<!-- format: saved_model_2 -->
<!-- unsupported_tag: value -->
"""

MARKDOWN_SAVED_MODEL_UNSUPPORTED_LANGUAGE = """# Module google/model/1
Simple description spanning
multiple lines.

<!-- asset-path: /path/to/model -->
<!-- module-type:   text-embedding   -->
<!-- fine-tunable:true -->
<!-- format: saved_model_2 -->
<!-- language: non-existent -->
"""

MAXIMAL_MARKDOWN_SAVED_MODEL_TEMPLATE = """# Module google/text-embedding-model/1
Simple description spanning
multiple lines.

<!-- asset-path: %s -->
<!-- module-type:   text-embedding   -->
<!-- fine-tunable:true -->
<!-- format: saved_model_2 -->
<!-- dataset: MNIST -->
<!-- interactive-model-name: vision -->
<!-- language: en -->
<!-- network-architecture: BERT -->
<!-- license: Apache-2.0 -->

## Overview
"""

MINIMAL_MARKDOWN_PLACEHOLDER_TEMPLATE = """# Placeholder google/text-embedding-model/1
Simple description spanning
multiple lines.

<!-- module-type:   text-embedding   -->
"""

MARKDOWN_PLACEHOLDER_UNSUPPORTED_TAG = """# Placeholder google/text-embedding-model/1
Simple description spanning
multiple lines.

<!-- module-type:   text-embedding   -->
<!-- unsupported_tag: value -->
"""

MARKDOWN_PLACEHOLDER_UNSUPPORTED_LANGUAGE = """# Placeholder google/a/1
Simple description spanning
multiple lines.

<!-- module-type: text-embedding   -->
<!-- language: non-existent -->
"""

MAXIMAL_MARKDOWN_PLACEHOLDER_TEMPLATE = """# Placeholder google/text-embedding-model/1
Simple description spanning
multiple lines.

<!-- dataset: MNIST -->
<!-- fine-tunable:true -->
<!-- interactive-model-name: vision -->
<!-- language: en -->
<!-- module-type:   text-embedding   -->
<!-- network-architecture: BERT -->
<!-- license: Apache-2.0 -->
"""

MINIMAL_MARKDOWN_LITE_TEMPLATE = """# Lite google/text-embedding-model/lite/1
Simple description spanning
multiple lines.

<!-- asset-path: %s -->
<!-- parent-model: google/text-embedding-model/1 -->

## Overview
"""

MINIMAL_MARKDOWN_LITE_WITH_FORBIDDEN_FORMAT = """# Lite google/model/lite/1
Simple description spanning
multiple lines.

<!-- asset-path: /path/to/model -->
<!-- parent-model: google/text-embedding-model/1 -->
<!-- format: saved_model -->

## Overview
"""

MINIMAL_MARKDOWN_LITE_WITH_UNSUPPORTED_TAG = """# Lite google/model/lite/1
Simple description spanning
multiple lines.

<!-- asset-path: /path/to/model -->
<!-- parent-model: google/text-embedding-model/1 -->
<!-- unsupported_tag: value -->

## Overview
"""

MINIMAL_MARKDOWN_TFJS_TEMPLATE = """# Tfjs google/text-embedding-model/tfjs/1
Simple description spanning
multiple lines.

<!-- asset-path: %s -->
<!-- parent-model:   google/text-embedding-model/1   -->

## Overview
"""

MINIMAL_MARKDOWN_CORAL_TEMPLATE = """# Coral google/text-embedding-model/coral/1
Simple description spanning
multiple lines.

<!-- asset-path: %s -->
<!-- parent-model:   google/text-embedding-model/1   -->

## Overview
"""

MINIMAL_MARKDOWN_WITH_UNKNOWN_PUBLISHER = """# Module publisher-without-page/text-embedding-model/1
Simple description spanning
multiple lines.

<!-- asset-path: /path/to/model -->
<!-- module-type:   text-embedding   -->
<!-- fine-tunable:true -->
<!-- format: saved_model_2 -->

## Overview
"""

MINIMAL_MARKDOWN_WITH_ALLOWED_LICENSE = """# Module google/model/1
Simple description.

<!-- asset-path: /path/to/model -->
<!-- module-type: text-embedding -->
<!-- fine-tunable: true -->
<!-- format: saved_model_2 -->
<!-- license: BSD-3-Clause -->

## Overview
"""

MINIMAL_MARKDOWN_WITH_UNKNOWN_LICENSE = """# Module google/model/1
Simple description.

<!-- asset-path: /path/to/model -->
<!-- module-type: text-embedding -->
<!-- fine-tunable: true -->
<!-- format: saved_model_2 -->
<!-- license: my_license -->

## Overview
"""

MINIMAL_MARKDOWN_WITH_BAD_MODULE_TYPE = """# Module google/model/1
Simple description.

<!-- asset-path: /path/to/model -->
<!-- module-type: something-embedding -->
<!-- fine-tunable: true -->
<!-- format: saved_model_2 -->
<!-- license: my_license -->

## Overview
"""

MINIMAL_MARKDOWN_WITH_UNSUPPORTED_FORMAT = """# Module google/model/1
Simple description.

<!-- asset-path: /path/to/model -->
<!-- module-type: text-embedding -->
<!-- fine-tunable: true -->
<!-- format: unsupported -->
<!-- license: Apache-2.0 -->

## Overview
"""

MARKDOWN_WITH_DOUBLE_SLASH_IN_HANDLE = """# Module google/model//1
Simple description.
"""

MARKDOWN_WITH_BAD_CHARS_IN_HANDLE = """# Module google/text-embedding&nbsp;/1
Simple description.
"""

MARKDOWN_WITH_MISSING_MODEL_IN_HANDLE = """# Module google/1
Simple description.
"""

MARKDOWN_WITH_MISSING_VERSION_IN_HANDLE = """# Module google/model
Simple description.
"""

MARKDOWN_WITHOUT_DESCRIPTION = """# Module google/text-embedding-model/1

<!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
<!-- format: saved_model_2 -->

## Overview
"""

MARKDOWN_WITHOUT_DESCRIPTION_WITHOUT_LINEBREAK = """# Module google/text-embedding-model/1
<!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
<!-- format: saved_model_2 -->

## Overview
"""

MARKDOWN_WITH_MISSING_METADATA = """# Module google/text-embedding-model/1
One line description.
<!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
<!-- format: saved_model_2 -->

## Overview
"""

MARKDOWN_WITH_FORBIDDEN_DUPLICATE_METADATA = """# Module google/model/1
One line description.
<!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
<!-- asset-path: https://path/to/text-embedding-model/model2.tar.gz -->
<!-- module-type: text-embedding -->
<!-- fine-tunable: true -->
<!-- format: saved_model_2 -->

## Overview
"""

MARKDOWN_WITH_ALLOWED_DUPLICATE_METADATA = """# Module google/model/1
One line description.
<!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
<!-- module-type: text-classification -->
<!-- module-type: text-embedding -->
<!-- dataset: MNIST -->
<!-- dataset: Wikipedia -->
<!-- language: en -->
<!-- language: fr -->
<!-- network-architecture: Transformer -->
<!-- network-architecture: EfficientNet -->
<!-- fine-tunable: true -->
<!-- format: saved_model_2 -->

## Overview
"""

MARKDOWN_WITH_LEGACY_TAG = """# Module google/text-embedding-model/1
One line description.
<!-- asset-path: legacy -->
<!-- module-type: text-embedding -->
<!-- fine-tunable: true -->
<!-- format: saved_model_2 -->

## Overview
"""

MARKDOWN_WITH_COLAB_BUTTON = """# Module google/text-embedding-model/1
One line description.
<!-- asset-path: legacy -->
<!-- module-type: text-embedding -->
<!-- fine-tunable: true -->
<!-- format: saved_model_2 -->

[![Open Colab notebook]](https://colab.research.google.com)

## Overview
"""

MARKDOWN_WITH_DEMO_BUTTON = """# Module google/text-embedding-model/1
One line description.
<!-- asset-path: legacy -->
<!-- module-type: text-embedding -->
<!-- fine-tunable: true -->
<!-- format: saved_model_2 -->

[![Open Demo]](https://teachablemachine.withgoogle.com/train/pose)

## Overview
"""

MARKDOWN_WITH_UNEXPECTED_LINES = """# Module google/text-embedding-model/1
One line description.
<!-- module-type: text-embedding -->

This should not be here.
<!-- format: saved_model_2 -->

## Overview
"""

MINIMAL_COLLECTION_MARKDOWN = """# Collection google/text-embedding-collection/1
Simple description spanning
multiple lines.

<!-- module-type: text-embedding -->

## Overview
"""

MARKDOWN_COLLECTION_UNSUPPORTED_LANGUAGE = """# Collection google/model/1
Simple description spanning
multiple lines.

<!-- module-type: text-embedding -->
<!-- language: non-existent -->

## Overview
"""

MAXIMAL_COLLECTION_MARKDOWN = """# Collection google/text-embedding-collection/1
Simple description spanning
multiple lines.

<!-- module-type: text-embedding -->
<!-- dataset: MNIST -->
<!-- language: en -->
<!-- network-architecture: BERT -->

## Overview
"""

MINIMAL_PUBLISHER_MARKDOWN = """# Publisher %s
Simple description spanning one line.

[![Icon URL]](https://path/to/icon.png)

## Overview
"""


class ValidatorTest(tf.test.TestCase):

  def setUp(self):
    super(tf.test.TestCase, self).setUp()
    self.tmp_dir = self.create_tempdir()
    self.tmp_root_dir = os.path.join(self.tmp_dir, "root")
    self.tmp_docs_dir = os.path.join(self.tmp_root_dir, "assets", "docs")
    self.model_path = os.path.join(self.tmp_dir, "model_1")
    self.not_a_model_path = os.path.join(self.tmp_dir, "not_a_model")
    self.save_dummy_model(self.model_path)
    self.minimal_markdown = MINIMAL_MARKDOWN_SAVED_MODEL_TEMPLATE % self.model_path
    self.maximal_markdown = MAXIMAL_MARKDOWN_SAVED_MODEL_TEMPLATE % self.model_path
    self.minimal_markdown_with_bad_model = (
        MINIMAL_MARKDOWN_SAVED_MODEL_TEMPLATE % self.not_a_model_path)
    language_yaml = "values:\n  - id: en\n    display_name: English"
    self.set_content("root/tags/language.yaml", language_yaml)
    self.asset_path_modified = mock.patch.object(
        validator.DocumentationParser,
        "_is_asset_path_modified",
        return_value=True)
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
                     MINIMAL_PUBLISHER_MARKDOWN % publisher)

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
    for markdown in [
        self.minimal_markdown, self.maximal_markdown,
        MARKDOWN_WITH_EMPTY_SECOND_LINE % self.model_path
    ]:
      self.set_up_publisher_page("google")
      self.set_content(
          "root/assets/docs/google/models/text-embedding-model/1.md", markdown)
      validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_markdown_parsed_placeholder(self):
    self.set_up_publisher_page("google")
    for markdown in [
        MINIMAL_MARKDOWN_PLACEHOLDER_TEMPLATE,
        MAXIMAL_MARKDOWN_PLACEHOLDER_TEMPLATE
    ]:
      self.set_content(
          "root/assets/docs/google/models/text-embedding-model/1.md", markdown)
      validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_minimal_markdown_parsed_lite(self):
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     (MINIMAL_MARKDOWN_LITE_TEMPLATE % self.model_path))
    self.set_up_publisher_page("google")
    validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_minimal_markdown_parsed_tfjs(self):
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     (MINIMAL_MARKDOWN_TFJS_TEMPLATE % self.model_path))
    self.set_up_publisher_page("google")
    validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_minimal_markdown_parsed_coral(self):
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     (MINIMAL_MARKDOWN_CORAL_TEMPLATE % self.model_path))
    self.set_up_publisher_page("google")
    validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_minimal_markdown_parsed_with_selected_files(self):
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     self.minimal_markdown)
    self.set_up_publisher_page("google")
    validator.validate_documentation_files(
        root_dir=self.tmp_root_dir,
        files_to_validate=["google/models/text-embedding-model/1.md"])

  def test_collection_markdown_parsed(self):
    self.set_up_publisher_page("google")
    for markdown in [MINIMAL_COLLECTION_MARKDOWN, MAXIMAL_COLLECTION_MARKDOWN]:
      self.set_content(
          "root/assets/docs/google/collections/text-embedding-collection/1.md",
          markdown)
      validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_minimal_publisher_markdown_parsed(self):
    self.set_up_publisher_page("some-publisher")
    validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_invalid_markdown_fails(self):
    self.set_content("root/assets/docs/publisher/model/1.md",
                     "INVALID MARKDOWN")
    with self.assertRaisesRegexp(validator.MarkdownDocumentationError,
                                 ".*First line.*"):
      validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_minimal_markdown_not_in_publisher_dir(self):
    self.set_content("root/assets/docs/gooogle/models/wrong-location/1.md",
                     self.minimal_markdown)
    with self.assertRaisesRegexp(validator.MarkdownDocumentationError,
                                 ".*placed in the publisher directory.*"):
      validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_fails_if_publisher_page_does_not_exist(self):
    self.set_content(
        "root/assets/docs/publisher-without-page/models/text-embedding-model/1.md",
        MINIMAL_MARKDOWN_WITH_UNKNOWN_PUBLISHER)
    with self.assertRaisesRegexp(validator.MarkdownDocumentationError,
                                 ".*Publisher documentation does not.*"):
      validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_minimal_markdown_does_not_end_with_md_fails(self):
    self.set_content("root/assets/docs/google/models/wrong-extension/1.mdz",
                     self.minimal_markdown)
    with self.assertRaisesRegexp(validator.MarkdownDocumentationError,
                                 r".*end with '\.md.'*"):
      validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_publisher_markdown_at_incorrect_location_fails(self):
    self.set_content("root/assets/docs/google/publisher.md",
                     MINIMAL_PUBLISHER_MARKDOWN % "some-publisher")
    with self.assertRaisesRegexp(validator.MarkdownDocumentationError,
                                 r".*some-publisher\.md.*"):
      validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_publisher_markdown_at_correct_location(self):
    self.set_up_publisher_page("some-publisher")
    validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_markdown_with_bad_handle(self):
    for markdown in [
        MARKDOWN_WITH_DOUBLE_SLASH_IN_HANDLE, MARKDOWN_WITH_BAD_CHARS_IN_HANDLE,
        MARKDOWN_WITH_MISSING_MODEL_IN_HANDLE,
        MARKDOWN_WITH_MISSING_VERSION_IN_HANDLE
    ]:
      self.set_content("root/assets/docs/google/models/model/1.md", markdown)
      with self.assertRaisesRegexp(validator.MarkdownDocumentationError,
                                   ".*First line of the documentation*"):
        validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_markdown_without_description(self):
    for markdown in [
        MARKDOWN_WITHOUT_DESCRIPTION,
        MARKDOWN_WITHOUT_DESCRIPTION_WITHOUT_LINEBREAK
    ]:
      self.set_content(
          "root/assets/docs/google/models/text-embedding-model/1.md", markdown)
      with self.assertRaisesRegexp(validator.MarkdownDocumentationError,
                                   ".*has to contain a short description.*"):
        validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_markdown_with_missing_metadata(self):
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     MARKDOWN_WITH_MISSING_METADATA)
    with self.assertRaisesRegexp(validator.MarkdownDocumentationError,
                                 ".*missing.*fine-tunable.*module-type.*"):
      validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_markdown_with_unsupported_format_metadata(self):
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     MINIMAL_MARKDOWN_WITH_UNSUPPORTED_FORMAT)
    with self.assertRaisesRegexp(
        validator.MarkdownDocumentationError, "The 'format' metadata.*but "
        "was 'unsupported'."):
      validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_markdown_with_forbidden_duplicate_metadata(self):
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     MARKDOWN_WITH_FORBIDDEN_DUPLICATE_METADATA)
    with self.assertRaisesRegexp(validator.MarkdownDocumentationError,
                                 ".*duplicate.*asset-path.*"):
      validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_markdown_with_allowed_duplicate_metadata(self):
    self.set_up_publisher_page("google")
    language_yaml = """values:
      - id: en
        display_name: English
      - id: fr
        display_name: French"""
    self.set_content("root/tags/language.yaml", language_yaml)
    self.set_content("root/assets/docs/google/models/model/1.md",
                     MARKDOWN_WITH_ALLOWED_DUPLICATE_METADATA)
    validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_markdown_with_unexpected_lines(self):
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     MARKDOWN_WITH_UNEXPECTED_LINES)
    with self.assertRaisesRegexp(validator.MarkdownDocumentationError,
                                 ".*Unexpected line.*"):
      validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_minimal_markdown_parsed_full(self):
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     self.minimal_markdown)
    self.set_up_publisher_page("google")
    documentation_parser = validator.DocumentationParser(
        self.tmp_root_dir, self.tmp_docs_dir)
    documentation_parser.validate(
        file_path=self.get_full_path(
            "root/assets/docs/google/models/text-embedding-model/1.md"),
        do_smoke_test=True)
    self.assertEqual("Simple description spanning multiple lines.",
                     documentation_parser.parsed_description)
    expected_metadata = {
        "asset-path": {self.model_path},
        "module-type": {"text-embedding"},
        "fine-tunable": {"true"},
        "format": {"saved_model_2"},
    }
    self.assertAllEqual(expected_metadata, documentation_parser.parsed_metadata)

  def test_asset_path_is_github_download_url_test(self):
    self.set_content(
        "root/assets/docs/google/models/text-embedding-model/1.md",
        MINIMAL_MARKDOWN_SAVED_MODEL_TEMPLATE %
        "https://github.com/some_repo/releases/download/some_path")
    self.set_up_publisher_page("google")
    with self.assertRaisesRegexp(validator.MarkdownDocumentationError,
                                 ".*cannot be automatically fetched.*"):
      validator.validate_documentation_files(
          root_dir=self.tmp_root_dir,
          files_to_validate=["google/models/text-embedding-model/1.md"],
          do_smoke_test=True)

  def test_asset_path_is_legacy_and_modified(self):
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     MARKDOWN_WITH_LEGACY_TAG)
    self.set_up_publisher_page("google")
    with self.assertRaisesRegexp(validator.MarkdownDocumentationError,
                                 ".*failed to parse.*"):
      validator.validate_documentation_files(
          root_dir=self.tmp_root_dir,
          files_to_validate=["google/models/text-embedding-model/1.md"],
          do_smoke_test=True)

  def test_asset_path_is_legacy_and_unmodified(self):
    self.asset_path_modified = mock.patch.object(
        validator.DocumentationParser,
        "_is_asset_path_modified",
        return_value=False)
    self.asset_path_modified.start()
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     MARKDOWN_WITH_LEGACY_TAG)
    self.set_up_publisher_page("google")
    validator.validate_documentation_files(
        root_dir=self.tmp_root_dir,
        files_to_validate=["google/models/text-embedding-model/1.md"],
        do_smoke_test=True)

  def test_bad_model_does_not_pass_smoke_test(self):
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     self.minimal_markdown_with_bad_model)
    self.set_up_publisher_page("google")
    with self.assertRaisesRegexp(validator.MarkdownDocumentationError,
                                 ".*failed to parse.*"):
      validator.validate_documentation_files(
          root_dir=self.tmp_root_dir,
          files_to_validate=["google/models/text-embedding-model/1.md"],
          do_smoke_test=True
      )

  def test_markdown_with_allowed_license(self):
    self.set_content("root/assets/docs/google/models/model/1.md",
                     MINIMAL_MARKDOWN_WITH_ALLOWED_LICENSE)
    self.set_up_publisher_page("google")
    validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_markdown_colab_button(self):
    self.set_content("root/assets/docs/google/models/model/1.md",
                     MARKDOWN_WITH_COLAB_BUTTON)
    self.set_up_publisher_page("google")
    validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_markdown_demo_button(self):
    self.set_content("root/assets/docs/google/models/model/1.md",
                     MARKDOWN_WITH_DEMO_BUTTON)
    self.set_up_publisher_page("google")
    validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_markdown_with_unknown_license(self):
    self.set_content("root/assets/docs/google/models/model/1.md",
                     MINIMAL_MARKDOWN_WITH_UNKNOWN_LICENSE)
    self.set_up_publisher_page("google")
    with self.assertRaisesRegexp(validator.MarkdownDocumentationError,
                                 ".*specify a license id from list.*"):
      validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_markdown_with_bad_module_type(self):
    self.set_content("root/assets/docs/google/models/model/1.md",
                     MINIMAL_MARKDOWN_WITH_BAD_MODULE_TYPE)
    self.set_up_publisher_page("google")
    with self.assertRaisesRegexp(validator.MarkdownDocumentationError,
                                 ".*metadata has to start with.*"):
      validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_markdown_with_forbidden_format_metadata(self):
    self.set_content("root/assets/docs/google/models/model/1.md",
                     MINIMAL_MARKDOWN_LITE_WITH_FORBIDDEN_FORMAT)
    self.set_up_publisher_page("google")
    with self.assertRaisesRegexp(
        validator.MarkdownDocumentationError,
        r".*contains unsupported metadata properties: \['format'\].*"):
      validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_markdown_with_unsupported_metadata(self):
    self.set_up_publisher_page("google")
    for markdown in [
        MARKDOWN_SAVED_MODEL_UNSUPPORTED_TAG,
        MARKDOWN_PLACEHOLDER_UNSUPPORTED_TAG,
        MINIMAL_MARKDOWN_LITE_WITH_UNSUPPORTED_TAG
    ]:
      self.set_content("root/assets/docs/google/models/model/1.md", markdown)
      with self.assertRaisesRegexp(
          validator.MarkdownDocumentationError,
          r".*contains unsupported metadata properties: \['unsupported_tag'\].*"
      ):
        validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_markdown_with_unsupported_language(self):
    self.set_up_publisher_page("google")
    for markdown in [
        MARKDOWN_COLLECTION_UNSUPPORTED_LANGUAGE,
        MARKDOWN_PLACEHOLDER_UNSUPPORTED_LANGUAGE,
        MARKDOWN_SAVED_MODEL_UNSUPPORTED_LANGUAGE
    ]:
      self.set_content("root/assets/docs/google/models/model/1.md", markdown)
      with self.assertRaisesRegexp(
          validator.MarkdownDocumentationError,
          r".*Unsupported languages were found: {'non-existent'}. "
          r"Please add them to .*"):
        validator.validate_documentation_dir(root_dir=self.tmp_root_dir)

  def test_model_with_invalid_filenames_fails_smoke_test(self):
    self.set_content("root/assets/docs/google/models/text-embedding-model/1.md",
                     self.minimal_markdown)
    self.set_up_publisher_page("google")
    with open(os.path.join(self.model_path, ".invalid_file"), "w") as bad_file:
      bad_file.write("This file shouldn't be here")
    documentation_parser = validator.DocumentationParser(
        self.tmp_root_dir, self.tmp_docs_dir)
    with self.assertRaisesRegexp(validator.MarkdownDocumentationError,
                                 r"Invalid filepath.*\.invalid_file"):
      documentation_parser.validate(
          file_path=self.get_full_path(
              "root/assets/docs/google/models/text-embedding-model/1.md"),
          do_smoke_test=True)


if __name__ == "__main__":
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
