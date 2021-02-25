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
"""Markdown documentation validator for published models.

1) To validate selected files, run from the project root path:
$ python tools/validator.py vtab/models/wae-ukl/1.md [other_files]

This will download and smoke test the model specified on asset-path metadata.

2) To validate all documentation files, run from the project root path:
$ python tools/validator.py

This does not download and smoke test the model.

3) To validate files from outside the project root path, use the --root_dir
flag:
$ python tools/validator.py --root_dir=path_to_project_root
"""

import abc
import argparse
import os
import re
import subprocess
import sys
from absl import app
from absl import logging

import tensorflow as tf
import tensorflow_hub as hub

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.saved_model import loader_impl
# pylint: enable=g-direct-tensorflow-import

FLAGS = None

# Regex pattern for the first line of the documentation of Saved Models.
# Example: "Module google/universal-sentence-encoder/1"
MODEL_HANDLE_PATTERN = (
    r"# Module (?P<publisher>[\w-]+)/(?P<name>([\w\.-]+(/[\w\.-]+)*))/(?P<vers>\d+)")  # pylint: disable=line-too-long
# Regex pattern for the first line of the documentation of placeholder MD files.
# Example: "Placeholder google/universal-sentence-encoder/1"
PLACEHOLDER_HANDLE_PATTERN = (
    r"# Placeholder "
    r"(?P<publisher>[\w-]+)/(?P<name>([\w\.-]+(/[\w\.-]+)*))/(?P<vers>\d+)")  # pylint: disable=line-too-long
# Regex pattern for the first line of the documentation of TF Lite models.
# Example: "# Lite google/spice/1"
LITE_HANDLE_PATTERN = (
    r"# Lite (?P<publisher>[\w-]+)/(?P<name>([\w\.-]+(/[\w\.-]+)*))/(?P<vers>\d+)")  # pylint: disable=line-too-long
# Regex pattern for the first line of the documentation of TFJS models.
# Example: "# Tfjs google/spice/1/default/1"
TFJS_HANDLE_PATTERN = (
    r"# Tfjs (?P<publisher>[\w-]+)/(?P<name>([\w\.-]+(/[\w\.-]+)*))/(?P<vers>\d+)")  # pylint: disable=line-too-long
# Regex pattern for the first line of the documentation of Coral models.
# Example: "# Coral tensorflow/mobilenet_v2_1.0_224_quantized/1/default/1"
CORAL_HANDLE_PATTERN = (
    r"# Coral (?P<publisher>[\w-]+)/(?P<name>([\w\.-]+(/[\w\.-]+)*))/(?P<vers>\d+)")  # pylint: disable=line-too-long
# Regex pattern for the first line of the documentation of publishers.
# Example: "Publisher google"
PUBLISHER_HANDLE_PATTERN = r"# Publisher (?P<publisher>[\w-]+)"
# Regex pattern for the first line of the documentation of collections.
# Example: "Collection google/universal-sentence-encoders/1"
COLLECTION_HANDLE_PATTERN = (
    r"# Collection (?P<publisher>[\w-]+)/(?P<name>(\w|-|/|&|;|\.)+)/(\d+)")
# Regex pattern for the line of the documentation describing model metadata.
# Example: "<!-- fine-tunable: true -->"
# Note: Both key and value consumes free space characters, but later on these
# are stripped.
METADATA_LINE_PATTERN = r"^<!--(?P<key>(\w|\s|-)+):(?P<value>.+)-->$"

# Specifies whether a SavedModel is a Hub Module or a TF1/TF2 SavedModel.
SAVED_MODEL_FORMATS = ("hub", "saved_model", "saved_model_2")

# Map a handle pattern to the corresponding model type name.
HANDLE_PATTERN_TO_MODEL_TYPE = {
    MODEL_HANDLE_PATTERN: "Module",
    PLACEHOLDER_HANDLE_PATTERN: "Placeholder",
    LITE_HANDLE_PATTERN: "Lite",
    TFJS_HANDLE_PATTERN: "Tfjs",
    CORAL_HANDLE_PATTERN: "Coral"
}


def _recursive_list_dir(root_dir):
  """Yields all files of a root directory tree."""
  for dirname, _, filenames in tf.io.gfile.walk(root_dir):
    for filename in filenames:
      yield os.path.join(dirname, filename)


class Filesystem(object):
  """Convenient (and mockable) file system access."""

  def get_contents(self, filename):
    """Returns file contents as a string."""
    with tf.io.gfile.GFile(filename, "r") as f:
      return f.read()

  def file_exists(self, filename):
    """Returns whether file exists."""
    return tf.io.gfile.exists(filename)

  def recursive_list_dir(self, root_dir):
    """Yields all files of a root directory tree."""
    return _recursive_list_dir(root_dir)


class MarkdownDocumentationError(Exception):
  """Problem with markdown syntax parsing."""


def _validate_file_paths(model_dir):
  valid_path_regex = re.compile(r"(/[\w-][!',_\w\.\-=:% ]*)+")
  for filepath in _recursive_list_dir(model_dir):
    if not valid_path_regex.fullmatch(filepath):
      raise MarkdownDocumentationError(f"Invalid filepath in asset: {filepath}")


class ParsingPolicy(object):
  """The base class for type specific parsing policies.

  Documentation files for models, placeholders, publishers and collections share
  a publisher field, a readable name, a correct file path etc.
  """

  def __init__(self, publisher, model_name, model_version, required_metadata,
               optional_metadata):
    self._publisher = publisher
    self._model_name = model_name
    self._model_version = model_version
    self._required_metadata = required_metadata
    self._optional_metadata = optional_metadata

  @property
  @abc.abstractmethod
  def type_name(self):
    """Return readable name of the parsed type."""

  @property
  def publisher(self):
    return self._publisher

  @property
  def supported_metadata(self):
    """Return which metadata tags are supported."""
    return self._required_metadata + self._optional_metadata

  def get_top_level_dir(self, root_dir):
    """Returns the top level publisher directory."""
    return os.path.join(root_dir, self._publisher)

  def assert_correct_file_path(self, file_path, root_dir):
    if not file_path.endswith(".md"):
      raise MarkdownDocumentationError(
          "Documentation file does not end with '.md': %s" % file_path)

    publisher_dir = self.get_top_level_dir(root_dir)
    if not file_path.startswith(publisher_dir + "/"):
      raise MarkdownDocumentationError(
          "Documentation file is not on a correct path. Documentation for a "
          f"{self.type_name} with publisher '{self._publisher}' should be "
          f"placed in the publisher directory: '{publisher_dir}'")

  def assert_can_resolve_asset(self, asset_path):
    """Check whether the asset path can be resolved."""
    pass

  def assert_metadata_contains_required_fields(self, metadata):
    required_metadata = set(self._required_metadata)
    provided_metadata = set(metadata.keys())
    if not provided_metadata.issuperset(required_metadata):
      raise MarkdownDocumentationError(
          "The MD file is missing the following required metadata properties: "
          "%s. Please refer to README.md for information about markdown "
          "format." % sorted(required_metadata.difference(provided_metadata)))

  def assert_metadata_contains_supported_fields(self, metadata):
    supported_metadata = set(self.supported_metadata)
    provided_metadata = set(metadata.keys())
    if not supported_metadata.issuperset(provided_metadata):
      raise MarkdownDocumentationError(
          "The MD file contains unsupported metadata properties: "
          f"{sorted(provided_metadata.difference(supported_metadata))}. Please "
          "refer to README.md for information about markdown format.")

  def assert_no_duplicate_metadata(self, metadata):
    duplicate_metadata = list()
    for key, values in metadata.items():
      if key in self.supported_metadata and len(values) > 1:
        duplicate_metadata.append(key)
    if duplicate_metadata:
      raise MarkdownDocumentationError(
          "There are duplicate metadata values. Please refer to "
          "README.md for information about markdown format. In particular the "
          f"duplicated metadata are: {sorted(duplicate_metadata)}")

  def assert_correct_module_types(self, metadata):
    if "module-type" in metadata:
      allowed_prefixes = ["image-", "text-", "audio-", "video-"]
      for value in metadata["module-type"]:
        if all([not value.startswith(prefix) for prefix in allowed_prefixes]):
          raise MarkdownDocumentationError(
              "The 'module-type' metadata has to start with any of 'image-'"
              ", 'text', 'audio-', 'video-', but is: '{value}'")

  def assert_correct_metadata(self, metadata):
    """Assert that correct metadata is present."""
    self.assert_metadata_contains_required_fields(metadata)
    self.assert_metadata_contains_supported_fields(metadata)
    self.assert_no_duplicate_metadata(metadata)
    self.assert_correct_module_types(metadata)


class CollectionParsingPolicy(ParsingPolicy):
  """ParsingPolicy for collection documentation."""

  def __init__(self, publisher, model_name, model_version):
    super(CollectionParsingPolicy,
          self).__init__(publisher, model_name, model_version, ["module-type"],
                         ["dataset", "language", "network-architecture"])

  @property
  def type_name(self):
    return "Collection"


class PlaceholderParsingPolicy(ParsingPolicy):
  """ParsingPolicy for placeholder files."""

  def __init__(self, publisher, model_name, model_version):
    super(PlaceholderParsingPolicy, self).__init__(
        publisher, model_name, model_version, ["module-type"], [
            "dataset", "fine-tunable", "interactive-model-name", "language",
            "license", "network-architecture"
        ])

  @property
  def type_name(self):
    return "Placeholder"


class SavedModelParsingPolicy(ParsingPolicy):
  """ParsingPolicy for SavedModel documentation."""

  def __init__(self, publisher, model_name, model_version):
    super(SavedModelParsingPolicy, self).__init__(
        publisher, model_name, model_version,
        ["asset-path", "module-type", "fine-tunable", "format"], [])

  @property
  def type_name(self):
    return "Module"

  def assert_correct_metadata(self, metadata):
    self.assert_metadata_contains_required_fields(metadata)
    self.assert_no_duplicate_metadata(metadata)
    self.assert_correct_module_types(metadata)

    format_value = list(metadata["format"])[0]
    if format_value not in SAVED_MODEL_FORMATS:
      raise MarkdownDocumentationError(
          f"The 'format' metadata should be one of {SAVED_MODEL_FORMATS} "
          f"but was '{format_value}'.")

  def assert_can_resolve_asset(self, asset_path):
    """Attempt to hub.resolve the given asset path."""
    try:
      resolved_model = hub.resolve(asset_path)
      loader_impl.parse_saved_model(resolved_model)
      _validate_file_paths(resolved_model)
    except Exception as e:  # pylint: disable=broad-except
      raise MarkdownDocumentationError(
          f"The model on path {asset_path} failed to parse. Please make sure "
          "that the asset-path metadata points to a valid TF2 SavedModel or a "
          "TF1 Hub module, compressed as described in section 'Model' of "
          f"README.md. Underlying reason for failure: {e}.")


class TfjsParsingPolicy(ParsingPolicy):
  """ParsingPolicy for TF.js documentation."""

  def __init__(self, publisher, model_name, model_version):
    super(TfjsParsingPolicy,
          self).__init__(publisher, model_name, model_version,
                         ["asset-path", "parent-model"], [])

  @property
  def type_name(self):
    return "Tfjs"


class LiteParsingPolicy(TfjsParsingPolicy):
  """ParsingPolicy for TFLite documentation."""

  @property
  def type_name(self):
    return "Lite"


class CoralParsingPolicy(TfjsParsingPolicy):
  """ParsingPolicy for Coral documentation."""

  @property
  def type_name(self):
    return "Coral"


class PublisherParsingPolicy(ParsingPolicy):
  """ParsingPolicy for publisher documentation.

  Publisher files should always be at root/publisher/publisher.md and they
  should not contain a 'format' tag as it has no effect.
  """

  def __init__(self, publisher, model_name, model_version):
    super(PublisherParsingPolicy, self).__init__(publisher, model_name,
                                                 model_version, [], [])

  @property
  def type_name(self):
    return "Publisher"

  def get_expected_file_path(self, root_dir):
    """Returns the expected path of the documentation file."""
    return os.path.join(root_dir, self._publisher, self._publisher + ".md")

  def assert_correct_file_path(self, file_path, root_dir):
    """Extend base method by also checking for /publisher/publisher.md."""
    expected_file_path = self.get_expected_file_path(root_dir)
    if expected_file_path and file_path != expected_file_path:
      raise MarkdownDocumentationError(
          "Documentation file is not on a correct path. Documentation for the "
          f"publisher '{self.publisher}' should be submitted to "
          f"'{expected_file_path}'")
    super().assert_correct_file_path(file_path, root_dir)


class DocumentationParser(object):
  """Class used for parsing model documentation strings."""

  def __init__(self, documentation_dir, filesystem):
    self._documentation_dir = documentation_dir
    self._filesystem = filesystem
    self._parsed_metadata = dict()
    self._parsed_description = ""
    self._file_path = ""
    self._lines = []
    self._current_index = 0

  @property
  def parsed_description(self):
    return self._parsed_description

  @property
  def parsed_metadata(self):
    return self._parsed_metadata

  def raise_error(self, message):
    message_with_file = f"Error at file {self._file_path}: {message}"
    raise MarkdownDocumentationError(message_with_file)

  def get_policy_from_first_line(self, first_line):
    """Return an appropriate ParsingPolicy instance for the first line."""
    patterns_and_policies = [
        (MODEL_HANDLE_PATTERN, SavedModelParsingPolicy),
        (PLACEHOLDER_HANDLE_PATTERN, PlaceholderParsingPolicy),
        (LITE_HANDLE_PATTERN, LiteParsingPolicy),
        (TFJS_HANDLE_PATTERN, TfjsParsingPolicy),
        (CORAL_HANDLE_PATTERN, CoralParsingPolicy),
        (PUBLISHER_HANDLE_PATTERN, PublisherParsingPolicy),
        (COLLECTION_HANDLE_PATTERN, CollectionParsingPolicy),
    ]
    for pattern, policy in patterns_and_policies:
      match = re.match(pattern, first_line)
      if not match:
        continue
      groups = match.groupdict()
      return policy(
          groups.get("publisher"), groups.get("name"), groups.get("vers"))

    self.raise_error(
        "First line of the documentation file must match one of the following "
        "formats depending on the MD type:\n"
        f"TF Model: {MODEL_HANDLE_PATTERN}\n"
        f"TFJS: {TFJS_HANDLE_PATTERN}\n"
        f"Lite: {LITE_HANDLE_PATTERN}\n"
        f"Coral: {CORAL_HANDLE_PATTERN}\n"
        f"Publisher: {PUBLISHER_HANDLE_PATTERN}\n"
        f"Collection: {COLLECTION_HANDLE_PATTERN}\n"
        f"Placeholder: {PLACEHOLDER_HANDLE_PATTERN}\n"
        "For example '# Module google/text-embedding-model/1'. Instead the "
        f"first line is '{first_line}'")

  def assert_publisher_page_exists(self):
    """Assert that publisher page exists for the publisher of this model."""
    # Use a publisher policy to get the expected documentation page path.
    publisher_policy = PublisherParsingPolicy(self.policy.publisher, None, None)
    expected_publisher_doc_file_path = publisher_policy.get_expected_file_path(
        self._documentation_dir)
    if not self._filesystem.file_exists(expected_publisher_doc_file_path):
      self.raise_error(
          "Publisher documentation does not exist. "
          f"It should be added to {expected_publisher_doc_file_path}.")

  def consume_description(self):
    """Consume second line with a short model description."""
    description_lines = []
    self._current_index = 1
    # Allow an empty line between handle and description.
    if not self._lines[self._current_index]:
      self._current_index += 1
    while self._lines[self._current_index] and not self._lines[
        self._current_index].startswith("<!--"):
      description_lines.append(self._lines[self._current_index])
      self._current_index += 1
    self._parsed_description = " ".join(description_lines)
    if not self._parsed_description:
      raise MarkdownDocumentationError(
          "Second line of the documentation file has to contain a short "
          "description. For example 'Word2vec text embedding model.'.")

  def consume_metadata(self):
    """Consume all metadata."""
    while self._current_index < len(
        self._lines) and (not self._lines[self._current_index].startswith("#")):
      if not self._lines[self._current_index]:
        # Empty line is ok.
        self._current_index += 1
        continue
      match = re.match(METADATA_LINE_PATTERN, self._lines[self._current_index])
      if match:
        # Add found metadata.
        groups = match.groupdict()
        key = groups.get("key").strip()
        value = groups.get("value").strip()
        if key not in self._parsed_metadata:
          self._parsed_metadata[key] = set()
        self._parsed_metadata[key].add(value)
        self._current_index += 1
        continue
      if self._lines[self._current_index].startswith("[![Icon URL]]"):
        # Icon for publishers.
        self._current_index += 1
        continue
      if self._lines[self._current_index].startswith(
          "[![Open Colab notebook]]"):
        # Colab.
        self._current_index += 1
        continue
      if self._lines[self._current_index].startswith(
          "[![Open Demo]]"):
        # Demo button.
        self._current_index += 1
        continue
      # Not an empty line and not expected metadata.
      raise MarkdownDocumentationError(
          f"Unexpected line found: '{self._lines[self._current_index]}'. "
          "Please refer to [README.md]"
          "(https://github.com/tensorflow/tfhub.dev/blob/master/README.md) "
          "for information about markdown format.")

  def assert_allowed_license(self):
    """Validate provided license."""
    if "license" in self._parsed_metadata:
      license_id = list(self._parsed_metadata["license"])[0]
      allowed_license_ids = [
          "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause", "CC-BY-NC-4.0",
          "GPL-2.0", "GPL-3.0", "LGPL-2.0", "LGPL-2.1", "LGPL-3.0", "MIT",
          "MPL-2.0", "CDDL-1.0", "EPL-2.0", "custom"
      ]
      if license_id not in allowed_license_ids:
        self.raise_error(
            f"The license {license_id} provided in metadata is not allowed. "
            "Please specify a license id from list of allowed ids: "
            f"[{allowed_license_ids}]. Example: <!-- license: Apache-2.0 -->")

  def _is_asset_path_modified(self):
    """Return True if the asset-path tag has been added or modified."""
    # pylint: disable=subprocess-run-check
    command = f"git diff {self._file_path} | grep '+<!-- asset-path:'"
    result = subprocess.run(command, shell=True)
    # grep exits with code 1 if it cannot find "+<!-- asset-path" in `git diff`.
    # Raise an error if the exit code is not 0 or 1.
    if result.returncode == 0:
      return True
    elif result.returncode == 1:
      return False
    else:
      self.raise_error(
          f"{command} returned unexpected exit code {result.returncode}")

  def smoke_test_asset(self):
    """Smoke test asset provided on asset-path metadata."""
    if "asset-path" not in self._parsed_metadata:
      return

    if not self._is_asset_path_modified():
      logging.info("Skipping asset smoke test since the tag is not added or "
                   "modified.")
      return
    asset_path = list(self._parsed_metadata["asset-path"])[0]

    # GitHub's robots.txt disallows fetches to */download, which means that
    # the asset-path URL cannot be fetched. Markdown validation should fail if
    # asset-path matches this regex.
    github_download_url_regex = re.compile(
        "https://github.com/.*/releases/download/.*")
    if github_download_url_regex.fullmatch(asset_path):
      self.raise_error(
          f"The asset-path {asset_path} is a url that cannot be automatically "
          "fetched. Please provide an asset-path that is allowed to be fetched "
          "by its robots.txt.")
    self.policy.assert_can_resolve_asset(asset_path)

  def validate(self, file_path, do_smoke_test):
    """Validate one documentation markdown file."""
    self._file_path = file_path
    raw_content = self._filesystem.get_contents(self._file_path)
    self._lines = raw_content.split("\n")
    first_line = self._lines[0].replace("&zwnj;", "")
    self.policy = self.get_policy_from_first_line(first_line)

    try:
      self.policy.assert_correct_file_path(self._file_path,
                                           self._documentation_dir)
      # Populate _parsed_description with the description
      self.consume_description()
      # Populate _parsed_metadata with the metadata tag mapping
      self.consume_metadata()
      self.policy.assert_correct_metadata(self._parsed_metadata)
    except MarkdownDocumentationError as e:
      self.raise_error(e)
    self.assert_allowed_license()
    self.assert_publisher_page_exists()
    if do_smoke_test:
      self.smoke_test_asset()


def validate_documentation_files(documentation_dir,
                                 files_to_validate=None,
                                 filesystem=Filesystem()):
  """Validate documentation files in a directory."""
  file_paths = list(filesystem.recursive_list_dir(documentation_dir))
  do_smoke_test = bool(files_to_validate)
  validated = 0
  for file_path in file_paths:
    if files_to_validate and file_path[len(documentation_dir) +
                                       1:] not in files_to_validate:
      continue
    logging.info("Validating %s.", file_path)
    documentation_parser = DocumentationParser(documentation_dir, filesystem)
    documentation_parser.validate(file_path, do_smoke_test)
    validated += 1
  logging.info("Found %d matching files - all validated successfully.",
               validated)
  if not do_smoke_test:
    logging.info(
        "No models were smoke tested. To download and smoke test a specific "
        "model, specify files directly in the command line, for example: "
        "'python tools/validator.py vtab/models/wae-ukl/1.md'")
  return validated


def main(_):
  root_dir = FLAGS.root_dir or os.getcwd()
  documentation_dir = os.path.join(root_dir, "assets", "docs")
  logging.info("Using %s for documentation directory.", documentation_dir)

  files_to_validate = None
  if FLAGS.file:
    files_to_validate = FLAGS.file
    logging.info("Going to validate files %s in documentation directory %s.",
                 files_to_validate, documentation_dir)
  else:
    logging.info("Going to validate all files in documentation directory %s.",
                 documentation_dir)

  validate_documentation_files(
      documentation_dir=documentation_dir, files_to_validate=files_to_validate)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "file",
      type=str,
      default=None,
      help=("Path to files to validate. Path is relative to `--root_dir`. "
            "The model will be smoke tested only for files specified by this "
            "flag."),
      nargs="*")
  parser.add_argument(
      "--root_dir",
      type=str,
      default=None,
      help=("Root directory that contains documentation files under "
            "./assets/docs. Defaults to current directory."))
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
