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
from typing import AbstractSet, Mapping, MutableSequence

from absl import app
from absl import logging
import attr
import tensorflow as tf
import tensorflow_hub as hub
import filesystem_utils
import yaml_parser as yaml_parser_lib

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.saved_model import loader_impl
# pylint: enable=g-direct-tensorflow-import

FLAGS = None

# Relative path from tfhub.dev/ to the docs/ directory.
DOCS_PATH = "assets/docs"

# Regex pattern for the first line of the documentation of Saved Models.
# Example: "Module google/universal-sentence-encoder/1"
MODEL_HANDLE_PATTERN = (
    r"# Module "
    r"(?P<publisher>[\w-]+)/(?P<name>([\w\.-]+(/[\w\.-]+)*))/(?P<vers>\d+)")
# Regex pattern for the first line of the documentation of placeholder MD files.
# Example: "Placeholder google/universal-sentence-encoder/1"
PLACEHOLDER_HANDLE_PATTERN = (
    r"# Placeholder "
    r"(?P<publisher>[\w-]+)/(?P<name>([\w\.-]+(/[\w\.-]+)*))/(?P<vers>\d+)")
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

# These metadata tags can be set to more than one value.
REPEATED_TAG_KEYS = ("dataset", "language", "module-type", "task",
                     "network-architecture")

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

TARFILE_SUFFIX = ".tar.gz"
TFLITE_SUFFIX = ".tflite"

# Dict key that maps to the specified asset-path of the Markdown file.
ASSET_PATH_KEY = "asset-path"


class MarkdownDocumentationError(Exception):
  """Problem with markdown syntax parsing."""


def _validate_file_paths(model_dir: str) -> None:
  valid_path_regex = re.compile(r"(/[\w-][!',_\w\.\-=:% ]*)+")
  for filepath in filesystem_utils.recursive_list_dir(model_dir):
    if not valid_path_regex.fullmatch(filepath):
      raise MarkdownDocumentationError(f"Invalid filepath in asset: {filepath}")


def _is_asset_path_modified(file_path: str) -> bool:
  """Returns True if the asset-path tag has been added or modified."""
  git_diff = subprocess.Popen(["git", "diff", "origin/master", file_path],
                              stdout=subprocess.PIPE)
  grep_asset_path = subprocess.Popen(["grep", "+<!-- asset-path:"],
                                     stdin=git_diff.stdout,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
  git_diff.stdout.close()
  grep_asset_path.communicate()
  return_code = grep_asset_path.returncode
  # grep exits with code 1 if it cannot find "+<!-- asset-path" in `git diff`.
  # Raise an error if the exit code is not 0 or 1.
  if return_code == 0:
    return True
  elif return_code == 1:
    return False
  else:
    raise MarkdownDocumentationError(
        f"Internal: grep command returned unexpected exit code {return_code}")


@attr.s(auto_attribs=True)
class ValidationConfig:
  """A simple value class containing information for the validation process.

  Attributes:
    skip_asset_check: A boolean indicating whether the "asset-path" tag should
      be skipped for validation. Defaults to False.
    do_smoke_test: A boolean indicating whether the referenced asset should be
      downloaded. Defaults to False.
  """
  skip_asset_check: bool = False
  do_smoke_test: bool = False


class ParsingPolicy:
  """The base class for type specific parsing policies.

  Documentation files for models, placeholders, publishers and collections share
  a publisher field, a readable name, a correct file path etc.
  """

  def __init__(self, publisher: str, model_name: str, model_version: str,
               required_metadata: MutableSequence[str],
               optional_metadata: MutableSequence[str]) -> None:
    self._publisher = publisher
    self._model_name = model_name
    self._model_version = model_version
    self._required_metadata = required_metadata
    self._optional_metadata = optional_metadata

  @property
  @abc.abstractmethod
  def type_name(self) -> str:
    """A readable name for the parsed type."""
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def supported_asset_path_suffix(self) -> str:
    """String describing the supported file ending of the compressed file."""
    raise NotImplementedError

  @property
  def publisher(self) -> str:
    return self._publisher

  @property
  def supported_metadata(self) -> MutableSequence[str]:
    """Return which metadata tags are supported."""
    return self._required_metadata + self._optional_metadata

  def get_top_level_dir(self, root_dir: str) -> str:
    """Returns the top level publisher directory."""
    return os.path.join(root_dir, self._publisher)

  def assert_correct_file_path(self, file_path: str, root_dir: str) -> None:
    if not file_path.endswith(".md"):
      raise MarkdownDocumentationError(
          "Documentation file does not end with '.md': %s" % file_path)

    publisher_dir = self.get_top_level_dir(root_dir)
    if not file_path.startswith(publisher_dir + "/"):
      raise MarkdownDocumentationError(
          "Documentation file is not on a correct path. Documentation for a "
          f"{self.type_name} with publisher '{self._publisher}' should be "
          f"placed in the publisher directory: '{publisher_dir}'")

  def _assert_can_resolve_asset(self, asset_path: str) -> None:
    """Check whether the asset path can be resolved."""
    pass

  def _assert_metadata_contains_required_fields(
      self, metadata: Mapping[str, AbstractSet[str]]) -> None:
    required_metadata = set(self._required_metadata)
    provided_metadata = set(metadata.keys())
    if not provided_metadata.issuperset(required_metadata):
      raise MarkdownDocumentationError(
          "The MD file is missing the following required metadata properties: "
          "%s. Please refer to https://www.tensorflow.org/hub/writing_model_documentation for information about markdown "
          "format." % sorted(required_metadata.difference(provided_metadata)))

  def _assert_metadata_contains_supported_fields(
      self, metadata: Mapping[str, AbstractSet[str]]) -> None:
    supported_metadata = set(self.supported_metadata)
    provided_metadata = set(metadata.keys())
    if not supported_metadata.issuperset(provided_metadata):
      raise MarkdownDocumentationError(
          "The MD file contains unsupported metadata properties: "
          f"{sorted(provided_metadata.difference(supported_metadata))}. Please "
          "refer to https://www.tensorflow.org/hub/writing_model_documentation for information about markdown format."
      )

  def _assert_no_duplicate_metadata(
      self, metadata: Mapping[str, AbstractSet[str]]) -> None:
    duplicate_metadata = list()
    for key, values in metadata.items():
      if key not in REPEATED_TAG_KEYS and len(values) > 1:
        duplicate_metadata.append(key)
    if duplicate_metadata:
      raise MarkdownDocumentationError(
          "There are duplicate metadata values. Please refer to "
          "https://www.tensorflow.org/hub/writing_model_documentation for "
          "information about markdown format. In particular the duplicated "
          f"metadata are: {sorted(duplicate_metadata)}")

  def _assert_correct_module_types(
      self, metadata: Mapping[str, AbstractSet[str]]) -> None:
    if "module-type" in metadata:
      allowed_prefixes = ["image-", "text-", "audio-", "video-"]
      for value in metadata["module-type"]:
        if all([not value.startswith(prefix) for prefix in allowed_prefixes]):
          raise MarkdownDocumentationError(
              "The 'module-type' metadata has to start with any of 'image-'"
              ", 'text', 'audio-', 'video-', but is: '{value}'")

  def _assert_correct_tag_values(
      self, metadata: Mapping[str, AbstractSet[str]],
      yaml_parser: yaml_parser_lib.YamlParser) -> None:
    """Checks that all tag values are defined in the respective YAML files.

    Args:
      metadata: Mapping of metadata fields to their values e.g.
                {"language": {"en", "fr"}}.
      yaml_parser: YamlParser containing all supported tag values.

    Raises:
      MarkdownDocumentationError: if a tag key contains elements in its set that
        are not defined in the respective YAML file.
      yaml.parser.ParserError: if the YAML file containing all supported values
        is no valid YAML file.
      FileNotFoundError: if the YAML file containing all supported values does
        not exist.
    """
    for tag_name, yaml_path in yaml_parser_lib.TAG_TO_YAML_MAP.items():
      if tag_name not in metadata:
        continue

      supported_values = yaml_parser.get_supported_values(tag_name)
      unsupported_values = metadata[tag_name] - supported_values
      if unsupported_values:
        raise MarkdownDocumentationError(
            f"Unsupported values for {tag_name} tag were found: "
            f"{sorted(unsupported_values)}. Please add them to "
            f"{yaml_parser_lib.TAG_TO_YAML_MAP[tag_name]}")

  def _assert_task_equals_module_type(
      self, metadata: Mapping[str, AbstractSet[str]]) -> None:
    """Asserts that if a task tag is given, it equals the module-type tag."""
    if "task" not in metadata:
      return

    actual_value = sorted(metadata["task"])
    expected_value = sorted(metadata.get("module-type", {}))
    if actual_value != expected_value:
      raise MarkdownDocumentationError(
          f"Expected 'task' tag to be {expected_value!r} but was "
          f"{actual_value!r}.")

  def assert_correct_metadata(self, metadata: Mapping[str, AbstractSet[str]],
                              yaml_parser: yaml_parser_lib.YamlParser) -> None:
    """Asserts that correct metadata is present."""
    self._assert_metadata_contains_required_fields(metadata)
    self._assert_metadata_contains_supported_fields(metadata)
    self._assert_no_duplicate_metadata(metadata)
    self._assert_correct_module_types(metadata)
    self._assert_correct_tag_values(metadata, yaml_parser)
    self._assert_task_equals_module_type(metadata)

  def assert_correct_asset_path(self, validation_config: ValidationConfig,
                                metadata: Mapping[str, AbstractSet[str]],
                                file_path: str) -> None:
    """Checks whether the given asset path can be downloaded.

    If an asset path is added or modified, the function checks whether the path
    has the correct file ending. If the path leads to github.com, it checks that
    the asset is not forbidden to be fetched by GitHub's robots.txt file. If
    `do_smoke_test` is True, it tries to download and parse the asset.

    Args:
      validation_config: The config specifying whether the referenced asset
        should be downloaded. That should only be used for validating individual
        files.
      metadata: Mapping of metadata fields to their values e.g.
        {"asset-path": {"model.tar.gz"}}
      file_path: Path to the validated file

    Raises:
      MarkdownDocumentationError:
        - if the asset-path key does not contain exactly one element in its set.
        - if the one element does not end in the expected suffix.
        - if github.com/robots.txt forbids downloading the asset.
        - if the asset can be downloaded but not be resolved to a SavedModel.
    """
    if ASSET_PATH_KEY not in metadata:
      return

    if not _is_asset_path_modified(file_path):
      logging.info("Skipping asset path validation since the tag is not added "
                   "or modified.")
      return

    if len(metadata[ASSET_PATH_KEY]) != 1:
      raise MarkdownDocumentationError(
          "No more than one asset-path tag may be specified.")

    asset_path = list(metadata[ASSET_PATH_KEY])[0]
    if not asset_path.endswith(self.supported_asset_path_suffix):
      raise MarkdownDocumentationError(
          f"Expected asset-path to end with {self.supported_asset_path_suffix} "
          f"but was {asset_path}.")

    # GitHub's robots.txt disallows fetches to */download, which means that
    # the asset-path URL cannot be fetched. Markdown validation should fail if
    # asset-path matches this regex.
    github_download_url_regex = re.compile(
        "https://github.com/.*/releases/download/.*")
    if github_download_url_regex.fullmatch(asset_path):
      raise MarkdownDocumentationError(
          f"The asset-path {asset_path} is a url that cannot be automatically "
          "fetched. Please provide an asset-path that is allowed to be fetched "
          "by its robots.txt.")

    if validation_config.do_smoke_test:
      self._assert_can_resolve_asset(asset_path)


class CollectionParsingPolicy(ParsingPolicy):
  """ParsingPolicy for collection documentation."""

  def __init__(self, publisher: str, model_name: str,
               model_version: str) -> None:
    super(CollectionParsingPolicy, self).__init__(
        publisher,
        model_name,
        model_version,
        required_metadata=["module-type"],
        optional_metadata=[
            "dataset", "language", "network-architecture", "task"
        ])

  @property
  def type_name(self) -> str:
    return "Collection"


class PlaceholderParsingPolicy(ParsingPolicy):
  """ParsingPolicy for placeholder files."""

  def __init__(self, publisher: str, model_name: str,
               model_version: str) -> None:
    super(PlaceholderParsingPolicy, self).__init__(
        publisher,
        model_name,
        model_version,
        required_metadata=["module-type"],
        optional_metadata=[
            "dataset", "fine-tunable", "interactive-model-name", "language",
            "license", "network-architecture", "task"
        ])

  @property
  def type_name(self) -> str:
    return "Placeholder"


class SavedModelParsingPolicy(ParsingPolicy):
  """ParsingPolicy for SavedModel documentation."""

  def __init__(self, publisher: str, model_name: str,
               model_version: str) -> None:
    super(SavedModelParsingPolicy, self).__init__(
        publisher,
        model_name,
        model_version,
        required_metadata=[
            "asset-path", "module-type", "fine-tunable", "format"
        ],
        optional_metadata=[
            "dataset", "interactive-model-name", "language", "license",
            "network-architecture", "task"
        ])

  @property
  def type_name(self) -> str:
    return "Module"

  @property
  def supported_asset_path_suffix(self) -> str:
    return TARFILE_SUFFIX

  def _assert_can_resolve_asset(self, asset_path: str) -> None:
    """Attempts to hub.resolve the given asset path."""
    try:
      resolved_model = hub.resolve(asset_path)
      loader_impl.parse_saved_model(resolved_model)
      _validate_file_paths(resolved_model)
    except Exception as e:  # pylint: disable=broad-except
      raise MarkdownDocumentationError(
          f"The model on path {asset_path} failed to parse. Please make sure "
          "that the asset-path metadata points to a valid TF2 SavedModel or a "
          "TF1 Hub module as described on "
          "https://www.tensorflow.org/hub/exporting_tf2_saved_model. "
          f"Underlying reason for failure: {e}.")

  def assert_correct_metadata(self, metadata: Mapping[str, AbstractSet[str]],
                              yaml_parser: yaml_parser_lib.YamlParser) -> None:
    super().assert_correct_metadata(metadata, yaml_parser)

    format_value = list(metadata.get("format", ""))[0]
    if format_value not in SAVED_MODEL_FORMATS:
      raise MarkdownDocumentationError(
          f"The 'format' metadata should be one of {SAVED_MODEL_FORMATS} "
          f"but was '{format_value}'.")


class TfjsParsingPolicy(ParsingPolicy):
  """ParsingPolicy for TF.js documentation."""

  def __init__(self, publisher: str, model_name: str,
               model_version: str) -> None:
    super(TfjsParsingPolicy, self).__init__(
        publisher,
        model_name,
        model_version,
        required_metadata=["asset-path", "parent-model"],
        optional_metadata=["interactive-model-name"])

  @property
  def type_name(self) -> str:
    return "Tfjs"

  @property
  def supported_asset_path_suffix(self) -> str:
    return TARFILE_SUFFIX


class LiteParsingPolicy(TfjsParsingPolicy):
  """ParsingPolicy for TFLite documentation."""

  @property
  def type_name(self) -> str:
    return "Lite"

  @property
  def supported_asset_path_suffix(self) -> str:
    return TFLITE_SUFFIX


class CoralParsingPolicy(LiteParsingPolicy):
  """ParsingPolicy for Coral documentation."""

  @property
  def type_name(self) -> str:
    return "Coral"


class PublisherParsingPolicy(ParsingPolicy):
  """ParsingPolicy for publisher documentation.

  Publisher files should always be at root/publisher/publisher.md and they
  should not contain a 'format' tag as it has no effect.
  """

  def __init__(self,
               publisher: str,
               model_name: str = "",
               model_version: str = "") -> None:
    super(PublisherParsingPolicy, self).__init__(publisher, model_name,
                                                 model_version, [], [])

  @property
  def type_name(self) -> str:
    return "Publisher"

  def get_expected_file_path(self, root_dir: str) -> str:
    """Returns the expected path of the documentation file."""
    return os.path.join(root_dir, self._publisher, self._publisher + ".md")

  def assert_correct_file_path(self, file_path: str, root_dir: str) -> None:
    """Extend base method by also checking for /publisher/publisher.md."""
    expected_file_path = self.get_expected_file_path(root_dir)
    if expected_file_path and file_path != expected_file_path:
      raise MarkdownDocumentationError(
          "Documentation file is not on a correct path. Documentation for the "
          f"publisher '{self.publisher}' should be submitted to "
          f"'{expected_file_path}'")
    super().assert_correct_file_path(file_path, root_dir)


class DocumentationParser:
  """Class used for parsing model documentation strings."""

  def __init__(self, root_dir: str, documentation_dir: str) -> None:
    self._root_dir = root_dir
    self._documentation_dir = documentation_dir
    self._parsed_metadata = dict()
    self._parsed_description = ""
    self._file_path = ""
    self._lines = []
    self._current_index = 0
    self.policy = None

  @property
  def parsed_description(self) -> str:
    return self._parsed_description

  @property
  def parsed_metadata(self) -> str:
    return self._parsed_metadata

  def _raise_error(self, message: str) -> None:
    raise MarkdownDocumentationError(message)

  def _get_policy_from_first_line(self, first_line: str) -> ParsingPolicy:
    """Returns an appropriate ParsingPolicy instance for the first line."""
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
    # pytype: disable=bad-return-type
    self._raise_error(
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
    # pytype: enable=bad-return-type

  def _assert_publisher_page_exists(self) -> None:
    """Asserts that publisher page exists for the publisher of this model."""
    # Use a publisher policy to get the expected documentation page path.
    publisher_policy = PublisherParsingPolicy(self.policy.publisher)
    expected_publisher_doc_file_path = publisher_policy.get_expected_file_path(
        self._documentation_dir)
    if not tf.io.gfile.exists(expected_publisher_doc_file_path):
      self._raise_error(
          "Publisher documentation does not exist. "
          f"It should be added to {expected_publisher_doc_file_path}.")

  def _consume_description(self) -> None:
    """Consumes second line with a short model description."""
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

  def _consume_metadata(self) -> None:
    """Consumes all metadata."""
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
        key = groups.get("key")
        value = groups.get("value")
        if key is None or value is None:
          raise MarkdownDocumentationError(
              f"(key, value) must not be None but got ({key}, {value}).")
        key = key.strip()
        value = value.strip()
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
      if self._lines[self._current_index].startswith("[![Open Demo]]"):
        # Demo button.
        self._current_index += 1
        continue
      # Not an empty line and not expected metadata.
      raise MarkdownDocumentationError(
          f"Unexpected line found: '{self._lines[self._current_index]}'. "
          "Please refer to "
          "https://www.tensorflow.org/hub/writing_model_documentation for "
          "information about markdown format.")

  def _assert_allowed_license(self) -> None:
    """Validates provided license."""
    if "license" in self._parsed_metadata:
      license_id = list(self._parsed_metadata["license"])[0]
      allowed_license_ids = [
          "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause", "CC-BY-NC-4.0",
          "GPL-2.0", "GPL-3.0", "LGPL-2.0", "LGPL-2.1", "LGPL-3.0", "MIT",
          "MPL-2.0", "CDDL-1.0", "EPL-2.0", "custom"
      ]
      if license_id not in allowed_license_ids:
        self._raise_error(
            f"The license {license_id} provided in metadata is not allowed. "
            "Please specify a license id from list of allowed ids: "
            f"[{allowed_license_ids}]. Example: <!-- license: Apache-2.0 -->")

  def validate(self, validation_config: ValidationConfig, file_path: str,
               yaml_parser: yaml_parser_lib.YamlParser) -> None:
    """Validate one documentation markdown file."""
    self._file_path = file_path
    raw_content = filesystem_utils.get_content(self._file_path)
    self._lines = raw_content.split("\n")
    first_line = self._lines[0].replace("&zwnj;", "")
    self.policy = self._get_policy_from_first_line(first_line)

    try:
      self.policy.assert_correct_file_path(self._file_path,
                                           self._documentation_dir)
      # Populate _parsed_description with the description
      self._consume_description()
      # Populate _parsed_metadata with the metadata tag mapping
      self._consume_metadata()
      self.policy.assert_correct_metadata(self._parsed_metadata, yaml_parser)
      if not validation_config.skip_asset_check:
        self.policy.assert_correct_asset_path(validation_config,
                                              self._parsed_metadata,
                                              self._file_path)
    except MarkdownDocumentationError as e:
      self._raise_error(str(e))
    self._assert_allowed_license()
    self._assert_publisher_page_exists()


def validate_documentation_dir(validation_config: ValidationConfig,
                               root_dir: str,
                               relative_docs_path: str = DOCS_PATH) -> None:
  """Validate Markdown files in `root_dir/relative_docs_path`.

  Args:
    validation_config: TestConfig specifying whether the "asset-path" tag should
      be validated.
    root_dir: Absolute path to the top-level dir that contains Markdown files
      and YAML config files.
    relative_docs_path: Relative path under `root_dir` containing the Markdown
      files. Defaults to "assets/docs".

  Raises:
    MarkdownDocumentationError: if invalid Markdown files have been found.
  """
  documentation_dir = os.path.join(root_dir, relative_docs_path)
  logging.info("Validating all files in %s.", documentation_dir)
  relative_paths = [
      os.path.relpath(file_path, documentation_dir)
      for file_path in filesystem_utils.recursive_list_dir(documentation_dir)
  ]
  validate_documentation_files(
      validation_config,
      root_dir,
      relative_paths,
      relative_docs_path=relative_docs_path)


def validate_documentation_files(validation_config: ValidationConfig,
                                 root_dir: str,
                                 files_to_validate: MutableSequence[str],
                                 relative_docs_path: str = DOCS_PATH) -> None:
  """Validate specified Markdown documentation files.

  Args:
    validation_config: TestConfig specifying whether the "asset-path" tag should
      be validated and the remote path be downloaded.
    root_dir: Absolute path to the top-level dir that contains Markdown files
      and YAML config files.
    files_to_validate: List of file paths in `root_dir` that should be
      validated.
    relative_docs_path: Relative path under `root_dir` containing the Markdown
      files. Defaults to "assets/docs".

  Raises:
    MarkdownDocumentationError: if invalid Markdown files have been found.
  """
  documentation_dir = os.path.join(root_dir, relative_docs_path)
  yaml_parser = yaml_parser_lib.YamlParser(root_dir)
  logging.info("Going to validate files %s in documentation directory %s.",
               files_to_validate, documentation_dir)
  validated = 0
  file_to_error = dict()

  for file_path in files_to_validate:
    logging.info("Validating %s.", file_path)
    documentation_parser = DocumentationParser(root_dir, documentation_dir)
    try:
      absolute_path = os.path.join(documentation_dir, file_path)
      documentation_parser.validate(validation_config, absolute_path,
                                    yaml_parser)
      validated += 1
    except MarkdownDocumentationError as e:
      file_to_error[file_path] = str(e)
  if not validation_config.do_smoke_test:
    logging.info(
        "No models were smoke tested. To download and smoke test a specific "
        "model, specify files directly in the command line, for example: "
        "'python tools/validator.py vtab/models/wae-ukl/1.md'")
  if file_to_error:
    raise MarkdownDocumentationError(
        f"Found the following errors: {file_to_error}")
  logging.info("Found %d matching files - all validated successfully.",
               validated)


def main(_):
  root_dir = FLAGS.root_dir or os.getcwd()

  if FLAGS.file:
    validate_documentation_files(
        ValidationConfig(skip_asset_check=False, do_smoke_test=True), root_dir,
        FLAGS.file)
  else:
    validate_documentation_dir(
        ValidationConfig(skip_asset_check=False, do_smoke_test=False), root_dir)


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
