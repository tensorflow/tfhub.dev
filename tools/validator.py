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
import fnmatch
import itertools
import os
import re
import subprocess
import sys
import tarfile
import time
import textwrap
from typing import AbstractSet, Iterator, Mapping, MutableSequence, Optional, Sequence, Type, TypeVar
import urllib.request

from absl import app
from absl import logging
import attr
import tensorflow as tf
import filesystem_utils
import yaml_parser as yaml_parser_lib

from google.protobuf import message
from google.protobuf import text_format
# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.protobuf import saved_model_pb2
# pylint: enable=g-direct-tensorflow-import

FLAGS = None

_CI_ENV_KEY = "GITHUB_ACTION"
_SLEEP_SECONDS = 5

# Relative path from tfhub.dev/ to the docs/ directory.
DOCS_PATH = "assets/docs"
# Path parts to locations where model/collection documentation is stored.
_COLLECTIONS_DIR = "collections"
_COLLECTIONS_FILE_NAME = "1.md"  # Collections should only be version 1.
_MODELS_DIR = "models"
_TFJS_DIR = "tfjs"
_LITE_DIR = "lite"
_CORAL_DIR = "coral"

_PUBLISHER_ID_PATTERN = r"[a-z\d-]+"  # Publisher name like "google".
_MODEL_NAME_PATTERN = r"[\w.-]+(/[\w.-]+)*"  # Model name like "BERT/uncased".
_MODEL_VERSION_PATTERN = r"\d+"  # Integer specifying the version like 1.

# Regex pattern for the first line of the documentation of Saved Models.
# Example: "Module google/universal-sentence-encoder/1"
MODEL_HANDLE_PATTERN = (
    "# Module "
    f"(?P<publisher>{_PUBLISHER_ID_PATTERN})/"
    f"(?P<name>{_MODEL_NAME_PATTERN})/"
    f"(?P<vers>{_MODEL_VERSION_PATTERN})")
# Regex pattern for the first line of the documentation of placeholder MD files.
# Example: "Placeholder google/universal-sentence-encoder/1"
PLACEHOLDER_HANDLE_PATTERN = (
    "# Placeholder "
    f"(?P<publisher>{_PUBLISHER_ID_PATTERN})/"
    f"(?P<name>{_MODEL_NAME_PATTERN})/"
    f"(?P<vers>{_MODEL_VERSION_PATTERN})")
# Regex pattern for the first line of the documentation of TF Lite models.
# Example: "# Lite google/spice/1"
LITE_HANDLE_PATTERN = (
    "# Lite "
    f"(?P<publisher>{_PUBLISHER_ID_PATTERN})/"
    f"(?P<name>{_MODEL_NAME_PATTERN})/"
    f"(?P<vers>{_MODEL_VERSION_PATTERN})")
# Regex pattern for the first line of the documentation of TFJS models.
# Example: "# Tfjs google/spice/1/default/1"
TFJS_HANDLE_PATTERN = (
    "# Tfjs "
    f"(?P<publisher>{_PUBLISHER_ID_PATTERN})/"
    f"(?P<name>{_MODEL_NAME_PATTERN})/"
    f"(?P<vers>{_MODEL_VERSION_PATTERN})")
# Regex pattern for the first line of the documentation of Coral models.
# Example: "# Coral tensorflow/mobilenet_v2_1.0_224_quantized/1/default/1"
CORAL_HANDLE_PATTERN = (
    "# Coral "
    f"(?P<publisher>{_PUBLISHER_ID_PATTERN})/"
    f"(?P<name>{_MODEL_NAME_PATTERN})/"
    f"(?P<vers>{_MODEL_VERSION_PATTERN})")
# Regex pattern for the first line of the documentation of publishers.
# Example: "Publisher google"
PUBLISHER_HANDLE_PATTERN = (
    f"# Publisher (?P<publisher>{_PUBLISHER_ID_PATTERN})")
# Regex pattern for the first line of the documentation of collections.
# Example: "Collection google/universal-sentence-encoders/1"
COLLECTION_HANDLE_PATTERN = (
    "# Collection "
    f"(?P<publisher>{_PUBLISHER_ID_PATTERN})/"
    r"(?P<name>(\w|-|/|&|;|\.)+)/"  # Collection name like "wiki40b-lm".
    f"({_MODEL_VERSION_PATTERN})")
# Regex pattern for URLs to model pages on tfhub.dev.
# Examples:
# https://tfhub.dev/google/bert
# https://tfhub.dev/google/bert/4
# https://tfhub.dev/google/lite-model/yamnet/tflite/1
_TFHUB_MODEL_URL_PATTERN = re.compile(
    "https://tfhub.dev/"
    f"(?P<publisher>{_PUBLISHER_ID_PATTERN})/"  # Publisher name.
    "(?P<prefix>(tfjs|lite|coral)-model/)?"  # Model-type specific prefix.
    f"(?P<name>{_MODEL_NAME_PATTERN})"  # Model name including version if set.
)
# Regex pattern for the line of the documentation describing model metadata.
# Example: "<!-- fine-tunable: true -->"
# Note: Both key and value consumes free space characters, but later on these
# are stripped.
METADATA_LINE_PATTERN = r"^<!--(?P<key>(\w|\s|-)+):(?P<value>.+)-->$"

# Regex pattern for files in the SavedModel directory.
# Example: "variables/variables.data-00000-of-00001"
_FILE_NAME_PATTERN = r"[\w-][-!',_\w.=:% ]*"
PATH_PATTERN = re.compile(f"({_FILE_NAME_PATTERN})+(/{_FILE_NAME_PATTERN})*")

# Dict keys that map to the specified metadata values of the Markdown files.
ARCHITECTURE_KEY = "network-architecture"
ASSET_PATH_KEY = "asset-path"
COLAB_KEY = "colab"
DATASET_KEY = "dataset"
DEMO_KEY = "demo"
FINE_TUNABLE_KEY = "fine-tunable"
FORMAT_KEY = "format"
LANGUAGE_KEY = "language"
LICENSE_KEY = "license"
PARENT_MODEL_KEY = "parent-model"
TASK_KEY = "task"
VISUALIZER_KEY = "interactive-visualizer"

# These metadata tags can be set to more than one value.
REPEATED_TAG_KEYS = (DATASET_KEY, LANGUAGE_KEY, TASK_KEY, ARCHITECTURE_KEY)

# Specifies whether a SavedModel is a Hub Module or a TF1/TF2 SavedModel.
SAVED_MODEL_FORMATS = ("hub", "saved_model", "saved_model_2")

TARFILE_SUFFIX = ".tar.gz"
TFLITE_SUFFIX = ".tflite"


# Allowed SavedModel files:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md  # pylint: disable=line-too-long
ALLOWED_SAVED_MODEL_PATHS = frozenset([
    "saved_model.pb", "saved_model.pbtxt", "tfhub_module.pb",
    "keras_metadata.pb", "assets/*", "assets.extra/*", "variables/variables.*"
])

ParsingPolicyType = TypeVar("ParsingPolicyType", bound="ParsingPolicy")


class MarkdownDocumentationError(Exception):
  """Problem with markdown syntax parsing."""


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


def _should_sleep() -> bool:
  return os.environ.get(_CI_ENV_KEY, None) is not None


def _check_that_saved_model_pb_parses(tar: tarfile.TarFile,
                                      tar_member: tarfile.TarInfo) -> None:
  """Tries to load a saved_model.pb from the tarfile into a SavedModel proto.

  Args:
    tar: TarFile containing the possible SavedModel assets.
    tar_member: TarInfo that corresponds to the file being saved in the tarfile.

  Raises:
    MarkdownDocumentationError:
      - if saved_model.pb is no regular file in the tarfile.
      - if saved_model.pb cannot be loaded into a SavedModel proto.
  """
  data = tar.extractfile(tar_member)
  if data is None:
    raise MarkdownDocumentationError("saved_model.pb is no regular file.")
  try:
    saved_model_pb2.SavedModel.FromString(data.read())
  except message.DecodeError as e:
    raise MarkdownDocumentationError("Could not parse saved_model.pb.") from e


def _check_that_saved_model_pbtxt_parses(tar: tarfile.TarFile,
                                         tar_member: tarfile.TarInfo) -> None:
  """Tries to load a saved_model.pbtxt from the tarfile into a SavedModel proto.

  Args:
    tar: TarFile containing the possible SavedModel assets.
    tar_member: TarInfo that corresponds to the file being saved in the tarfile.

  Raises:
    MarkdownDocumentationError:
      - if saved_model.pbtxt is no regular file in the tarfile.
      - if saved_model.pbtxt cannot be loaded into a SavedModel proto.
  """
  data = tar.extractfile(tar_member)
  if data is None:
    raise MarkdownDocumentationError("saved_model.pbtxt is no regular file.")
  try:
    text_format.Parse(data.read().decode("utf-8"), saved_model_pb2.SavedModel())
  except text_format.ParseError as e:
    raise MarkdownDocumentationError(
        "Could not parse saved_model.pbtxt.") from e


def _validate_file_name(file_path: str) -> None:
  """Checks that the file name is allowed.

  Args:
    file_path: Relative path to a file.

  Raises:
    MarkdownDocumentationError:
      - if the path is invalid since it e.g. starts with a dot.
      - if the path is forbidden since it cannot be used by the SavedModel e.g.
        when "vocab.txt" is located in "variables/" and not in "assets/".
  """
  if not PATH_PATTERN.fullmatch(file_path):
    raise MarkdownDocumentationError(f"Invalid filepath in asset: {file_path}")
  if not any(
      fnmatch.fnmatch(file_path, pattern)
      for pattern in ALLOWED_SAVED_MODEL_PATHS):
    raise MarkdownDocumentationError(
        f"File cannot be used by SavedModel: {file_path}")


@attr.s(auto_attribs=True)
class ValidationConfig:
  """A simple value class containing information for the validation process.

  Attributes:
    skip_file_path_check: A boolean indicating whether the check should be
      skipped that ensures that files are only stored at their allowed paths.
      Defaults to False.
    skip_asset_check: A boolean indicating whether the "asset-path" tag should
      be skipped for validation. Defaults to False.
    skip_content_check: A boolean indicating whether the content that is
      rendered on the model detail page should be skipped for validation.
      Defaults to False.
    do_smoke_test: A boolean indicating whether the referenced asset should be
      downloaded. Defaults to False.
  """
  skip_file_path_check: bool = False
  skip_asset_check: bool = False
  skip_content_check: bool = False
  do_smoke_test: bool = False


class ParsingPolicy(metaclass=abc.ABCMeta):
  """The base class for type specific parsing policies.

  Documentation files for models, placeholders, publishers and collections share
  a publisher field, a readable name, a correct file path etc.
  """

  def __init__(self,
               yaml_parser_by_tag_name: Mapping[
                   str, yaml_parser_lib.AbstractYamlParser],
               publisher: str,
               model_name: str,
               model_version: str,
               required_metadata: AbstractSet[str],
               optional_metadata: AbstractSet[str],
               supported_asset_path_suffix: Optional[str] = None) -> None:
    """Initializes a ParsingPolicy instance for validating a document.

    Args:
      yaml_parser_by_tag_name: Mapping from a tag name to its
        yaml_parser_lib.YamlParser instance to be used for accessing allowed
        metadata values from the YAML config files.
      publisher: The name of the publisher of the model e.g. 'Google'.
      model_name: The name of the model e.g. 'ALBERT'.
      model_version: The version of the model e.g. '1'.
      required_metadata: Set of strings containing the required tag names that
        need to be set in the Markdown document e.g. {"asset-path"}.
      optional_metadata: Set of strings containing the optional tag names that
        can be set in the Markdown document e.g. {"dataset"}.
      supported_asset_path_suffix: Optional; The file ending of the 'asset-path'
        tag, if that tag needs to be set.
    """
    self._yaml_parser_by_tag_name = yaml_parser_by_tag_name
    self._publisher = publisher
    self._model_name = model_name
    self._model_version = model_version
    self._required_metadata = required_metadata
    self._optional_metadata = optional_metadata
    self._supported_asset_path_suffix = supported_asset_path_suffix

  @classmethod
  def from_string(
      cls: Type[ParsingPolicyType], first_line: str,
      yaml_parser_by_tag_name: Mapping[str, yaml_parser_lib.AbstractYamlParser]
  ) -> ParsingPolicyType:
    """Returns an appropriate ParsingPolicy instance for the Markdown string."""
    for pattern, policy in POLICY_BY_PATTERN.items():
      match = re.fullmatch(pattern, first_line)
      if not match:
        continue
      groups = match.groupdict()
      return policy(yaml_parser_by_tag_name, groups.get("publisher"),
                    groups.get("name"), groups.get("vers"))
    raise MarkdownDocumentationError(
        textwrap.dedent(f"""\
      First line of the documentation file must match one of the following
      formats depending on the MD type:
      TF Model: {MODEL_HANDLE_PATTERN}
      TFJS: {TFJS_HANDLE_PATTERN}
      Lite: {LITE_HANDLE_PATTERN}
      Coral: {CORAL_HANDLE_PATTERN}
      Publisher: {PUBLISHER_HANDLE_PATTERN}
      Collection: {COLLECTION_HANDLE_PATTERN}
      Placeholder: {PLACEHOLDER_HANDLE_PATTERN}
      For example '# Module google/text-embedding-model/1'.
      Instead the first line is '{first_line}'"""))

  @property
  @abc.abstractmethod
  def type_name(self) -> str:
    """A readable name for the parsed type."""
    raise NotImplementedError

  @property
  def publisher(self) -> str:
    return self._publisher

  @property
  def id(self) -> str:
    return f"{self.publisher}/{self._model_name}/{self._model_version}"

  @property
  def supported_metadata(self) -> AbstractSet[str]:
    """Return which metadata tags are supported."""
    return set.union(self._required_metadata, self._optional_metadata)

  def get_allowed_file_paths(self, documentation_dir: str) -> Sequence[str]:
    """Returns the paths at which the documentation can be stored.

    Args:
      documentation_dir: Absolute path to the `assets/docs` dir.

    Returns:
      Sequence of paths that can contain the documentation files for the model.
      Paths can contain wildcards.
    """
    # TODO(b/198250794): Migrate to having only exactly one allowed path.
    raise NotImplementedError

  def assert_correct_file_path(self, file_path: str,
                               documentation_dir: str) -> None:
    """Checks that the file is stored at an allowed location.

    Model documentation should be stored at
    PUBLISHER/(models/)NAME/(tfjs/|lite/|coral/)VERSION.md, collections at
    PUBLISHER/collections/NAME/1.md and publisher documentation files at
    PUBLISHER/PUBLISHER.md.

    Args:
      file_path: Relative path to the file from the `assets/docs` directory.
      documentation_dir: Absolute path to the `assets/docs` dir.

    Raises:
      MarkdownDocumentationError:
        - if the file is not stored at an allowed location.
    """
    maybe_wildcard_paths = self.get_allowed_file_paths(documentation_dir)
    absolute_paths = map(tf.io.gfile.glob, maybe_wildcard_paths)
    flat_absolute_paths = list(itertools.chain.from_iterable(absolute_paths))
    actual_path = os.path.join(documentation_dir, file_path)
    if actual_path not in flat_absolute_paths:
      raise MarkdownDocumentationError(
          f"Expected {self.id} to have documentation stored in one of "
          f"{maybe_wildcard_paths} but was {actual_path}.")

  def validate_asset_path(self, validation_config: ValidationConfig,
                          metadata: Mapping[str, AbstractSet[str]],
                          file_path: str) -> None:
    """Checks whether the 'asset-path' tag is valid, if it is set.

    Not all documentation files set the 'asset-path' tag but if they do, the tag
    should be checked for being valid. This method does not validate the tag
    while ModelParsingPolicy overwrites this method since model files need to
    set the 'asset-path' tag. In that case, its value needs to have the correct
    file ending, must not be forbidden by GitHubs robots.txt file etc.

    Args:
      validation_config: The config specifying whether the referenced asset
        should be downloaded. That should only be used for validating individual
        files.
      metadata: Mapping of metadata fields to their values e.g.
        {"asset-path": {"model.tar.gz"}}
      file_path: Path to the validated file
    """
    del validation_config, metadata, file_path
    logging.info("Skipping validating 'asset-path' tag since the tag is not "
                 "supported.")

  def _check_valid_remote_asset(self, asset_path: str) -> None:
    """Checks whether the remote asset is valid."""

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
    if TASK_KEY in metadata:
      allowed_prefixes = ["image-", "text-", "audio-", "video-"]
      for value in metadata[TASK_KEY]:
        if not value.startswith(tuple(allowed_prefixes)):
          raise MarkdownDocumentationError(
              "The 'task' metadata has to start with any of 'image-'"
              f", 'text', 'audio-', 'video-', but is: '{value}'")

  def _assert_correct_tag_values(
      self, metadata: Mapping[str, AbstractSet[str]]) -> None:
    """Checks that all tag values are defined in the respective YAML files.

    Args:
      metadata: Mapping of metadata fields to their values e.g.
        {"language": {"en", "fr"}}.

    Raises:
      FileNotFoundError: if a YAML file containing the parser config does not
        exist.
      yaml.parser.ParserError: if a YAML file containing the parser config is no
        valid YAML file.
      MarkdownDocumentationError: if a tag value from `metadata` is invalid as
        determined by its respective parser from `yaml_parser_by_tag_name`.
    """
    for tag_name in metadata:
      if tag_name not in self._yaml_parser_by_tag_name:
        # Not every tag has a parser associated with it.
        continue

      yaml_parser = self._yaml_parser_by_tag_name[tag_name]
      try:
        yaml_parser.assert_tag_values_are_correct(metadata[tag_name])
      except ValueError as e:
        raise MarkdownDocumentationError(
            f"Validating {tag_name} failed: {e}") from e

  def assert_correct_metadata(self,
                              metadata: Mapping[str, AbstractSet[str]]) -> None:
    """Asserts that correct metadata is present."""
    self._assert_metadata_contains_required_fields(metadata)
    self._assert_metadata_contains_supported_fields(metadata)
    self._assert_no_duplicate_metadata(metadata)
    self._assert_correct_module_types(metadata)
    self._assert_correct_tag_values(metadata)

  def assert_correct_content(self, documentation_dir: str,
                             lines: Sequence[str]) -> None:
    """Ensures that the content to be displayed is correct."""


class ModelParsingPolicy(ParsingPolicy):
  """Base class for additionally validating the 'asset-path' tag for models.

  In constrast to Markdown files for collections, publishers, and placeholders,
  SavedModel/Tfjs/Lite/Coral documentation needs to specify an 'asset-path' tag,
  which should point to a world-readable location and end in a model-specific
  suffix like '.tar.gz' or '.tflite'.
  """

  def __init__(self, yaml_parser_by_tag_name: Mapping[
      str, yaml_parser_lib.AbstractYamlParser], publisher: str, model_name: str,
               model_version: str, required_metadata: AbstractSet[str],
               optional_metadata: AbstractSet[str],
               supported_asset_path_suffix: str) -> None:
    super().__init__(
        yaml_parser_by_tag_name,
        publisher,
        model_name,
        model_version,
        required_metadata,
        optional_metadata,
        supported_asset_path_suffix=supported_asset_path_suffix)

  def validate_asset_path(self, validation_config: ValidationConfig,
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
    if not _is_asset_path_modified(file_path):
      logging.info("Skipping asset path validation since the tag is not added "
                   "or modified.")
      return

    if len(metadata[ASSET_PATH_KEY]) != 1:
      raise MarkdownDocumentationError(
          "No more than one asset-path tag may be specified.")

    asset_path = list(metadata[ASSET_PATH_KEY])[0]
    if not asset_path.endswith(self._supported_asset_path_suffix):
      raise MarkdownDocumentationError(
          f"Expected asset-path to end with {self._supported_asset_path_suffix}"
          f" but was {asset_path}.")

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
      self._check_valid_remote_asset(asset_path)


class CollectionParsingPolicy(ParsingPolicy):
  """ParsingPolicy for collection documentation."""

  def __init__(self, yaml_parser_by_tag_name: Mapping[
      str, yaml_parser_lib.AbstractYamlParser], publisher: str, model_name: str,
               model_version: str) -> None:
    super(CollectionParsingPolicy, self).__init__(
        yaml_parser_by_tag_name,
        publisher,
        model_name,
        model_version,
        required_metadata={TASK_KEY},
        optional_metadata={DATASET_KEY, LANGUAGE_KEY, ARCHITECTURE_KEY})

  @property
  def type_name(self) -> str:
    return "Collection"

  @property
  def id(self) -> str:
    return f"{self.publisher}/{self._model_name}"

  def get_allowed_file_paths(self, documentation_dir: str) -> Sequence[str]:
    """Returns the absolute paths for PUBLISHER/collections/NAME/1.md."""
    return [
        os.path.join(documentation_dir, self._publisher, _COLLECTIONS_DIR,
                     self._model_name, _COLLECTIONS_FILE_NAME),
    ]

  def assert_correct_content(self, documentation_dir: str,
                             lines: Sequence[str]) -> None:
    """Checks that referenced model pages have existing documentation files.

    A collection should contain at least one link of the form
    https://tfhub.dev/PUBLISHER/((tfjs|lite|coral)-model/)NAME(/VERSION).
    For each URL, the corresponding model should have a documentation file.

    Args:
      documentation_dir: Absolute path to the `assets/docs` directory.
      lines: Sequence of strings representing the file content.

    Raises:
      MarkdownDocumentationError:
        - if no model URL is contained in the collection.
        - if a model URL is contained whose model does not have a documentation.
    """
    super().assert_correct_content(documentation_dir, lines)

    found_one_model = False
    for line in lines:
      for policy in _get_policies_for_line_with_model_urls(line):
        found_one_model = True
        maybe_wildcard_paths = policy.get_allowed_file_paths(documentation_dir)
        absolute_paths = map(tf.io.gfile.glob, maybe_wildcard_paths)
        for absolute_path in itertools.chain.from_iterable(absolute_paths):
          if tf.io.gfile.exists(absolute_path):
            break
        else:
          raise MarkdownDocumentationError("No documentation file found in "
                                           f"{maybe_wildcard_paths}.")

    if not found_one_model:
      raise MarkdownDocumentationError(
          "A collection needs to contain at least one model URL.")


class PlaceholderParsingPolicy(ParsingPolicy):
  """ParsingPolicy for placeholder files."""

  def __init__(self, yaml_parser_by_tag_name: Mapping[
      str, yaml_parser_lib.AbstractYamlParser], publisher: str, model_name: str,
               model_version: str) -> None:
    super(PlaceholderParsingPolicy, self).__init__(
        yaml_parser_by_tag_name,
        publisher,
        model_name,
        model_version,
        required_metadata={TASK_KEY},
        optional_metadata={
            DATASET_KEY, FINE_TUNABLE_KEY,
            VISUALIZER_KEY, LANGUAGE_KEY, LICENSE_KEY, ARCHITECTURE_KEY
        })

  @property
  def type_name(self) -> str:
    return "Placeholder"

  def get_allowed_file_paths(self, documentation_dir: str) -> Sequence[str]:
    """Returns the absolute paths for PUBLISHER/(models/)NAME/VERSION.md."""
    return [
        os.path.join(documentation_dir, self._publisher, self._model_name,
                     f"{self._model_version}.md"),
        os.path.join(documentation_dir, self._publisher, _MODELS_DIR,
                     self._model_name, f"{self._model_version}.md")
    ]


class SavedModelParsingPolicy(ModelParsingPolicy):
  """ParsingPolicy for SavedModel documentation."""

  def __init__(self, yaml_parser_by_tag_name: Mapping[
      str, yaml_parser_lib.AbstractYamlParser], publisher: str, model_name: str,
               model_version: str) -> None:
    super(SavedModelParsingPolicy, self).__init__(
        yaml_parser_by_tag_name,
        publisher,
        model_name,
        model_version,
        required_metadata={
            ASSET_PATH_KEY, FINE_TUNABLE_KEY, FORMAT_KEY, TASK_KEY
        },
        optional_metadata={
            ARCHITECTURE_KEY, COLAB_KEY, DATASET_KEY, LANGUAGE_KEY, LICENSE_KEY,
            VISUALIZER_KEY
        },
        supported_asset_path_suffix=TARFILE_SUFFIX)

  @property
  def type_name(self) -> str:
    return "Module"

  def get_allowed_file_paths(self, documentation_dir: str) -> Sequence[str]:
    """Returns the absolute paths for PUBLISHER/(models/)NAME/VERSION.md."""
    return [
        os.path.join(documentation_dir, self._publisher, self._model_name,
                     f"{self._model_version}.md"),
        os.path.join(documentation_dir, self._publisher, _MODELS_DIR,
                     self._model_name, f"{self._model_version}.md")
    ]

  def _check_valid_remote_asset(self, remote_archive: str) -> None:
    """Checks whether the remote archive contains valid SavedModel files.

    Args:
      remote_archive: URL pointing to the remote archive.

    Raises:
      MarkdownDocumentationError:
        - if the remote file is no valid tarfile.
        - if a file name within the tarfile is invalid since it e.g. starts with
          a dot.
        - if no saved_model.pb(txt) file is contained in the archive.
        - if the contained saved_model.pb(txt) file cannot be loaded into a
          SavedModel proto.
    """
    valid_saved_model_proto_found = False
    # Wait before each check to prevent exhausting storage read quota.
    if _should_sleep():
      time.sleep(_SLEEP_SECONDS)
    with urllib.request.urlopen(remote_archive) as url_contents:
      try:
        with tarfile.open(fileobj=url_contents, mode="r|gz") as tar:
          for tar_member in tar:
            if not tar_member.isfile():
              continue
            normalized_path = os.path.normpath(tar_member.name)  # Strip './'.
            normalized_name = os.path.basename(normalized_path)
            _validate_file_name(normalized_path)
            if normalized_name == tf.saved_model.SAVED_MODEL_FILENAME_PB:
              _check_that_saved_model_pb_parses(tar, tar_member)
              valid_saved_model_proto_found = True
            elif normalized_name == tf.saved_model.SAVED_MODEL_FILENAME_PBTXT:
              _check_that_saved_model_pbtxt_parses(tar, tar_member)
              valid_saved_model_proto_found = True
      except tarfile.ReadError as e:
        raise MarkdownDocumentationError(f"Could not read tarfile: {e}") from e
      if not valid_saved_model_proto_found:
        raise MarkdownDocumentationError(
            f"The model from {remote_archive} does not contain a valid "
            "saved_model.pb or saved_model.pbtxt file. Please make sure that "
            "the asset-path metadata points to a valid TF2 SavedModel or a TF1 "
            "Hub module as described on "
            "https://www.tensorflow.org/hub/exporting_tf2_saved_model.")

  def assert_correct_metadata(self,
                              metadata: Mapping[str, AbstractSet[str]]) -> None:
    super().assert_correct_metadata(metadata)

    format_value = list(metadata.get(FORMAT_KEY, ""))[0]
    if format_value not in SAVED_MODEL_FORMATS:
      raise MarkdownDocumentationError(
          f"The 'format' metadata should be one of {SAVED_MODEL_FORMATS} "
          f"but was '{format_value}'.")


class TfjsParsingPolicy(ModelParsingPolicy):
  """ParsingPolicy for TF.js documentation."""

  def __init__(self, yaml_parser_by_tag_name: Mapping[
      str, yaml_parser_lib.AbstractYamlParser], publisher: str, model_name: str,
               model_version: str) -> None:
    super(TfjsParsingPolicy, self).__init__(
        yaml_parser_by_tag_name,
        publisher,
        model_name,
        model_version,
        required_metadata={ASSET_PATH_KEY, PARENT_MODEL_KEY},
        optional_metadata={
            COLAB_KEY, DEMO_KEY, VISUALIZER_KEY
        },
        supported_asset_path_suffix=TARFILE_SUFFIX)

  @property
  def type_name(self) -> str:
    return "Tfjs"

  def get_allowed_file_paths(self, documentation_dir: str) -> Sequence[str]:
    """Returns the paths for PUBLISHER/(models/)MODEL/tfjs/VERSION.md."""
    return [
        os.path.join(documentation_dir, self._publisher, self._model_name,
                     _TFJS_DIR, f"{self._model_version}.md"),
        os.path.join(documentation_dir, self._publisher, _MODELS_DIR,
                     self._model_name, _TFJS_DIR, f"{self._model_version}.md")
    ]


class LiteParsingPolicy(ModelParsingPolicy):
  """ParsingPolicy for TFLite documentation."""

  def __init__(self, yaml_parser_by_tag_name: Mapping[
      str, yaml_parser_lib.AbstractYamlParser], publisher: str, model_name: str,
               model_version: str) -> None:
    super(LiteParsingPolicy, self).__init__(
        yaml_parser_by_tag_name,
        publisher,
        model_name,
        model_version,
        required_metadata={ASSET_PATH_KEY, PARENT_MODEL_KEY},
        optional_metadata={
            COLAB_KEY, VISUALIZER_KEY
        },
        supported_asset_path_suffix=TFLITE_SUFFIX)

  @property
  def type_name(self) -> str:
    return "Lite"

  def get_allowed_file_paths(self, documentation_dir: str) -> Sequence[str]:
    """Returns the paths for PUBLISHER/(models/)MODEL/lite/VERSION.md."""
    return [
        os.path.join(documentation_dir, self._publisher, self._model_name,
                     _LITE_DIR, f"{self._model_version}.md"),
        os.path.join(documentation_dir, self._publisher, _MODELS_DIR,
                     self._model_name, _LITE_DIR, f"{self._model_version}.md")
    ]


class CoralParsingPolicy(LiteParsingPolicy):
  """ParsingPolicy for Coral documentation."""

  @property
  def type_name(self) -> str:
    return "Coral"

  def get_allowed_file_paths(self, documentation_dir: str) -> Sequence[str]:
    """Returns the paths for PUBLISHER/(models/)MODEL/coral/VERSION.md."""
    return [
        os.path.join(documentation_dir, self._publisher, self._model_name,
                     _CORAL_DIR, f"{self._model_version}.md"),
        os.path.join(documentation_dir, self._publisher, _MODELS_DIR,
                     self._model_name, _CORAL_DIR, f"{self._model_version}.md")
    ]


class PublisherParsingPolicy(ParsingPolicy):
  """ParsingPolicy for publisher documentation.

  Publisher files should always be at root/publisher/publisher.md and they
  should not contain a 'format' tag as it has no effect.
  """

  def __init__(self,
               yaml_parser_by_tag_name: Mapping[
                   str, yaml_parser_lib.AbstractYamlParser],
               publisher: str,
               model_name: str = "",
               model_version: str = "") -> None:
    super(PublisherParsingPolicy,
          self).__init__(yaml_parser_by_tag_name, publisher, model_name,
                         model_version, set(), set())

  @property
  def type_name(self) -> str:
    return "Publisher"

  @property
  def id(self) -> str:
    return self.publisher

  def get_allowed_file_paths(self, documentation_dir: str) -> Sequence[str]:
    """Returns a Sequence including the path to PUBLISHER/PUBLISHER.md."""
    return [
        os.path.join(documentation_dir, self.publisher, f"{self.publisher}.md")
    ]


class DocumentationParser:
  """Class used for parsing model documentation strings."""

  def __init__(
      self, root_dir: str, documentation_dir: str,
      yaml_parser_by_tag_name: Mapping[str, yaml_parser_lib.AbstractYamlParser]
  ) -> None:
    self._root_dir = root_dir
    self._documentation_dir = documentation_dir
    self._yaml_parser_by_tag_name = yaml_parser_by_tag_name
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
    return self._parsed_metadata  # pytype: disable=bad-return-type  # bind-properties

  def _raise_error(self, error_message: str) -> None:
    raise MarkdownDocumentationError(error_message)

  def _assert_publisher_page_exists(self) -> None:
    """Asserts that publisher page exists for the publisher of this model."""
    # Use a publisher policy to get the expected documentation page path.
    publisher_policy = PublisherParsingPolicy(self._yaml_parser_by_tag_name,
                                              self.policy.publisher)
    expected_publisher_doc_file_path = publisher_policy.get_allowed_file_paths(
        self._documentation_dir)[0]
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
      # Not an empty line and not expected metadata.
      raise MarkdownDocumentationError(
          f"Unexpected line found: '{self._lines[self._current_index]}'. "
          "Please refer to "
          "https://www.tensorflow.org/hub/writing_model_documentation for "
          "information about markdown format.")

  def validate(self, validation_config: ValidationConfig,
               file_path: str) -> None:
    """Validate one documentation markdown file."""
    self._file_path = file_path
    raw_content = filesystem_utils.get_content(self._file_path)
    self._lines = raw_content.split("\n")
    first_line = self._lines[0].replace("&zwnj;", "")
    self.policy = ParsingPolicy.from_string(first_line,
                                            self._yaml_parser_by_tag_name)

    try:
      self._assert_publisher_page_exists()
      if not validation_config.skip_file_path_check:
        self.policy.assert_correct_file_path(self._file_path,
                                             self._documentation_dir)
      # Populate _parsed_description with the description
      self._consume_description()
      # Populate _parsed_metadata with the metadata tag mapping
      self._consume_metadata()
      self.policy.assert_correct_metadata(self._parsed_metadata)
      if not validation_config.skip_content_check:
        self.policy.assert_correct_content(self._documentation_dir,
                                           self._lines[self._current_index:])
      if not validation_config.skip_asset_check:
        self.policy.validate_asset_path(validation_config,
                                        self._parsed_metadata, self._file_path)
    except MarkdownDocumentationError as e:
      self._raise_error(str(e))


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
  # Passing this map prevents re-initializing the needed parsers for each
  # document, which would be IO heavy due to reading configs from YAML files.
  yaml_parser_by_tag_name = {
      tag_name:
      yaml_parser_lib.AbstractYamlParser.from_tag_name(root_dir, tag_name)
      for tag_name in yaml_parser_lib.TAG_TO_YAML_MAP
  }
  logging.info("Going to validate files %s in documentation directory %s.",
               files_to_validate, documentation_dir)
  validated = 0
  file_to_error = dict()

  for file_path in files_to_validate:
    logging.info("Validating %s.", file_path)
    documentation_parser = DocumentationParser(root_dir, documentation_dir,
                                               yaml_parser_by_tag_name)
    try:
      absolute_path = os.path.join(documentation_dir, file_path)
      documentation_parser.validate(validation_config, absolute_path)
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


POLICY_BY_URL_PREFIX = {
    "lite-model/": LiteParsingPolicy,
    "coral-model/": CoralParsingPolicy,
    "tfjs-model/": TfjsParsingPolicy
}

POLICY_BY_PATTERN = {
    MODEL_HANDLE_PATTERN: SavedModelParsingPolicy,
    PLACEHOLDER_HANDLE_PATTERN: PlaceholderParsingPolicy,
    LITE_HANDLE_PATTERN: LiteParsingPolicy,
    TFJS_HANDLE_PATTERN: TfjsParsingPolicy,
    CORAL_HANDLE_PATTERN: CoralParsingPolicy,
    PUBLISHER_HANDLE_PATTERN: PublisherParsingPolicy,
    COLLECTION_HANDLE_PATTERN: CollectionParsingPolicy
}


def _get_policies_for_line_with_model_urls(
    line: str) -> Iterator[ModelParsingPolicy]:
  """Yields a parsing policy for models that correspond to found tfhub.dev URLs.

  For each tfhub.dev URL that can be found in the given string, the
  corresponding parsing policy is yielded. The URL must match
  https://tfhub.dev/PUBLISHER/((tfjs|lite|coral)-model/)NAME(/VERSION) so a
  PUBLISHER and a NAME must be specified while a VERSION is optional. If no
  version is specified, a '*' will be used to simplify finding all documentation
  files that could be rendered.

  Args:
    line: String that could contain a tfhub.dev model URL.

  Yields:
    A ModelParsingPolicy which can be used to find the documentation files for
    the model corresponding to the URL.
  """
  for url_match in _TFHUB_MODEL_URL_PATTERN.finditer(line):
    groupdict = url_match.groupdict()
    publisher = groupdict.get("publisher")
    prefix = groupdict.get("prefix")
    name = groupdict.get("name")

    trailing_version = re.search(rf"(?<=/){_MODEL_VERSION_PATTERN}$", name)
    if trailing_version is None:
      version = "*"
    else:
      # Move the trailing version from the model name to version.
      version = trailing_version.group(0)
      name = re.sub(f"/{version}$", "", name)

    policy_class = POLICY_BY_URL_PREFIX.get(prefix, SavedModelParsingPolicy)
    yield policy_class({}, publisher, name, version)


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
