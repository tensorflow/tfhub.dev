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
"""Library for validating tag definition files written in YAML."""


import os
from typing import AbstractSet, Dict, List
from absl import logging
import ruamel.yaml.nodes as nodes
import filesystem_utils
import yaml

VALUES_KEY = "values"
REQUIRED_TOP_LEVEL_KEYS = frozenset([VALUES_KEY])
DEFAULT_REQUIRED_ITEM_LEVEL_KEYS = ["id", "display_name"]
DEFAULT_OPTIONAL_ITEM_LEVEL_KEYS = frozenset(
    ["url", "description", "aggregation_rule"])
TAG_TO_ADDITIONAL_REQUIRED_KEYS = {"task.yaml": ["domains"]}
YAML_STR_TAG = "tag:yaml.org,2002:str"  # https://yaml.org/type/str.html


def get_required_item_level_keys(file_name: str) -> AbstractSet[str]:
  """Returns the required keys of a YAML item for a file."""
  return set(DEFAULT_REQUIRED_ITEM_LEVEL_KEYS +
             TAG_TO_ADDITIONAL_REQUIRED_KEYS.get(file_name, []))


def get_supported_item_level_keys(file_name: str) -> AbstractSet[str]:
  """Returns union of required and optional keys of a YAML item for a file."""
  return set.union(
      get_required_item_level_keys(file_name),
      set(DEFAULT_OPTIONAL_ITEM_LEVEL_KEYS))


class TagDefinitionError(Exception):
  """Problem with tag definition in a YAML file."""


class UniqueStringKeyLoader(yaml.SafeLoader):
  """YAML loader that only allows unique keys and string values."""

  def construct_scalar(self, node: nodes.ScalarNode) -> nodes.ScalarNode:
    """Returns ScalarNode that contains only string values.

    Note: We only support scalars of type string.

    Args:
      node: ScalarNode that should be validated.

    Raises:
      TagDefinitionError: if a non-string value is passed.
    """
    if node.tag != YAML_STR_TAG:
      raise TagDefinitionError(f"Found non-string value: {node}")
    return super().construct_scalar(node)

  def construct_mapping(
      self,
      node: nodes.ScalarNode,
      deep: bool = False) -> Dict[nodes.ScalarNode, nodes.ScalarNode]:
    """Overrides default mapper with a version that detects duplicate keys.

    Args:
      node: ScalarNode for which a mapping should be returned.
      deep: Whether to reverse the order of constructing objects.

    Returns:
      mapping: Dict mapping from key node to value node.

    Raises:
      TagDefinitionError: if duplicate keys are found.
    """
    mapping = {}
    for key_node, value_node in node.value:
      key = self.construct_object(key_node, deep=deep)
      if key in mapping:
        raise TagDefinitionError(f"Found duplicate key: {key}")
      value = self.construct_object(value_node, deep=deep)
      mapping[key] = value
    return mapping


class TagDefinitionFileParser(object):
  """Class used for parsing and validating a tag definition YAML file.

  An example of a valid tag file (i.e. dataset.yml) looks like this:

  values:                  # No other top-level keys apart from 'values' allowed
    - id: mnist            # Required
      display_name: MNIST  # Required
      url: http://yann.lecun.com/exdb/mnist/ # Optional
      description: Hand-written digits       # Optional
      aggregation_rule: UNION                # Optional
    - id: "no"  # Quote special keywords since only strings are allowed
      display_name: Norwegian census dataset

  """

  def __init__(self, file_path: str):
    self._file_path = file_path

  def assert_valid_top_level_keys(self, loaded_yaml: Dict[str, str]):
    """The top-most tag keys should be valid.

    Args:
      loaded_yaml: Loaded YAML, which is a result of yaml.load().

    Raises:
      TagDefinitionError:
        - if the top-level keys are unequal to REQUIRED_TOP_LEVEL_KEYS.
    """
    if loaded_yaml.keys() != REQUIRED_TOP_LEVEL_KEYS:
      raise TagDefinitionError(
          f"Expected top-level keys {set(REQUIRED_TOP_LEVEL_KEYS)} "
          f"but got {sorted(loaded_yaml.keys())}.")

  def assert_valid_item_level_keys(self, file_name: str,
                                   item: Dict[str, str]) -> None:
    """Validates the keys of the given item.

    Args:
      file_name: Name of the YAML file that is being validated e.g. 'task.yaml'.
      item: Dict representing a tag item e.g. {id=mnist, display_name=MNIST}.

    Raises:
      TagDefinitionError:
        - if not all keys specified in REQUIRED_ITEM_LEVEL_KEYS are set.
        - if keys different from SUPPORTED_ITEM_LEVEL_KEYS are set.
    """
    missing_required_field = get_required_item_level_keys(
        file_name) - item.keys()
    if missing_required_field:
      raise TagDefinitionError(
          f"Missing required item-level keys: {missing_required_field}.")
    unsupported_field = item.keys() - get_supported_item_level_keys(file_name)
    if unsupported_field:
      raise TagDefinitionError(
          f"Unsupported item-level keys: {unsupported_field}.")

  def parse_yaml_file(self, file_content: str) -> None:
    """Parses a YAML file and checks that it is supported.

    Args:
      file_content: file content that should be parsed and validated.

    Raises:
      TagDefinitionError:
        - if `file_content` cannot be parsed as YAML.
        - if `assert_valid_top_level_keys` or `assert_valid_item_level_keys`
          fail.
    """
    loaded_yaml = yaml.load(file_content, Loader=UniqueStringKeyLoader)
    file_name = os.path.basename(self._file_path)
    if loaded_yaml is None:
      raise TagDefinitionError("Cannot parse file to YAML.")
    self.assert_valid_top_level_keys(loaded_yaml)
    for item in loaded_yaml[VALUES_KEY]:
      self.assert_valid_item_level_keys(file_name, item)

  def validate(self):
    """Validate one tag definition file.

    Raises:
      TagDefinitionError if
        - the file cannot be parses as YAML.
        - the file contains duplicate keys.
        - the file contains non-string values.
        - the top-level keys are unequal to REQUIRED_TOP_LEVEL_KEYS.
        - an item does not at least specify the keys id, display_name.
        - an item contains a key that is not in SUPPORTED_ITEM_LEVEL_KEYS.
    """
    file_content = filesystem_utils.get_content(self._file_path)
    self.parse_yaml_file(file_content)


def validate_tag_files(file_paths: List[str]) -> Dict[str, TagDefinitionError]:
  """Validate all tag definition files at the given paths.

  Args:
    file_paths: List of absolute paths to the YAML files.

  Returns:
    file_to_error: Dict mapping from file path to occuring error.
  """
  file_to_error = dict()
  for file_path in file_paths:
    logging.info("Validating %s.", file_path)
    try:
      TagDefinitionFileParser(file_path).validate()
    except TagDefinitionError as e:
      file_to_error[file_path] = e
  return file_to_error


def validate_tag_dir(directory: str) -> Dict[str, TagDefinitionError]:
  """Validate all tag definition files in the given directory.

  Args:
    directory: Directory path over which should be walked to find all tag files.

  Returns:
    file_to_error: Dict mapping from file path to occuring error.
  """
  file_paths = list(filesystem_utils.recursive_list_dir(directory))
  return validate_tag_files(file_paths)
