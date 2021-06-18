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


import abc
import os
from typing import AbstractSet, Any, List, Mapping, Optional, Type, TypeVar
import urllib

from absl import logging
import ruamel.yaml.nodes as nodes
import filesystem_utils
import yaml

TagDefinitionFileParserT = TypeVar(
    "TagDefinitionFileParserT", bound="TagDefinitionFileParser")


class TagDefinitionError(Exception):
  """Problem with tag definition in a YAML file."""


class UniqueStringKeyLoader(yaml.SafeLoader):
  """YAML loader that only allows unique keys and string values."""

  YAML_STR_TAG = "tag:yaml.org,2002:str"  # https://yaml.org/type/str.html

  def construct_scalar(self, node: nodes.ScalarNode) -> nodes.ScalarNode:
    """Returns ScalarNode that contains only string values.

    Note: We only support scalars of type string.

    Args:
      node: ScalarNode that should be validated.

    Raises:
      TagDefinitionError: if a non-string value is passed.
    """
    if node.tag != self.YAML_STR_TAG:
      raise TagDefinitionError(f"Found non-string value: {node}")
    return super().construct_scalar(node)

  def construct_mapping(
      self,
      node: nodes.ScalarNode,
      deep: bool = False) -> Mapping[nodes.ScalarNode, nodes.ScalarNode]:
    """Overrides default mapper with a version that detects duplicate keys.

    Args:
      node: ScalarNode for which a mapping should be returned.
      deep: Whether to reverse the order of constructing objects.

    Returns:
      mapping: Mapping from key node to value node.

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


class TagDefinitionFileParser(metaclass=abc.ABCMeta):
  """Abstract class to parse and validate a YAML file.

  Attributes:
    _file_path: absolute path to the YAML file that should be validated.
  """

  def __init__(self, file_path: str) -> None:
    self._file_path = file_path

  @classmethod
  def create_tag_parser(cls: Type[TagDefinitionFileParserT],
                        file_path: str) -> TagDefinitionFileParserT:
    """Returns the correct parser instance for the given YAML file.

    Args:
      file_path: absolute path to the YAML file that should be validated.

    Returns:
      EnumerableTagParser or TaskTagParser instance for validating the file.

    Raises:
      TagDefinitionError: if no parser is associated to the YAML file.
    """
    file_name = os.path.basename(file_path)
    parser_from_file_name = {
        "dataset.yaml": EnumerableTagParser,
        "language.yaml": EnumerableTagParser,
        "network_architecture.yaml": EnumerableTagParser,
        "task.yaml": TaskTagParser,
        "license.yaml": LicenseTagParser,
        "interactive_visualizer.yaml": InteractiveVisualizerTagParser
    }
    if file_name not in parser_from_file_name:
      raise TagDefinitionError(f"No parser is registered for {file_name}.")
    return parser_from_file_name[file_name](file_path)

  @property
  @abc.abstractmethod
  def required_top_keys(self) -> AbstractSet[str]:
    """Set of required top-level keys in a YAML file."""

  @abc.abstractmethod
  def _validate_yaml_config(self, loaded_yaml: Mapping[str, Any]) -> None:
    """Parses a YAML file and checks that it is supported.

    Args:
      loaded_yaml: Loaded YAML, which is a result of yaml.load().
    """

  def validate(self):
    """Validates the tag definition file.

    Raises:
      TagDefinitionError:
        - the file cannot be parsed as YAML.
        - the file contains duplicate keys.
        - the file contains non-string values.
    """
    file_content = filesystem_utils.get_content(self._file_path)
    loaded_yaml = yaml.load(file_content, Loader=UniqueStringKeyLoader)
    if loaded_yaml is None:
      raise TagDefinitionError("Cannot parse file to YAML.")
    self._validate_yaml_config(loaded_yaml)


class EnumerableTagParser(TagDefinitionFileParser):
  """Class used for validating an enumerable tag definition YAML file.

  An example of a valid enumerable tag file (i.e. dataset.yaml) looks like this:

  values:                  # No other top-level keys apart from 'values' allowed
    - id: mnist            # Required
      display_name: MNIST  # Required
    - id: "no"  # Quote special keywords since only strings are allowed
      display_name: Norwegian census dataset

  Enumerable tag definition files contain a list of all items that tag can be
  set to. A tag file thus has top-level keys (i.e. 'values') and item-level keys
  (i.e. 'id', 'display_name').
  """

  VALUES_KEY = "values"
  ID_KEY = "id"
  DISPLAY_NAME_KEY = "display_name"

  def __init__(self,
               file_path: str,
               required_top_keys: Optional[AbstractSet[str]] = None,
               required_item_keys: Optional[AbstractSet[str]] = None,
               optional_item_keys: Optional[AbstractSet[str]] = None) -> None:
    super().__init__(file_path)
    if required_top_keys is None:
      self._required_top_keys = {self.VALUES_KEY}
    else:
      self._required_top_keys = required_top_keys
    if required_item_keys is None:
      self._required_item_keys = {self.ID_KEY, self.DISPLAY_NAME_KEY}
    else:
      self._required_item_keys = required_item_keys
    if optional_item_keys is None:
      self._optional_item_keys = set()
    else:
      self._optional_item_keys = optional_item_keys

  @property
  def required_top_keys(self) -> AbstractSet[str]:
    """The set of required keys at the root of the YAML file."""
    return self._required_top_keys

  @property
  def required_item_keys(self) -> AbstractSet[str]:
    """The set of required keys at the item-level of the YAML file."""
    return self._required_item_keys

  @property
  def optional_item_keys(self) -> AbstractSet[str]:
    """The set of optional keys at the item-level of the YAML file."""
    return self._optional_item_keys

  @property
  def supported_item_keys(self) -> AbstractSet[str]:
    """The union of required and optional keys at the item-level."""
    return set.union(self.required_item_keys, self.optional_item_keys)

  def _assert_valid_top_level_keys(self, loaded_yaml: Mapping[str,
                                                              str]) -> None:
    """Ensures that the top-most tag keys are valid.

    Args:
      loaded_yaml: Loaded YAML, which is a result of yaml.load().

    Raises:
      TagDefinitionError:
        - if the top-level keys are unequal to the required top-level keys.
    """
    if loaded_yaml.keys() != self.required_top_keys:
      raise TagDefinitionError(
          f"Expected top-level keys {self.required_top_keys} "
          f"but got {sorted(loaded_yaml.keys())}.")

  def _assert_valid_item_level_keys(self, item: Mapping[str, str]) -> None:
    """Validates the keys of the given item.

    Args:
      item: Mapping representing a tag item e.g. {id=mnist, display_name=MNIST}.

    Raises:
      TagDefinitionError:
        - if not all required item-level keys are set.
        - if keys are different from the supported item-level keys.
    """
    missing_required_field = self.required_item_keys - set(item.keys())
    if missing_required_field:
      raise TagDefinitionError(f"Missing required item-level keys: "
                               f"{sorted(missing_required_field)}.")
    unsupported_field = set(item.keys()) - self.supported_item_keys
    if unsupported_field:
      raise TagDefinitionError(
          f"Unsupported item-level keys: {unsupported_field}.")

  def _validate_yaml_config(self, loaded_yaml: Mapping[str, Any]) -> None:
    """Parses a YAML file and checks that it is supported.

    Args:
      loaded_yaml: Loaded YAML, which is a result of yaml.load().

    Raises:
      TagDefinitionError if
        - if the top-level keys are unequal to the required top-level keys.
        - if not all required item-level keys are set.
        - if keys are different from the supported item-level keys.
    """

    self._assert_valid_top_level_keys(loaded_yaml)
    for item in loaded_yaml[self.VALUES_KEY]:
      self._assert_valid_item_level_keys(item)


class LicenseTagParser(EnumerableTagParser):
  """Class for parsing and validating license.yaml.

  On an item-level, this tag requires to set the 'id' and 'display_name' fields
  while it optionally allows to set the 'url' field. An example of a valid file
  looks like this:

  values:
    - id: apache-2.0
      display_name: Apache-2.0
      url: https://opensource.org/licenses/Apache-2.0  # Optional
  """

  URL_KEY = "url"

  def __init__(self, file_path: str) -> None:
    super().__init__(file_path, optional_item_keys={self.URL_KEY})


class TaskTagParser(EnumerableTagParser):
  """Class for parsing and validating task.yaml.

  Unlike other enumerable tag files, this tag requires to also set the 'domains'
  field at the item-level. An example of a valid file looks like this:

  values:
    - id: image-detection
      display_name: Image detection
      domains:
        - image
  """

  DOMAINS_KEY = "domains"

  def __init__(self, file_path: str) -> None:
    super().__init__(
        file_path,
        required_item_keys={
            self.ID_KEY, self.DISPLAY_NAME_KEY, self.DOMAINS_KEY
        })


class InteractiveVisualizerTagParser(EnumerableTagParser):
  r"""Class for parsing and validating interactive_visualizer.yaml.

  This tag requires to set both 'id' and 'url_template' at the item-level.
  An example of a valid file looks like this:

  values:
    - id: tflite_object_detector
      url_template: "https://www.gstatic.com/visualizer.html?\
        modelHandle={MODEL_HANDLE}"
  """

  URL_TEMPLATE_KEY = "url_template"
  SUPPORTED_VARIABLES = frozenset({
      "MODEL_HANDLE", "MODEL_NAME", "MODEL_URL", "PUBLISHER_NAME",
      "PUBLISHER_ICON_URL"
  })
  ALLOWED_URL_PREFIXES = frozenset({
      "https://www.gstatic.com/",
      "https://storage.googleapis.com/tfhub-visualizers/",
      "https://storage.googleapis.com/interactive_visualizer/"
  })
  HTTPS_SCHEME = "https"

  def __init__(self, file_path: str) -> None:
    super().__init__(
        file_path, required_item_keys={self.ID_KEY, self.URL_TEMPLATE_KEY})

  def _assert_valid_url(self, possible_url: str) -> None:
    """Checks that the given string is a valid and allowed URL.

    The HTTPS URL should be parseable, not contain spaces and start with an
    allowed prefix.

    Args:
      possible_url: String that should be checked for being an allowed URL.

    Raises:
      TagDefinitionError:
        - if the string is not a valid URL.
        - if the URL contains spaces.
        - if the URL is not an HTTPS URL.
        - if the URL does not specify both a domain and a path.
        - if the URL does not start with one allowed URL prefix.
    """
    try:
      result = urllib.parse.urlparse(possible_url)
    except ValueError:
      raise TagDefinitionError(f"{possible_url} is not a valid URL.")
    if " " in possible_url:
      raise TagDefinitionError(f"{possible_url} must not contain spaces.")
    if result.scheme != self.HTTPS_SCHEME:
      raise TagDefinitionError(f"{possible_url} is not a HTTPS URL.")
    if not all([result.netloc, result.path]):
      raise TagDefinitionError(
          f"{possible_url} must specify a domain and a path.")
    # Keep the two preceding tests even if the next one is stricter. This should
    # allow to be future proof in case e.g. an http prefix is accidentially
    # added to ALLOWED_URL_PREFIXES.
    if not possible_url.startswith(tuple(self.ALLOWED_URL_PREFIXES)):
      raise TagDefinitionError(
          f"URL needs to start with any of {sorted(self.ALLOWED_URL_PREFIXES)}"
          f" but was {possible_url}.")

  def _assert_valid_item_level_keys(self, item: Mapping[str, str]) -> None:
    """Validates the keys of the given item.

    Args:
      item: Mapping representing a visualizer item e.g. {id=spice,
        url_template=https://www.gstatic.com/aihub/tfhub/demos/spice.html}.

    Raises:
      TagDefinitionError:
        - if not all required item-level keys are set.
        - if keys are different from the supported item-level keys.
        - if 'url_template' contains additional injectible variables.
        - if 'url_template' is not a valid URL.
        - if 'url_template' contains spaces.
        - if 'url_template' is no HTTPS URL.
        - if 'url_template' does not specify both a domain and a path.
        - if 'url_template' does not start with one allowed URL prefix.
    """
    super()._assert_valid_item_level_keys(item)
    variable_to_placeholder_map = {
        variable: "placeholder" for variable in self.SUPPORTED_VARIABLES
    }
    template_url = item[self.URL_TEMPLATE_KEY]
    try:
      formatted_url = template_url.format(**variable_to_placeholder_map)
    except KeyError:
      raise TagDefinitionError(
          f"Only substituting {sorted(list(self.SUPPORTED_VARIABLES))} is "
          f"allowed: {template_url}.")
    self._assert_valid_url(formatted_url)


def validate_tag_files(
    file_paths: List[str]) -> Mapping[str, TagDefinitionError]:
  """Validate all tag definition files at the given paths.

  Args:
    file_paths: List of absolute paths to the YAML files.

  Returns:
    file_to_error: Mapping from file path to occuring error.

    Raises:
      TagDefinitionError if:
        - a file cannot be parsed as YAML.
        - a YAML file contains duplicate keys.
        - a YAML file contains non-string values.
        - a YAML file does not contain the required top-level keys.
        - a YAML file does not set all required item-level keys.
        - a YAML file sets unsupported item-level keys.
  """
  file_to_error = dict()
  for file_path in file_paths:
    logging.info("Validating %s.", file_path)
    try:
      TagDefinitionFileParser.create_tag_parser(file_path).validate()
    except TagDefinitionError as e:
      file_to_error[file_path] = e
  return file_to_error


def validate_tag_dir(directory: str) -> Mapping[str, TagDefinitionError]:
  """Validate all tag definition files in the given directory.

  Args:
    directory: Directory path over which should be walked to find all tag files.

  Returns:
    file_to_error: Mapping from file path to occuring error.

    Raises:
      TagDefinitionError if:
        - a file cannot be parsed as YAML.
        - a YAML file contains duplicate keys.
        - a YAML file contains non-string values.
        - a YAML file does not contain the required top-level keys.
        - a YAML file does not set all required item-level keys.
        - a YAML file sets unsupported item-level keys.
  """
  file_paths = list(filesystem_utils.recursive_list_dir(directory))
  return validate_tag_files(file_paths)
