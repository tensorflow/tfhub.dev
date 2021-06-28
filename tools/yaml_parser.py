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
"""Helper utilities to load tags from YAML files."""

import abc
import collections
import os
from typing import AbstractSet, Any, Mapping, Sequence, Type, TypeVar
import urllib

import attr
import yaml

# Maps a tag name to the YAML file that contains the config for valid values.
TAG_TO_YAML_MAP = collections.OrderedDict({
    "colab": "tags/colab.yaml",
    "dataset": "tags/dataset.yaml",
    "demo": "tags/demo.yaml",
    "interactive-visualizer": "tags/interactive_visualizer.yaml",
    "language": "tags/language.yaml",
    "license": "tags/license.yaml",
    "network-architecture": "tags/network_architecture.yaml",
    "task": "tags/task.yaml"
})

# Field names in the enumerable YAML config files.
_ID_KEY = "id"
_VALUES_KEY = "values"
# Field names in the YAML config files for URLs.
_FORMAT_KEY = "format"
_URL_FORMAT = "url"
_SUPPORTED_FORMAT_VALUES = frozenset({_URL_FORMAT})
_REQUIRED_DOMAIN_KEY = "required_domain"
_HTTPS_SCHEME = "https"

AbstractYamlParserType = TypeVar(
    "AbstractYamlParserType", bound="AbstractYamlParser")
EnumerableTagValuesValidatorType = TypeVar(
    "EnumerableTagValuesValidatorType", bound="EnumerableTagValuesValidator")
UrlTagConfigType = TypeVar("UrlTagConfigType", bound="UrlTagConfig")


class AbstractYamlParser(metaclass=abc.ABCMeta):
  """Validates tag values depending on their tag type."""

  def __init__(self, root_dir: str, tag_name: str) -> None:
    self._root_dir = root_dir
    self._tag_name = tag_name
    self._relative_tag_file = TAG_TO_YAML_MAP[self._tag_name]

  @classmethod
  def from_tag_name(cls: Type[AbstractYamlParserType], root_dir: str,
                    tag_name: str) -> AbstractYamlParserType:
    """Builds an YamlParser from the root dir and the tag name."""
    parser_type_from_tag_name = {
        "colab": UrlYamlParser,
        "dataset": EnumerableYamlParser,
        "demo": UrlYamlParser,
        "interactive-visualizer": EnumerableYamlParser,
        "language": EnumerableYamlParser,
        "license": EnumerableYamlParser,
        "network-architecture": EnumerableYamlParser,
        "task": EnumerableYamlParser,
    }
    if tag_name not in parser_type_from_tag_name:
      raise ValueError(f"No supported parser found for tag {tag_name}.")
    return parser_type_from_tag_name[tag_name](root_dir, tag_name)

  @abc.abstractmethod
  def assert_tag_values_are_correct(self, tag_values: AbstractSet[str]) -> None:
    """Checks that the given tag values are valid."""


@attr.s(auto_attribs=True)
class EnumerableTagValue:
  """Representation of a single tag value.

  Attributes:
    id: String representing the unique identifier of an item e.g. 'en'.
  """
  id: str


@attr.s(auto_attribs=True)
class EnumerableTagValuesValidator:
  """Loads all tag values and validates them.

  Items of enumerable tags (dataset, license, ...) should have an 'id' field,
  which can be used in the model Markdown documentation.

  Attributes:
    values: Sequence containing the possible TagValues a tag can be set to.
  """
  values: Sequence[EnumerableTagValue]

  @classmethod
  def from_yaml(
      cls: Type[EnumerableTagValuesValidatorType],
      yaml_config: Mapping[str, Any]) -> EnumerableTagValuesValidatorType:
    """Builds an EnumerableTagValuesValidator instance from a YAML config.

    Args:
      yaml_config: A config loaded from a YAML file.

    Returns:
      An EnumerableTagValuesValidator instance loaded from the YAML config.

    Raises:
      ValueError:
        - if yaml_config does not contain a `values` field.
        - if a tag item within yaml_config does not contain an `id` field.
    """

    if _VALUES_KEY not in yaml_config:
      raise ValueError(f"YAML config should contain `{_VALUES_KEY}` key "
                       f"but was {yaml_config}.")

    values = list()
    for item in yaml_config[_VALUES_KEY]:
      if _ID_KEY not in item:
        raise ValueError(f"A tag item must contain an `{_ID_KEY}` field "
                         f" but was {item}.")
      values.append(EnumerableTagValue(id=item[_ID_KEY]))
    return cls(values)


class EnumerableYamlParser(AbstractYamlParser):
  """Checks that given tag values match supported values from the YAML file."""

  def __init__(self, root_dir: str, tag_name: str) -> None:
    super().__init__(root_dir, tag_name)
    self._supported_values = None

  def _load_supported_values(self) -> None:
    """Loads the supported values for the tag from the respective YAML file.

    Raises:
      FileNotFoundError: if the YAML file does not exist.
      yaml.parser.ParserError: if the YAML file is no valid YAML file.
      ValueError:
        - if the YAML file does not contain a `values` field.
        - if the YAML file contains an item that does not have an `id` field.
    """
    with open(os.path.join(self._root_dir,
                           self._relative_tag_file)) as yaml_file:
      yaml_config = yaml.safe_load(yaml_file.read())
    tag_validator = EnumerableTagValuesValidator.from_yaml(yaml_config)
    self._supported_values = {item.id for item in tag_validator.values}

  def _get_supported_values(self) -> AbstractSet[str]:
    """Returns the supported values that were loaded from the YAML file."""
    if self._supported_values is None:
      self._load_supported_values()
    return self._supported_values

  def assert_tag_values_are_correct(self, tag_values: AbstractSet[str]) -> None:
    """Checks that the tag values belong to `id` fields in the YAML file.

    Args:
      tag_values: Set of strings that should be validated against the YAML file
        belonging to this parser.

    Raises:
      FileNotFoundError: if the YAML file does not exist.
      yaml.parser.ParserError: if the YAML file is no valid YAML file.
      ValueError:
        - if the YAML file does not contain a `values` field.
        - if the YAML file contains an item that does not have an `id` field.
        - if `tag_values` contains values different from those specified in the
          YAML file of this parser.
    """
    supported_values = self._get_supported_values()
    unsupported_values = tag_values - supported_values
    if unsupported_values:
      raise ValueError(
          f"Unsupported values for {self._tag_name} tag were found: "
          f"{sorted(unsupported_values)}. Please add them to "
          f"{self._relative_tag_file}.")


@attr.s(auto_attribs=True)
class UrlTagConfig:
  """Config for a tag that can be set to an URL.

  Attributes:
    format: String specifying the allowed format. Must be one of
      _SUPPORTED_FORMAT_VALUES.
    required_domain: Optional; String specifying the allowed domain of an URL.
  """
  format: str
  required_domain: str = None

  @classmethod
  def from_yaml(cls: Type[UrlTagConfigType],
                yaml_config: Mapping[str, Any]) -> UrlTagConfigType:
    """Builds an UrlTagConfig instance from a YAML config.

    Args:
      yaml_config: A config loaded from a YAML file.

    Returns:
      An UrlTagConfig instance loaded from the YAML config.

    Raises:
      ValueError:
        - if yaml_config does not contain a `format` field.
        - if yaml_config['format'] is not one of the supported formats.
    """
    if _FORMAT_KEY not in yaml_config:
      raise ValueError(f"YAML config should contain `{_FORMAT_KEY}` key "
                       f"but was {yaml_config}.")

    format_value = yaml_config[_FORMAT_KEY]
    domain_value = None

    if format_value not in _SUPPORTED_FORMAT_VALUES:
      raise ValueError(
          f"'format' must be one of {set(_SUPPORTED_FORMAT_VALUES)} but was "
          f"{format_value}.")

    if _REQUIRED_DOMAIN_KEY in yaml_config:
      domain_value = yaml_config[_REQUIRED_DOMAIN_KEY]
    return cls(format=format_value, required_domain=domain_value)


class UrlYamlParser(AbstractYamlParser):
  """Validates URL values according to their YAML config."""

  def __init__(self, root_dir: str, tag_name: str) -> None:
    super().__init__(root_dir, tag_name)
    self._url_tag_config = None

  def _load_url_tag_config(self) -> None:
    """Loads the UrlTagConfig from the YAML file of this parser.

    Raises:
      FileNotFoundError: if the YAML file does not exist.
      yaml.parser.ParserError: if a YAML file is no valid YAML file.
      ValueError:
        - if the YAML content does not contain a `format` field.
        - if the 'format' field is not one of the supported formats.
    """
    with open(os.path.join(self._root_dir,
                           self._relative_tag_file)) as yaml_file:
      yaml_config = yaml.safe_load(yaml_file.read())
    self._url_tag_config = UrlTagConfig.from_yaml(yaml_config)

  def _assert_tag_value_is_url(self, tag_value: str) -> None:
    """Checks that `tag_value` is a valid URL as specified by the config.

    Args:
      tag_value: String that should be checked for being an allowed URL.

    Raises:
      ValueError:
        - if `tag_value` is not a valid URL.
        - if `tag_value` is not an HTTPS URL.
        - if the config specifies a `required_domain` but the domain of
          `tag_value` is unequal to that required domain.
    """
    try:
      result = urllib.parse.urlparse(tag_value)
    except ValueError:
      raise ValueError(f"{tag_value} is no valid URL.")
    if result.scheme != _HTTPS_SCHEME:
      raise ValueError(f"{tag_value} is not an HTTPS URL.")
    if (self._url_tag_config.required_domain is not None and
        self._url_tag_config.required_domain != result.netloc):
      raise ValueError("URL must lead to domain "
                       f"{self._url_tag_config.required_domain} but is "
                       f"{result.netloc}.")

  def assert_tag_values_are_correct(self, tag_values: AbstractSet[str]) -> None:
    """Checks that the passed values are valid wrt. the loaded config.

    Args:
      tag_values: Set of strings specifying tag values e.g.
        {"https://demopage.com"}.

    Raises:
      FileNotFoundError: if the YAML file containing the config does not exist.
      yaml.parser.ParserError: if the config YAML file is no valid YAML file.
      ValueError:
        - if the format of the loaded config is unknown.
        - if a tag value is not a valid URL.
        - if a tag value is not an HTTPS URL.
        - if the config specifies a `required_domain` but the domain of a tag
          value is unequal to that required domain.
    """
    if self._url_tag_config is None:
      self._load_url_tag_config()
    for tag_value in tag_values:
      if self._url_tag_config.format == _URL_FORMAT:
        self._assert_tag_value_is_url(tag_value)
      else:
        raise ValueError(f"Unknown format: {self._url_tag_config.format}.")
