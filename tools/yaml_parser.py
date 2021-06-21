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

import collections
import os
import re
from typing import AbstractSet, Any, Mapping, TypeVar, Sequence, Type

import attr
import yaml

# Maps a tag name to the YAML file path where supported values are configured.
TAG_TO_YAML_MAP = collections.OrderedDict({
    "dataset": "tags/dataset.yaml",
    "language": "tags/language.yaml",
    "task": "tags/task.yaml",
    "network-architecture": "tags/network_architecture.yaml",
    "license": "tags/license.yaml",
    "interactive-model-name": "tags/interactive_visualizer.yaml"
})

# Field names in the used YAML config files.
ID_KEY = "id"
DISPLAY_NAME = "display_name"
VALUES_KEY = "values"

EnumerableTagValuesValidatorType = TypeVar(
    "EnumerableTagValuesValidatorType", bound="EnumerableTagValuesValidator")


@attr.s(auto_attribs=True)
class TagValue:
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
  values: Sequence[TagValue]

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

    if VALUES_KEY not in yaml_config:
      raise ValueError(f"YAML config should contain `{VALUES_KEY}` key "
                       f"but was {yaml_config}.")

    values = list()
    for item in yaml_config[VALUES_KEY]:
      if ID_KEY not in item:
        raise ValueError(f"A tag item must contain an `{ID_KEY}` field but was "
                         f"{item}.")
      values.append(TagValue(id=item[ID_KEY]))
    return cls(values)

  def validate(self) -> None:
    """Ensures that the given config only contains valid values.

    Raises:
      ValueError: if the `id` field of an item is invalid.
    """
    id_pattern = r"[a-z-\d\.]+"
    for item in self.values:
      if re.fullmatch(id_pattern, item.id) is None:
        raise ValueError(f"The value of an id must match {id_pattern} but was "
                         f"{item.id}.")


class YamlParser:
  """Loads supported tags from the YAML config files.

     Attributes:
       root_dir: An absolute path to the root directory of the project.
       supported_values_map: An OrderedDict that maps from a tag name to all
         supported ids like:
         {"dataset": {"mnist", "imagenet"}, "language": {"en", "fr"}}.
  """

  def __init__(self, root_dir: str) -> None:
    """Creates a YamlParser by passing the absolute path to the root dir."""
    self._root_dir = root_dir
    self._supported_values_map = None

  def _load_supported_values(self) -> None:
    """Loads the supported values for each tag from the respective YAML file.

    Raises:
      FileNotFoundError: if a YAML file does not exist.
      yaml.parser.ParserError: if a YAML file is no valid YAML file.
      ValueError: if a YAML file contains unsupported values.
    """
    supported_values_map = collections.OrderedDict()
    for tag_name, yaml_path in TAG_TO_YAML_MAP.items():
      with open(os.path.join(self._root_dir, yaml_path)) as yaml_file:
        yaml_config = yaml.safe_load(yaml_file.read())
      tag_validator = EnumerableTagValuesValidator.from_yaml(yaml_config)
      tag_validator.validate()
      supported_values_map[tag_name] = {
          item.id for item in tag_validator.values
      }
    self._supported_values_map = supported_values_map

  def get_supported_values(self, tag_name: str) -> AbstractSet[str]:
    """Returns the supported values for a given Markdown tag.

    Args:
      tag_name: Key of a Markdown tag e.g. "dataset" or "language".

    Returns:
      Set of ids that are defined in the respective YAML config file.

    Raises:
      ValueError: if `tag_name` has no supported values configured in a YAML
        file.
    """
    if self._supported_values_map is None:
      self._load_supported_values()

    if tag_name not in self._supported_values_map:
      raise ValueError(f"No supported ids found for tag {tag_name}.")
    return self._supported_values_map[tag_name]
