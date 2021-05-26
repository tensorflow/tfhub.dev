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
from typing import AbstractSet

import yaml

# Maps a tag name to the YAML file path where supported values are configured.
TAG_TO_YAML_MAP = collections.OrderedDict({
    "dataset": "tags/dataset.yaml",
    "language": "tags/language.yaml"
})

# Field names in the used YAML config files.
ID_KEY = "id"
VALUES_KEY = "values"


class YamlParser(object):
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

  def load_supported_values(self) -> None:
    """Loads the supported values for each tag from the respective YAML file.

    Raises:
      yaml.parser.ParserError: if a YAML file is no valid YAML file.
      FileNotFoundError: if a YAML file does not exist.
    """
    supported_values_map = collections.OrderedDict()
    for tag_name, yaml_path in TAG_TO_YAML_MAP.items():
      with open(os.path.join(self._root_dir, yaml_path)) as yaml_file:
        yaml_config = yaml.safe_load(yaml_file.read())
      supported_values_map[tag_name] = {
          item[ID_KEY] for item in yaml_config[VALUES_KEY]
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
      self.load_supported_values()

    if tag_name not in self._supported_values_map:
      raise ValueError(f"No supported ids found for tag {tag_name}.")
    return self._supported_values_map[tag_name]
