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

import os
from typing import Set

import yaml

# Path to the language config file relative to the root directory.
LANGUAGE_YAML = "tags/language.yaml"

# Field names in the used YAML config files.
ID_KEY = "id"
VALUES_KEY = "values"


class YamlParser(object):
  """Loads supported tags from the YAML config files."""

  def __init__(self, root_dir: str):
    """Creates a YamlParser by passing the absolute path to the root dir."""
    self._root_dir = root_dir
    self._loaded_language_config = None

  def get_supported_languages(self) -> Set[str]:
    """Return the ids specified in the YAML file defining the languages.

    Returns:
      Set of language ids that are defined in the YAML config file.

    Raises:
      yaml.parser.ParserError: if the language.yaml file is no valid YAML file.
      FileNotFoundError: if the language.yaml does not exist.
    """

    if self._loaded_language_config is None:
      # Should point to //third_party/py/tfhub_dev/tags/language.yaml.
      with open(os.path.join(self._root_dir, LANGUAGE_YAML)) as yaml_file:
        self._loaded_language_config = yaml.safe_load(yaml_file.read())

    return {item[ID_KEY] for item in self._loaded_language_config[VALUES_KEY]}
