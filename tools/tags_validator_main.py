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
"""YAML validator for tag definition files.

1) To validate all tag definition files, run:
$ python tools/tags_validator.py

2) To validate selected files, pass their relative paths:
$ python tools/tags_validator.py tags/dataset.yml [other_files]

Use the --root_dir flag to validate tag files outside of the current project.
"""
import argparse
import os
import sys

from absl import app
from absl import logging
import tags_validator

FLAGS = None


def main(_):
  root_dir = FLAGS.root_dir or os.getcwd()
  documentation_dir = os.path.join(root_dir, "tags")
  logging.info("Using %s for documentation directory.", documentation_dir)
  file_to_error = dict()

  if FLAGS.file:
    logging.info("Going to validate files %s in documentation directory %s.",
                 FLAGS.file, documentation_dir)
    files_to_validate = [os.path.join(documentation_dir, f) for f in FLAGS.file]
    file_to_error = tags_validator.validate_tag_files(files_to_validate)
  else:
    logging.info("Going to validate all files in documentation directory %s.",
                 documentation_dir)
    file_to_error = tags_validator.validate_tag_dir(documentation_dir)
  if file_to_error:
    logging.error("The following files contain issues: %s", file_to_error)
  else:
    logging.info("Successfully validated all files.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "file",
      type=str,
      default=None,
      help=("Path to files to validate. Path is relative to `--root_dir`."),
      nargs="*")
  parser.add_argument(
      "--root_dir",
      type=str,
      default=None,
      help=("Root directory that contains tag definition files under "
            "./tags. Defaults to current directory."))
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
