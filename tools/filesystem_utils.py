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
"""Filesystem utilities for validating Markdown and YAML files."""

import os
import tarfile
import tensorflow as tf


def recursive_list_dir(root_dir):
  """Yields all files of a root directory tree."""
  for dirname, _, filenames in tf.io.gfile.walk(root_dir):
    for filename in filenames:
      yield os.path.join(dirname, filename)


def get_content(file_path):
  """Returns a file's content."""
  with tf.io.gfile.GFile(file_path, "r") as f:
    return f.read()


def create_archive(archive_path: str, file_path: str) -> None:
  """Creates a tar.gz archive from a file."""
  with tf.io.gfile.GFile(archive_path, "wb") as f:
    with tarfile.open(mode="w:gz", fileobj=f) as tar:
      with tf.io.gfile.GFile(file_path, "rb") as archived_file:
        tar_info = tarfile.TarInfo(name=os.path.basename(file_path))
        tar_info.size = archived_file.size()
        tar.addfile(tar_info, fileobj=archived_file)


def compress_local_directory_to_archive(local_directory: str,
                                        archive_path: str) -> None:
  """Compresses a local directory to a tar.gz archive."""
  with tf.io.gfile.GFile(archive_path, "wb") as f:
    with tarfile.open(mode="w:gz", fileobj=f) as tar:
      tar.add(local_directory, arcname="/")
