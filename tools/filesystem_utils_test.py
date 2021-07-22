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
"""Tests for tfhub_dev.tools.filesystem_utils."""

import os
import pathlib
import tarfile
import tensorflow as tf
import filesystem_utils


class FilesystemUtilsTest(tf.test.TestCase):

  def test_recursive_list_dir_with_empty_directory(self):
    root_dir = self.create_tempdir()
    files = list(filesystem_utils.recursive_list_dir(root_dir))
    self.assertEmpty(files)

  def test_recursive_list_dir_with_empty_nested_directories(self):
    root_dir = self.create_tempdir()
    pathlib.Path(os.path.join(root_dir, "a", "model1")).mkdir(parents=True)
    pathlib.Path(os.path.join(root_dir, "a", "model2")).mkdir(parents=True)
    files = list(filesystem_utils.recursive_list_dir(root_dir))
    self.assertEmpty(files)

  def test_recursive_list_dir_with_nested_files(self):
    root_dir = self.create_tempdir()
    file_paths = [
        os.path.join(root_dir, *p)
        for p in [("a", "1.md"), ("a", "2.md"), ("a", "b", "1.md")]
    ]
    for path in file_paths:
      self.create_tempfile(path)
    actual_files = list(filesystem_utils.recursive_list_dir(root_dir))
    self.assertCountEqual(actual_files, file_paths)

  def test_get_content_empty_file(self):
    temp_file = self.create_tempfile()
    self.assertEmpty(filesystem_utils.get_content(temp_file))

  def test_get_content_populated_file(self):
    content = "# Module"
    temp_file = self.create_tempfile(content=content)
    self.assertEqual(content, filesystem_utils.get_content(temp_file))

  def test_create_archive_with_valid_file_succeeds(self):
    file_name = "my_file"
    file_content = "content"
    file_path = self.create_tempfile(file_path=file_name, content=file_content)
    archive_path = self.create_tempfile().full_path

    filesystem_utils.create_archive(archive_path, file_path)

    with tf.io.gfile.GFile(archive_path, "rb") as archive_file:
      with tarfile.open(fileobj=archive_file, mode="r:gz") as tar_file:
        self.assertEqual(tar_file.getnames(), [file_name])
        member = tar_file.extractfile(file_name)
        self.assertIsNotNone(member)
        # pytype: disable=attribute-error, member is not None so .read() is
        # safe.
        self.assertEqual(member.read(), b"content")
        # pytype: enable=attribute-error

  def test_compress_local_directory_to_archive_from_valid_directory(self):
    temp_dir = self.create_tempdir()
    file_name1 = "file_name1"
    file_name2 = "file_name2"
    content1 = "content1"
    content2 = "content2"
    temp_dir.create_file(file_path=file_name1, content=content1)
    temp_dir.create_file(file_path=file_name2, content=content2)
    archive_path = self.create_tempfile().full_path

    filesystem_utils.compress_local_directory_to_archive(
        temp_dir.full_path, archive_path)

    with tf.io.gfile.GFile(archive_path, "rb") as archive_file:
      with tarfile.open(fileobj=archive_file, mode="r:gz") as tar_file:
        files = filter(lambda member: member.isfile(), tar_file.getmembers())
        self.assertCountEqual([file.name for file in files],
                              [file_name1, file_name2])
        for file_name, expected_content in zip([file_name1, file_name2],
                                               [content1, content2]):
          member = tar_file.extractfile(file_name)
          self.assertIsNotNone(member)
          # pytype: disable=attribute-error, member is not None so .read() is
          # safe.
          self.assertEqual(member.read(), expected_content.encode("utf-8"))
          # pytype: enable=attribute-error


if __name__ == "__main__":
  tf.test.main()
