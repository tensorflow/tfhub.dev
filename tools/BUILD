load("//:tfhub_dev.bzl", "pytype_strict_binary", "pytype_strict_library", "pytype_strict_test")
load("//tools/build_defs/testing:bzl_library.bzl", "bzl_library")  # buildifier: disable=load-on-top

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
licenses(["notice"])

pytype_strict_binary(
    name = "validator",
    srcs = ["validator.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":validator_lib"],
)

pytype_strict_test(
    name = "test_tag_configs",
    srcs = ["test_tag_configs.py"],
    data = [
        "//third_party/py/tfhub_dev:tag_config_files",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tags_validator",
        "@org_tensorflow_tfhub_dev//tools:expect_tensorflow_installed",
    ],
)

pytype_strict_binary(
    name = "tags_validator_main",
    srcs = ["tags_validator_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tags_validator",
        "@io_abseil_py//absl:app",
        "@io_abseil_py//absl/logging",
    ],
)

pytype_strict_library(
    name = "validator_lib",
    srcs = [
        "validator.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":filesystem_utils",
        "@io_abseil_py//absl:app",
        "@io_abseil_py//absl/logging",
        "@org_tensorflow_tfhub_dev//tools:expect_tensorflow_installed",
        "@org_tensorflow_tfhub_dev//tools:expect_tensorflow_hub_installed",
        "@org_tensorflow//tensorflow/python/saved_model:loader",
    ],
)

pytype_strict_library(
    name = "tags_validator",
    srcs = [
        "tags_validator.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":filesystem_utils",
        "@io_abseil_py//absl/logging",
        "//third_party/py/ruamel",
        "@org_tensorflow_tfhub_dev//tools:expect_tensorflow_installed",
        "@org_tensorflow_tfhub_dev//tools:expect_tensorflow_hub_installed",
        "//third_party/py/yaml",
    ],
)

pytype_strict_library(
    name = "filesystem_utils",
    srcs = ["filesystem_utils.py"],
    srcs_version = "PY3",
    deps = ["@org_tensorflow_tfhub_dev//tools:expect_tensorflow_installed"],
)

pytype_strict_test(
    name = "validator_test",
    srcs = ["validator_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":validator_lib",
        "@py_mock//py/mock",
        "@org_tensorflow_tfhub_dev//tools:expect_tensorflow_installed",
    ],
)

pytype_strict_test(
    name = "filesystem_utils_test",
    srcs = ["filesystem_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":filesystem_utils",
        "@org_tensorflow_tfhub_dev//tools:expect_tensorflow_installed",
    ],
)

pytype_strict_test(
    name = "tags_validator_test",
    srcs = ["tags_validator_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tags_validator",
        "@org_tensorflow_tfhub_dev//tools:expect_tensorflow_installed",
    ],
)

# We expect TensorFlow to already be installed on the system, e.g. via
# `pip install tensorflow`
py_library(
    name = "expect_tensorflow_installed",
)

# We expect TF Hub to already be installed on the system, e.g. via
# `pip install tensorflow-hub`
py_library(
    name = "expect_tensorflow_hub_installed",
)

bzl_library(
    name = "tfhub_dev_bzl",
    srcs = ["tfhub_dev.bzl"],
    visibility = ["//visibility:public"],
)
