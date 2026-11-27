# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""aiter backends of MSLK attention ops on ROCm. Importing them does not import
aiter; the dispatch modules (``mslk/attention/dispatch/``) choose when they run."""
