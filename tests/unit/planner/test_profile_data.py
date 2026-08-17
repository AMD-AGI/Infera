###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Validation of the profiling JSON.

A malformed profiling file must fail loudly at load time. If it slips through,
the damage shows up as a nonsensical replica count hours later, with nothing in
the logs pointing back at the data.
"""

from __future__ import annotations

import json

import pytest

from infera.planner.profile_data import ProfileDataError, load_profile_data, parse_profile_data

from .conftest import flat_profile_dict


class TestParseProfileData:
    def test_accepts_a_well_formed_document(self):
        data = parse_profile_data(flat_profile_dict())
        assert data.prefill.isl.tolist() == [1000, 2000]
        assert data.decode.itl_ms.shape == (2, 3)
        assert data.decode.max_kv_tokens == 100_000
        assert data.prefill_engine_num_gpu == 1

    def test_gpu_counts_default_to_one_when_absent(self):
        raw = flat_profile_dict()
        del raw["prefill_engine_num_gpu"]
        del raw["decode_engine_num_gpu"]
        data = parse_profile_data(raw)
        assert data.prefill_engine_num_gpu == 1
        assert data.decode_engine_num_gpu == 1

    @pytest.mark.parametrize("section", ["prefill", "decode"])
    def test_missing_section_is_rejected(self, section):
        raw = flat_profile_dict()
        del raw[section]
        with pytest.raises(ProfileDataError, match=section):
            parse_profile_data(raw)

    def test_mismatched_prefill_series_lengths_are_rejected(self):
        raw = flat_profile_dict()
        raw["prefill"]["ttft_ms"] = [100.0]
        with pytest.raises(ProfileDataError, match="same length"):
            parse_profile_data(raw)

    def test_decode_grid_shape_must_match_its_axes(self):
        raw = flat_profile_dict()
        # 2 columns where the kv_usage axis declares 3.
        raw["decode"]["itl_ms"] = [[10.0, 10.0], [10.0, 10.0]]
        with pytest.raises(ProfileDataError, match="expected"):
            parse_profile_data(raw)

    def test_unsorted_axis_is_rejected(self):
        raw = flat_profile_dict()
        raw["decode"]["kv_usage"] = [0.9, 0.5, 0.1]
        with pytest.raises(ProfileDataError, match="ascending"):
            parse_profile_data(raw)

    def test_non_positive_throughput_is_rejected(self):
        # A zero would become a division by zero in the replica calculation.
        raw = flat_profile_dict()
        raw["prefill"]["thpt_per_gpu"] = [10000.0, 0.0]
        with pytest.raises(ProfileDataError, match="positive"):
            parse_profile_data(raw)

    def test_missing_max_kv_tokens_is_rejected(self):
        raw = flat_profile_dict()
        del raw["decode"]["max_kv_tokens"]
        with pytest.raises(ProfileDataError, match="max_kv_tokens"):
            parse_profile_data(raw)


class TestLoadProfileData:
    def test_round_trips_through_a_file(self, tmp_path):
        path = tmp_path / "profile.json"
        path.write_text(json.dumps(flat_profile_dict()), encoding="utf-8")
        assert load_profile_data(path).decode.max_kv_tokens == 100_000

    def test_missing_file_explains_how_to_produce_one(self, tmp_path):
        with pytest.raises(ProfileDataError, match="profiling"):
            load_profile_data(tmp_path / "absent.json")

    def test_malformed_json_is_reported_as_such(self, tmp_path):
        path = tmp_path / "profile.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ProfileDataError, match="not valid JSON"):
            load_profile_data(path)
