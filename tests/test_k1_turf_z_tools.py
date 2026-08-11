import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from humanoidverse.tools.eval_getup_z_turf import load_specs

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SWEEP_PATH = _REPO_ROOT / "tools" / "build_k1_turf_z_sweep.py"
_SWEEP_SPEC = importlib.util.spec_from_file_location("build_k1_turf_z_sweep", _SWEEP_PATH)
assert _SWEEP_SPEC is not None and _SWEEP_SPEC.loader is not None
_SWEEP_MODULE = importlib.util.module_from_spec(_SWEEP_SPEC)
_SWEEP_SPEC.loader.exec_module(_SWEEP_MODULE)
slerp = _SWEEP_MODULE.slerp

_BANK_PATH = _REPO_ROOT / "tools" / "build_k1_turf_z_bank.py"
_BANK_SPEC = importlib.util.spec_from_file_location("build_k1_turf_z_bank", _BANK_PATH)
assert _BANK_SPEC is not None and _BANK_SPEC.loader is not None
_BANK_MODULE = importlib.util.module_from_spec(_BANK_SPEC)
_BANK_SPEC.loader.exec_module(_BANK_MODULE)
build_bank = _BANK_MODULE.build_bank


def test_slerp_preserves_radius_and_endpoints():
    a = np.array([4.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 4.0, 0.0], dtype=np.float32)

    middle = slerp(a, b, 0.5)

    np.testing.assert_allclose(slerp(a, b, 1.0), b, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(middle), 4.0, atol=1e-6)
    np.testing.assert_allclose(middle[:2], np.sqrt(8.0), atol=1e-6)


def test_slerp_rejects_mismatched_endpoint_norms():
    with pytest.raises(ValueError, match="same non-zero norm"):
        slerp(np.array([1.0, 0.0]), np.array([0.0, 2.0]), 0.5)


def test_load_specs_keeps_requested_order(tmp_path):
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps([{"name": "a"}, {"name": "b"}]))

    assert [spec["name"] for spec in load_specs(path, ["b", "a"])] == ["b", "a"]
    with pytest.raises(ValueError, match="Unknown candidates"):
        load_specs(path, ["missing"])


def test_load_specs_rejects_non_list(tmp_path):
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps({"name": "a"}))

    with pytest.raises(ValueError, match="JSON list"):
        load_specs(path, None)


def test_build_turf_bank_keeps_rollback_and_aliases_selected_z(tmp_path):
    output = tmp_path / "z_bank.npz"
    build_bank(_REPO_ROOT / "docs" / "k1_getup_opt_z.json", output)

    with np.load(output) as bank:
        assert set(bank.files) == {"z/getup_opt", "z/getup_turf", "z/standing_pooled"}
        np.testing.assert_array_equal(bank["z/getup_turf"], bank["z/standing_pooled"])
        np.testing.assert_allclose(np.linalg.norm(bank["z/getup_opt"]), 16.0, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(bank["z/getup_turf"]), 16.0, atol=1e-6)
