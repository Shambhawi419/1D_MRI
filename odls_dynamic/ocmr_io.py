"""
Reader for OCMR's ISMRMRD-format .h5 files, using h5py only (no ismrmrd
package needed on Colab).

Mirrors the official reader (github.com/MRIOSU/OCMR, Python/read_ocmr.py)
and returns the same 9D k-space array

    [kx, ky, kz, coil, phase(time), set, slice, rep, avg]

but reads every acquisition in one vectorised HDF5 read instead of one
`dset.read_acquisition()` call per line, which matters when streaming
dozens of files through a free Colab session. Verified against the official
reader on a real OCMR file (fs_0152_0_55T.h5): identical arrays.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, Tuple

import h5py
import numpy as np

# ISMRMRD flag numbers are 1-based bit positions (ismrmrd.ACQ_IS_NOISE_MEASUREMENT == 19).
ACQ_IS_NOISE_MEASUREMENT = 19


def _find(el, path: str):
    """Namespace-agnostic ElementTree lookup ('a/b/c')."""
    for part in path.split("/"):
        if el is None:
            return None
        el = el.find("{*}" + part)
    return el


def _int(el, path: str, default: int) -> int:
    node = _find(el, path)
    return int(node.text) if node is not None and node.text is not None else default


def _float(el, path: str, default: float = float("nan")) -> float:
    node = _find(el, path)
    return float(node.text) if node is not None and node.text is not None else default


def read_header(path: str) -> Dict:
    """Parses the ISMRMRD XML header into the handful of values the
    preprocessing needs. Cheap: only touches the 'dataset/xml' entry."""
    with h5py.File(path, "r") as h:
        raw = h["dataset/xml"][0]
    xml = raw.decode() if isinstance(raw, bytes) else str(raw)
    root = ET.fromstring(xml)
    enc = _find(root, "encoding")

    pilot_tone = 0
    for u in root.iter("{*}userParameterLong"):
        name, value = _find(u, "name"), _find(u, "value")
        if name is not None and name.text == "PilotTone" and value is not None:
            pilot_tone = int(value.text)

    return {
        "eNx": _int(enc, "encodedSpace/matrixSize/x", 0),
        "eNz": _int(enc, "encodedSpace/matrixSize/z", 1),
        # Like read_ocmr.py: ky size from the encoding limits, not the
        # (possibly zero-padded) encoded matrix size.
        "eNy": _int(enc, "encodingLimits/kspace_encoding_step_1/maximum", -1) + 1,
        "ky_center": _int(enc, "encodingLimits/kspace_encoding_step_1/center", -1),
        "n_coils": _int(root, "acquisitionSystemInformation/receiverChannels", 0),
        "n_phases": _int(enc, "encodingLimits/phase/maximum", 0) + 1,
        "n_sets": _int(enc, "encodingLimits/set/maximum", 0) + 1,
        "n_slices": _int(enc, "encodingLimits/slice/maximum", 0) + 1,
        "n_reps": _int(enc, "encodingLimits/repetition/maximum", 0) + 1,
        "n_avgs": _int(enc, "encodingLimits/average/maximum", 0) + 1,
        "fov_enc_x": _float(enc, "encodedSpace/fieldOfView_mm/x"),
        "fov_rec_x": _float(enc, "reconSpace/fieldOfView_mm/x"),
        "fov_enc_y": _float(enc, "encodedSpace/fieldOfView_mm/y"),
        "field_T": _float(root, "acquisitionSystemInformation/systemFieldStrength_T"),
        "pilot_tone": pilot_tone,
    }


def read_ocmr(path: str) -> Tuple[np.ndarray, Dict]:
    """Returns (kdata, header): kdata is complex64 with shape
    [kx, ky, kz, coil, phase, set, slice, rep, avg], as read_ocmr.py."""
    hdr = read_header(path)
    with h5py.File(path, "r") as h:
        acqs = h["dataset/data"][()]  # structured array: head / traj / data

    head = acqs["head"]
    flags = head["flags"].astype(np.uint64)
    is_noise = (flags & np.uint64(1 << (ACQ_IS_NOISE_MEASUREMENT - 1))) != 0
    imaging = np.flatnonzero(~is_noise)
    if imaging.size == 0:
        raise ValueError(f"{path}: no imaging acquisitions found")

    first = head[imaging[0]]
    n_samples = int(first["number_of_samples"])
    eNx = hdr["eNx"]
    # Asymmetric echo: the readout starts late, so pad the front (read_ocmr.py logic).
    kx_prezp = eNx - n_samples if int(first["center_sample"]) * 2 < eNx else 0

    kdata = np.zeros((eNx, hdr["eNy"], hdr["eNz"], hdr["n_coils"], hdr["n_phases"],
                      hdr["n_sets"], hdr["n_slices"], hdr["n_reps"], hdr["n_avgs"]),
                     dtype=np.complex64)

    idx = head["idx"]
    for a in imaging:
        h_a = head[a]
        n_ch, n_s = int(h_a["active_channels"]), int(h_a["number_of_samples"])
        data = np.asarray(acqs["data"][a], dtype=np.float32).view(np.complex64).reshape(n_ch, n_s)
        if hdr["pilot_tone"] == 1:
            data = data.copy()
            data[:, [0, 1, 2, n_s - 1]] = 0  # exclude pilot-tone artefact (read_ocmr.py)
        i = idx[a]
        kdata[kx_prezp:kx_prezp + n_s, i["kspace_encode_step_1"], i["kspace_encode_step_2"], :,
              i["phase"], i["set"], i["slice"], i["repetition"], i["average"]] = data.T

    hdr["kx_prezp"] = kx_prezp
    return kdata, hdr
