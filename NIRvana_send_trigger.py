"""Acquire NIRvana spectra using the lab's JSON-over-TCP/CSV protocol.

trigger(params) preserves the lab-mates' call and return convention:
it sends the supplied dictionary unchanged and returns
(wavelengths, frames_by_wavelength_intensities) as NumPy arrays.
"""

import json
import socket
from io import StringIO

import numpy as np
import pandas as pd


DEVICE_IP = "10.236.10.143"
PORT = 65432
CONNECT_TIMEOUT_S = 10
ACQUISITION_TIMEOUT_S = 10 * 60


class NIRvanaAcquisitionError(RuntimeError):
    """A connection, transmission, or CSV response failure."""


def _parse_spectrum_csv(data):
    df = pd.read_csv(StringIO(data), engine="python")
    required = {"Frame", "Wavelength", "Intensity"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Expected CSV columns {sorted(required)}; received {list(df.columns)}. "
            f"Response preview: {data[:200]!r}"
        )
    if df.empty:
        raise ValueError("CSV contains a header but no spectrum data.")

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="raise")
    if not np.isfinite(df[list(required)].to_numpy(dtype=float)).all():
        raise ValueError("CSV contains missing or non-finite frame, wavelength or intensity values.")
    if (df["Wavelength"] <= 0).any():
        raise ValueError("CSV contains non-positive wavelengths.")

    frames = np.sort(df["Frame"].unique())
    if not np.array_equal(frames, np.arange(1, len(frames) + 1)):
        raise ValueError(f"Expected consecutive frames starting at 1; received {frames.tolist()}.")
    if df.duplicated(["Frame", "Wavelength"]).any():
        raise ValueError("CSV contains duplicate wavelengths within a frame.")

    df = df.sort_values(["Frame", "Wavelength"])
    wavelengths = df.loc[df["Frame"] == 1, "Wavelength"].to_numpy()
    intensities = []
    for _, frame in df.groupby("Frame", sort=True):
        if not np.array_equal(frame["Wavelength"].to_numpy(), wavelengths):
            raise ValueError("CSV frames have different wavelength grids or missing pixels.")
        intensities.append(frame["Intensity"].to_numpy())
    return wavelengths, np.stack(intensities)


def trigger(params, *, device_ip=DEVICE_IP, port=PORT, on_socket=None,
            acquisition_timeout_s=ACQUISITION_TIMEOUT_S):
    """Send one request and return (wavelengths, intensities[frame, pixel]).

    The caller supplies the exact server request, including the exposure-time
    unit. on_socket lets the GUI close an in-progress request on Reset.
    """
    if "exposure_time" not in params:
        raise ValueError("NIRvana request must include 'exposure_time'.")
    message = json.dumps(params)
    phase = "connecting"
    received_bytes = 0
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if on_socket is not None:
                on_socket(sock)
            sock.settimeout(CONNECT_TIMEOUT_S)
            sock.connect((device_ip, port))
            sock.settimeout(acquisition_timeout_s)
            phase = "sending acquisition parameters"
            sock.sendall(message.encode("utf-8"))
            phase = "receiving CSV"
            chunks = []
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
                received_bytes += len(chunk)

        if not chunks:
            raise ValueError(
                "Server closed without sending spectrum data. Check the "
                "NIRvana_device_server.py console for 'Error handling connection'."
            )
        phase = "parsing CSV"
        data = b"".join(chunks).decode("utf-8")
        if not data.strip():
            raise ValueError("NIRvana returned only whitespace.")
        wavelengths, intensities = _parse_spectrum_csv(data)
        if "num_frames" in params and len(intensities) != int(params["num_frames"]):
            raise ValueError(
                f"Requested {params['num_frames']} frames; received {len(intensities)}."
            )
        return wavelengths, intensities
    except Exception as exc:
        raise NIRvanaAcquisitionError(
            f"NIRvana {device_ip}:{port} failed while {phase} "
            f"({received_bytes} bytes received, request={params!r}): "
            f"{type(exc).__name__}: {exc}"
        ) from exc
