"""Gate current/ramp protection and the SMU's supported voltage range.

Checks apply to physical SMU channels, after any plotted-coordinate transform.
Current and ramp settings must not be overridden by saved measurement metadata.
"""
import math

GATE_CURRENT_COMPLIANCE_A = 100e-9
GATE_MIN_CURRENT_COMPLIANCE_A = 1e-9
GATE_MAX_CURRENT_COMPLIANCE_A = 3.0
# Instrument range, rather than an additional sample-specific software cap.
GATE_MAX_ABS_VOLTAGE_V = 210
GATE_MAX_RAMP_STEP_V = 0.05
DEFAULT_GATE_RAMP_RATE_V_PER_S = 0.1


class GateRampCancelled(RuntimeError):
    """The ramp stopped at its last intermediate setpoint, not its target."""


def validate_current_compliance(value):
    """Validate a user-selected DC current limit before any instrument command."""
    current = float(value)
    if not math.isfinite(current) or not GATE_MIN_CURRENT_COMPLIANCE_A <= current <= GATE_MAX_CURRENT_COMPLIANCE_A:
        raise ValueError('Current compliance must be finite and between 1 nA and 3 A.')
    return current


def validate_gate_voltage(value, channel=None):
    """Return a finite physical gate voltage or fail before hardware access."""
    voltage = float(value)
    label = f"Channel {channel}" if channel is not None else "Gate channel"
    if not math.isfinite(voltage) or abs(voltage) > GATE_MAX_ABS_VOLTAGE_V:
        raise ValueError(f"{label}: voltage must be finite and within the SMU "
                         f"range of +/-{GATE_MAX_ABS_VOLTAGE_V:g} V.")
    return voltage


def validate_gate_targets(channel_1_values, channel_2_values):
    """Validate both complete physical-voltage grids before starting a sweep."""
    import numpy as np

    for channel, values in enumerate((channel_1_values, channel_2_values), 1):
        array = np.asarray(values, dtype=float)
        invalid = ~np.isfinite(array) | (np.abs(array) > GATE_MAX_ABS_VOLTAGE_V)
        if np.any(invalid):
            validate_gate_voltage(array[invalid].flat[0], channel)


def validate_positive_ramp_value(value, label):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be finite and greater than zero; "
                         "an invalid ramp cannot become a voltage jump.")
    return number


def validate_map_gate_targets(params, x_values, y_values):
    """Preflight every direct and forward-transformed map gate target.

    Map modules historically use a forward transform. Keep that convention and
    validate intermediate direct writes too when a virtual axis also appears.
    Return the physical channels used, so nongate maps need no connected SMU.
    """
    import numpy as np

    px, py = params['x_param'], params['y_param']
    x = np.asarray(x_values, dtype=float)[None, :]
    y = np.asarray(y_values, dtype=float)[:, None]
    channels = set()
    for axis, values in ((px, x), (py, y)):
        if axis in ('V1', 'V2'):
            channel = int(axis[-1])
            validate_gate_targets(values if channel == 1 else 0,
                                  values if channel == 2 else 0)
            channels.add(channel)
    c1_name, c2_name = params.get('c1_name', 'Vc1'), params.get('c2_name', 'Vc2')
    if px in ('Vc1', 'Vc2', c1_name, c2_name) or py in ('Vc1', 'Vc2', c1_name, c2_name):
        c1 = c2 = 0.0
        for axis, values in ((px, x), (py, y)):
            if axis in ('Vc1', c1_name):
                c1 = values
            elif axis in ('Vc2', c2_name):
                c2 = values
        v1 = params.get('coeff_a', 1.0) * c1 + params.get('coeff_b', 0.0) * c2
        v2 = params.get('coeff_c', 0.0) * c1 + params.get('coeff_d', 1.0) * c2
        validate_gate_targets(v1, v2)
        channels.update((1, 2))
    return channels
