"""Capacitance-bridge balancing, electrode coordinates and voltage protection.

Ported from the qcodes-qpg ``capacitance_measurement.py`` script and its
``include/capacitance_bridge.py`` (``CapacitanceBridgeSR830Lockin``). The module
has no Qt dependency so the algorithm can be tested offline and reused by
scripts. Coordinate conventions are exactly those of the original code:

    n0 = (1 + delta) (Vt - Vs) + (1 - delta) (Vb - Vs)   normalized density, V
    p0 = (1 + delta) (Vt - Vs) - (1 - delta) (Vb - Vs)   normalized field, V
    delta = (Ct - Cb) / (Ct + Cb)                        gate asymmetry

One electrode (normally the bridge/measurement node) is held at ``v_fixed`` and
the other two follow from (n0, p0). The bridge reports the sample capacitance
and dissipation as ratios to the reference capacitor, ``C/C_ref`` and
``D/C_ref``; multiply by ``C_ref`` for farads.
"""
from dataclasses import dataclass, field
import math

import numpy as np

ELECTRODES = ("top", "bottom", "sample")
ELECTRODE_LABELS = {"top": "Top gate", "bottom": "Bottom gate", "sample": "Sample"}
ELECTRODE_SYMBOLS = {"top": "V_top", "bottom": "V_bottom", "sample": "V_sample"}

# Measurement presets: AC excitation and measurement node are wiring facts that
# are recorded with the data; the preset only chooses the default DC-fixed
# electrode (the measurement node, whose DC level the bridge amplifier sets).
MEASUREMENT_TYPES = {
    "penetration": {
        "label": "Penetration field capacitance C_P",
        "symbol": "C_P",
        "fixed": "bottom",
        "excited": "Top gate",
        "measured": "Bottom gate",
    },
    "top_gate": {
        "label": "Top-gate capacitance C_T",
        "symbol": "C_T",
        "fixed": "sample",
        "excited": "Top gate",
        "measured": "Sample",
    },
    "bottom_gate": {
        "label": "Bottom-gate capacitance C_B",
        "symbol": "C_B",
        "fixed": "sample",
        "excited": "Bottom gate",
        "measured": "Sample",
    },
    "symmetric": {
        "label": "Symmetric capacitance C_S",
        "symbol": "C_S",
        "fixed": "sample",
        "excited": "Both gates",
        "measured": "Sample",
    },
    "custom": {
        "label": "Custom",
        "symbol": "C",
        "fixed": "sample",
        "excited": "(record in comment)",
        "measured": "(record in comment)",
    },
}

COORDINATE_MODES = ("density", "direct")

# Siglent SDG2042X amplitude range (Vpp, high-Z) used by the balance check.
AWG_MIN_AMPLITUDE_VPP = 2e-3


class BalanceError(RuntimeError):
    """The bridge could not be balanced; no capacitance calibration exists."""


class BalanceCancelled(RuntimeError):
    """Stop was requested while the bridge was being balanced."""


# ---------------------------------------------------------------------------
# Electrode coordinates
# ---------------------------------------------------------------------------

def validate_delta(delta):
    delta = float(delta)
    if not math.isfinite(delta) or abs(delta) >= 1:
        raise ValueError("Gate asymmetry delta must be finite with |delta| < 1.")
    return delta


def validate_electrode(name):
    if name not in ELECTRODES:
        raise ValueError(f"Unknown electrode {name!r}; expected one of {ELECTRODES}.")
    return name


def swept_electrodes(fixed):
    """The two electrodes that follow the sweep, in the original script's order.

    vb fixed -> (top, sample), vt fixed -> (bottom, sample), vs fixed -> (top, bottom).
    """
    return {"bottom": ("top", "sample"), "top": ("bottom", "sample"),
            "sample": ("top", "bottom")}[validate_electrode(fixed)]


def electrode_voltages(n0, p0, *, delta, v_fixed, fixed):
    """Return (V_top, V_bottom, V_sample) for normalized density/field coordinates."""
    delta = validate_delta(delta)
    v_fixed = float(v_fixed)
    n0 = np.asarray(n0, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    top_minus_sample = (n0 + p0) / (2.0 * (1.0 + delta))
    bottom_minus_sample = (n0 - p0) / (2.0 * (1.0 - delta))
    fixed = validate_electrode(fixed)
    if fixed == "sample":
        vs = np.full(np.broadcast(n0, p0).shape, v_fixed)
        vt = vs + top_minus_sample
        vb = vs + bottom_minus_sample
    elif fixed == "bottom":
        vb = np.full(np.broadcast(n0, p0).shape, v_fixed)
        vs = vb - bottom_minus_sample
        vt = vs + top_minus_sample
    else:
        vt = np.full(np.broadcast(n0, p0).shape, v_fixed)
        vs = vt - top_minus_sample
        vb = vs + bottom_minus_sample
    return vt, vb, vs


def density_field(vt, vb, vs, *, delta):
    """Inverse transform: normalized (n0, p0) for electrode voltages."""
    delta = validate_delta(delta)
    top = np.asarray(vt, dtype=float) - np.asarray(vs, dtype=float)
    bottom = np.asarray(vb, dtype=float) - np.asarray(vs, dtype=float)
    return (1 + delta) * top + (1 - delta) * bottom, (1 + delta) * top - (1 - delta) * bottom


def direct_voltages(x, y, *, v_fixed, fixed):
    """Direct mode: x and y are the voltages of the two swept electrodes."""
    first, second = swept_electrodes(fixed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    shape = np.broadcast(x, y).shape
    values = {fixed: np.full(shape, float(v_fixed)),
              first: np.broadcast_to(x, shape).astype(float),
              second: np.broadcast_to(y, shape).astype(float)}
    return values["top"], values["bottom"], values["sample"]


def coordinate_names(mode, fixed):
    if mode == "density":
        return "n0", "p0"
    if mode == "direct":
        return tuple(ELECTRODE_SYMBOLS[name] for name in swept_electrodes(fixed))
    raise ValueError(f"Unknown coordinate mode {mode!r}.")


def transform_description(mode, fixed, delta, v_fixed):
    """One-line human-readable statement of the active transform."""
    delta = validate_delta(delta)
    fixed_text = f"{ELECTRODE_SYMBOLS[fixed]} = {v_fixed:g} V (fixed)"
    if mode == "direct":
        a, b = swept_electrodes(fixed)
        return f"{ELECTRODE_SYMBOLS[a]} = x,  {ELECTRODE_SYMBOLS[b]} = y,  {fixed_text}"
    return (f"V_top - V_sample = (n0 + p0) / {2 * (1 + delta):.4g},  "
            f"V_bottom - V_sample = (n0 - p0) / {2 * (1 - delta):.4g},  {fixed_text}")


def axis_values(start, end, points):
    points = int(points)
    start, end = float(start), float(end)
    if points < 1:
        raise ValueError("Each sweep axis needs at least one point.")
    if not (math.isfinite(start) and math.isfinite(end)):
        raise ValueError("Sweep start and end values must be finite.")
    return np.linspace(start, end, points) if points > 1 else np.array([start])


@dataclass
class SweepPlan:
    """Complete target grid, in row = y, column = x layout, plus acquisition order."""
    mode: str
    fixed: str
    delta: float
    v_fixed: float
    x: np.ndarray
    y: np.ndarray
    vt: np.ndarray
    vb: np.ndarray
    vs: np.ndarray
    n0: np.ndarray
    p0: np.ndarray
    fast_axis: str
    serpentine: bool
    order: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=int))

    @property
    def shape(self):
        return len(self.y), len(self.x)

    @property
    def names(self):
        return coordinate_names(self.mode, self.fixed)

    def targets(self, iy, ix):
        return {"top": float(self.vt[iy, ix]), "bottom": float(self.vb[iy, ix]),
                "sample": float(self.vs[iy, ix])}

    def ordered_targets(self):
        """(N, 3) array of (V_top, V_bottom, V_sample) in acquisition order."""
        iy, ix = self.order[:, 0], self.order[:, 1]
        return np.column_stack([self.vt[iy, ix], self.vb[iy, ix], self.vs[iy, ix]])

    def lines(self):
        """Acquisition order grouped by slow-axis line: [(slow_index, [(iy, ix), ...]), ...]."""
        if not len(self.order):
            return []
        slow = self.order[:, 0] if self.fast_axis == "x" else self.order[:, 1]
        breaks = np.flatnonzero(np.diff(slow)) + 1
        return [(int(chunk[0, 0] if self.fast_axis == "x" else chunk[0, 1]),
                 [(int(iy), int(ix)) for iy, ix in chunk])
                for chunk in np.split(self.order, breaks)]


def build_sweep_plan(*, mode, fixed, delta, v_fixed, x_start, x_end, x_points,
                     y_start, y_end, y_points, fast_axis="x", serpentine=False):
    """Build every physical target before any hardware access."""
    if mode not in COORDINATE_MODES:
        raise ValueError(f"Unknown coordinate mode {mode!r}.")
    if fast_axis not in ("x", "y"):
        raise ValueError("Fast axis must be 'x' or 'y'.")
    fixed = validate_electrode(fixed)
    delta = validate_delta(delta)
    v_fixed = float(v_fixed)
    if not math.isfinite(v_fixed):
        raise ValueError("The fixed electrode voltage must be finite.")
    x = axis_values(x_start, x_end, x_points)
    y = axis_values(y_start, y_end, y_points)
    if x.size * y.size > 1_000_000:
        raise ValueError("A capacitance map is limited to 1,000,000 points.")
    xx, yy = np.meshgrid(x, y)
    if mode == "density":
        vt, vb, vs = electrode_voltages(xx, yy, delta=delta, v_fixed=v_fixed, fixed=fixed)
        n0, p0 = xx.copy(), yy.copy()
    else:
        vt, vb, vs = direct_voltages(xx, yy, v_fixed=v_fixed, fixed=fixed)
        n0, p0 = density_field(vt, vb, vs, delta=delta)
    ny, nx = len(y), len(x)
    if fast_axis == "x":
        iy, ix = np.divmod(np.arange(ny * nx), nx)
        if serpentine:
            ix = np.where(iy % 2 == 1, nx - 1 - ix, ix)
    else:
        ix, iy = np.divmod(np.arange(ny * nx), ny)
        if serpentine:
            iy = np.where(ix % 2 == 1, ny - 1 - iy, iy)
    order = np.column_stack([iy, ix]).astype(int)
    return SweepPlan(mode, fixed, delta, v_fixed, x, y, np.array(vt), np.array(vb),
                     np.array(vs), np.array(n0), np.array(p0), fast_axis, bool(serpentine), order)


# ---------------------------------------------------------------------------
# Voltage protection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ElectrodeLimits:
    """Code/GUI limits applied to every physical target, after the transform.

    The dielectric limits bound the voltage across each gate dielectric
    (|V_top - V_sample| and |V_bottom - V_sample|). All three are convex, so
    straight-line (coordinated) ramps between valid targets stay valid.
    """
    max_abs: float = 10.0
    max_top_sample: float = 10.0
    max_bottom_sample: float = 10.0

    def __post_init__(self):
        for name in ("max_abs", "max_top_sample", "max_bottom_sample"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Electrode voltage limits must be finite and positive.")

    def check(self, vt, vb, vs, *, output_caps=None, context="Target"):
        """Raise ValueError naming the first violation; never clip."""
        values = {"top": np.asarray(vt, dtype=float), "bottom": np.asarray(vb, dtype=float),
                  "sample": np.asarray(vs, dtype=float)}
        caps = dict(output_caps or {})
        for name, array in values.items():
            cap = min(self.max_abs, caps.get(name, math.inf))
            bad = ~np.isfinite(array) | (np.abs(array) > cap + 1e-12)
            if np.any(bad):
                value = array[bad].flat[0]
                raise ValueError(f"{context}: {ELECTRODE_LABELS[name]} would reach {value:.6g} V, "
                                 f"outside the ±{cap:g} V limit. Nothing was moved.")
        for name, limit in (("top", self.max_top_sample), ("bottom", self.max_bottom_sample)):
            difference = values[name] - values["sample"]
            bad = np.abs(difference) > limit + 1e-12
            if np.any(bad):
                value = difference[bad].flat[0]
                raise ValueError(f"{context}: |{ELECTRODE_SYMBOLS[name]} - V_sample| would reach "
                                 f"{abs(value):.6g} V, above the {limit:g} V dielectric limit. "
                                 "Nothing was moved.")


def plan_summary(plan):
    """Extremes over the whole sweep, for the pre-start preview."""
    def span(array):
        return float(np.nanmin(array)), float(np.nanmax(array))
    return {
        "top": span(plan.vt), "bottom": span(plan.vb), "sample": span(plan.vs),
        "top_sample": float(np.nanmax(np.abs(plan.vt - plan.vs))),
        "bottom_sample": float(np.nanmax(np.abs(plan.vb - plan.vs))),
        "points": int(plan.vt.size),
    }


def interpolate_path(start, target, max_step):
    """Straight-line electrode path with no component changing more than max_step."""
    max_step = float(max_step)
    if not math.isfinite(max_step) or max_step <= 0:
        raise ValueError("The coordinated ramp step must be positive.")
    distance = max(abs(target[name] - start[name]) for name in target)
    count = max(1, math.ceil(distance / max_step - 1e-12))
    return [{name: (target[name] if index == count else
                    start[name] + (target[name] - start[name]) * index / count)
             for name in target} for index in range(1, count + 1)]


def estimate_duration(plan, *, point_settle, line_settle, ramp_rate, point_overhead=0.15):
    """Rough run time in seconds: settling, read overhead and ramp travel.

    ``ramp_rate`` is one rate (V/s) or a dict role -> rate. Coordinated moves
    ramp the electrodes in turn within each small step, so travel times add.
    """
    rates = ramp_rate if isinstance(ramp_rate, dict) else {role: ramp_rate for role in ELECTRODES}
    rates = np.array([max(float(rates.get(role, math.inf)), 1e-6) for role in ELECTRODES])
    total = len(plan.order) * (float(point_settle) + point_overhead)
    if len(plan.order):
        slow = plan.order[:, 0] if plan.fast_axis == "x" else plan.order[:, 1]
        total += (1 + np.count_nonzero(np.diff(slow))) * float(line_settle)
        travel = np.abs(np.diff(plan.ordered_targets(), axis=0)).sum(axis=0)
        total += float(np.sum(travel / rates))
    return total


# ---------------------------------------------------------------------------
# Bridge balancing (port of LinearBalancingBridge / CapacitanceBridgeSR830Lockin)
# ---------------------------------------------------------------------------

def ac_scale_factor(ref_atten_db, sample_atten_db, sample_amplitude):
    """Original ``balance()`` scale: 10**(-(ref_atten - sample_atten)/20) / ch1_v."""
    amplitude = float(sample_amplitude)
    if not math.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("The sample excitation amplitude must be positive.")
    return 10 ** (-(float(ref_atten_db) - float(sample_atten_db)) / 20.0) / amplitude


def vector_phase_deg(vector):
    """Original ``vec_phase``: AWG phase (0-360 deg) for a balance vector."""
    phase = -math.degrees(math.atan2(float(vector[1]), float(vector[0])))
    if phase < 0:
        phase += 360.0
    return 0.0 if phase >= 360.0 else phase


def response_matrix(s1, m1, s2, m2):
    """Rotation-and-scale response matrix from two excitations (original algorithm)."""
    ds_in = np.asarray(s2, float) - np.asarray(s1, float)
    ds_out = np.asarray(m2, float) - np.asarray(m1, float)
    if np.linalg.norm(ds_in) == 0:
        raise BalanceError("The two balance probe vectors must differ.")
    magnitude = np.linalg.norm(ds_out) / np.linalg.norm(ds_in)
    shift = math.atan2(ds_out[1], ds_out[0]) - math.atan2(ds_in[1], ds_in[0])
    return magnitude * np.array([[math.cos(shift), -math.sin(shift)],
                                 [math.sin(shift), math.cos(shift)]])


def lockin_wait_time(time_constant, filter_slope):
    """Original ``wait_time``: 5/7/9/10 time constants for 6/12/18/24 dB/oct."""
    factor = {6: 5, 12: 7, 18: 9}.get(int(round(float(filter_slope))), 10)
    return factor * float(time_constant)


@dataclass
class BalanceResult:
    matrix: np.ndarray
    offset: np.ndarray
    balance_vector: np.ndarray
    ac_scale: float
    iterations: int
    converged: bool
    residual: float
    tolerance: float
    history: list = field(default_factory=list)
    message: str = ""

    @property
    def capacitance(self):
        """Original ``capacitance()``: C_s/C_ref = -vb_x * ac_scale."""
        return float(-self.balance_vector[0] * self.ac_scale)

    @property
    def dissipation(self):
        """Original ``capacitance()``: D_s/C_ref = vb_y * ac_scale."""
        return float(self.balance_vector[1] * self.ac_scale)

    @property
    def offbalance_coefficients(self):
        """Original ``offBalance()``: (Ctg, Dtg) scaling for off-balance X/Y."""
        det = float(np.linalg.det(self.matrix))
        vbx, vby = (float(v) for v in self.balance_vector)
        vdx, vdy = (float(v) for v in self.offset)
        norm = vbx ** 2 + vby ** 2
        if det == 0 or norm == 0:
            raise BalanceError("The balance result is degenerate (zero response or zero balance vector).")
        d_coeff = (vby * vdx - vbx * vdy) / norm / det * self.ac_scale
        c_coeff = -1.0 * (vbx * vdx + vby * vdy) / norm / det * self.ac_scale
        return c_coeff, d_coeff

    def offbalance(self, x, y):
        """Capacitance/dissipation (ratios) from off-balance lock-in X and Y."""
        c_coeff, d_coeff = self.offbalance_coefficients
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return c_coeff * x + d_coeff * y + self.capacitance, d_coeff * x - c_coeff * y + self.dissipation

    @property
    def magnitude(self):
        return float(math.hypot(self.matrix[0, 0], self.matrix[1, 0]))

    @property
    def phase_deg(self):
        return float(math.degrees(math.atan2(self.matrix[1, 0], self.matrix[0, 0])))

    @property
    def reference_amplitude(self):
        return float(np.linalg.norm(self.balance_vector))

    @property
    def reference_phase(self):
        return vector_phase_deg(self.balance_vector)

    def as_dict(self):
        c_coeff, d_coeff = self.offbalance_coefficients
        return {
            "matrix": self.matrix.tolist(), "offset": self.offset.tolist(),
            "balance_vector": self.balance_vector.tolist(), "ac_scale": self.ac_scale,
            "capacitance_ratio": self.capacitance, "dissipation_ratio": self.dissipation,
            "c_coefficient": c_coeff, "d_coefficient": d_coeff,
            "iterations": self.iterations, "converged": self.converged,
            "residual_V": self.residual, "tolerance_V": self.tolerance, "message": self.message,
        }


class LinearBridge:
    """Iterative linear balancing of a two-arm capacitance bridge.

    ``excite(vector)`` sets the reference excitation (amplitude and relative
    phase) and ``measure()`` returns the settled lock-in (X, Y). Both are
    supplied by the caller, so the same algorithm drives real or simulated
    instruments. ``keep_running()`` is polled before every hardware step.
    """

    def __init__(self, excite, measure, *, s1=(0.1, 0.1), s2=(-0.1, -0.1), tolerance=0.015e-3,
                 iterations=10, ac_scale=1.0, max_amplitude=1.0,
                 min_amplitude=AWG_MIN_AMPLITUDE_VPP, log=None, keep_running=None):
        self._excite = excite
        self._measure = measure
        self.s1 = np.asarray(s1, dtype=float).reshape(2)
        self.s2 = np.asarray(s2, dtype=float).reshape(2)
        self.tolerance = float(tolerance)
        self.iterations = int(iterations)
        self.ac_scale = float(ac_scale)
        self.max_amplitude = float(max_amplitude)
        self.min_amplitude = float(min_amplitude)
        self._log = log or (lambda message: None)
        self._keep_running = keep_running or (lambda: True)
        if not (self.tolerance > 0 and self.iterations >= 1 and self.max_amplitude > 0):
            raise ValueError("Balance tolerance, iterations and amplitude limit must be positive.")
        for vector in (self.s1, self.s2):
            amplitude = np.linalg.norm(vector)
            if not np.all(np.isfinite(vector)) or not self.min_amplitude <= amplitude:
                raise ValueError(f"Probe vector {vector.tolist()} has amplitude {amplitude:g} V, "
                                 f"below the AWG minimum of {self.min_amplitude:g} V.")
            if not np.all(np.abs(vector) < self.max_amplitude):
                raise ValueError(f"Probe vector {vector.tolist()} exceeds the {self.max_amplitude:g} V limit.")
        self.history = []

    def _check(self):
        if not self._keep_running():
            raise BalanceCancelled("Bridge balance stopped.")

    def excite_and_measure(self, vector):
        self._check()
        self._excite(np.asarray(vector, dtype=float))
        self._check()
        measured = np.asarray(self._measure(), dtype=float).reshape(2)
        if not np.all(np.isfinite(measured)):
            raise BalanceError("The lock-in returned a non-finite reading during balancing.")
        self.history.append((np.asarray(vector, float).tolist(), measured.tolist()))
        self._log(f"  excite |s|={np.linalg.norm(vector):.4g} V, phase={vector_phase_deg(vector):.2f} deg"
                  f" -> X={measured[0]:.4e} V, Y={measured[1]:.4e} V")
        return measured

    def balance(self, matrix=None, offset=None, start=None):
        """Measure the response matrix (unless given) and iterate to balance."""
        self.history = []
        if matrix is None:
            self._log("Measuring the bridge response matrix with two probe excitations.")
            m1 = self.excite_and_measure(self.s1)
            m2 = self.excite_and_measure(self.s2)
            matrix = response_matrix(self.s1, m1, self.s2, m2)
            start_vector, start_measurement = self.s1, m1
        else:
            matrix = np.asarray(matrix, dtype=float)
            start_vector = np.asarray(start, dtype=float)
            start_measurement = self.excite_and_measure(start_vector)
        if abs(np.linalg.det(matrix)) < 1e-300:
            raise BalanceError("The bridge shows no response to the reference excitation. "
                               "Check the AWG reference output, attenuators and lock-in input.")
        self._log(f"Response |M| = {math.hypot(matrix[0, 0], matrix[1, 0]):.4g}, "
                  f"arg M = {math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])):.2f} deg")
        constant = (start_measurement - matrix @ start_vector) if offset is None else np.asarray(offset, float)
        inverse = np.linalg.inv(matrix)
        iteration, residual, converged = 1, math.inf, False
        vector = start_vector
        while True:
            vector = -inverse @ constant
            if not np.all(np.abs(vector) < self.max_amplitude):
                raise BalanceError(
                    f"Balance point {np.round(vector, 6).tolist()} V is out of range "
                    f"(|component| must stay below {self.max_amplitude:g} V). "
                    "Remove attenuators from the reference or reduce the sample excitation.")
            if np.linalg.norm(vector) < self.min_amplitude:
                raise BalanceError(
                    f"Balance amplitude {np.linalg.norm(vector):.3g} V is below the AWG minimum "
                    f"{self.min_amplitude:g} V. Add reference attenuation or raise the sample excitation.")
            measured = self.excite_and_measure(vector)
            constant = measured - matrix @ vector
            iteration += 1
            residual = float(np.linalg.norm(measured))
            self._log(f"Iteration {iteration - 1}: residual |m_b| = {residual:.3e} V "
                      f"(tolerance {self.tolerance:.3e} V)")
            if residual < self.tolerance:
                converged = True
                message = "Balanced."
                break
            if iteration > self.iterations:
                message = f"Hit the maximum number of iterations ({self.iterations}); residual {residual:.3e} V."
                break
        result = BalanceResult(matrix=matrix, offset=constant, balance_vector=vector,
                               ac_scale=self.ac_scale, iterations=iteration - 1,
                               converged=converged, residual=residual,
                               tolerance=self.tolerance, history=list(self.history), message=message)
        self._log(f"{message} C/C_ref = {result.capacitance:.6g}, D/C_ref = {result.dissipation:.6g}")
        return result

    def rebalance(self, previous):
        """Re-balance from the previous balance point, reusing its response matrix."""
        self._log("Rebalancing with the stored response matrix.")
        return self.balance(matrix=previous.matrix, start=previous.balance_vector)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class SimulatedSample:
    """Deterministic, smooth sample response in units of C_ref (C - iD).

    A gap opens at charge neutrality with |p0| (bilayer-like). For a gate
    capacitance it appears as a dip; for penetration field capacitance as a peak.
    """

    def __init__(self, kind="gate", delta=0.0):
        self.kind = kind
        self.delta = float(delta)

    def complex_capacitance(self, vt, vb, vs):
        n0, p0 = density_field(vt, vb, vs, delta=self.delta)
        neutral = 0.05 * p0
        gap = 0.15 + 0.35 * math.tanh(abs(p0) / 1.5)
        width = 0.08 + 0.04 * abs(p0)
        profile = math.exp(-0.5 * ((n0 - neutral) / width) ** 2)
        oscillation = 0.01 * math.cos(2 * math.pi * n0 / 0.35) * math.exp(-abs(n0) / 2.0)
        if self.kind == "penetration":
            capacitance = 0.3 + 0.5 * gap * profile + 0.3 * oscillation
        else:
            capacitance = 0.8 * (1 - gap * profile) + 0.02 * math.tanh(n0) + oscillation
        dissipation = 0.18 * gap * math.exp(-0.5 * ((n0 - neutral) / (0.7 * width)) ** 2)
        return complex(capacitance, -dissipation)


class SimulatedBridgeSignal:
    """Lock-in input of a bridge referenced to the sample excitation (AWG CH1).

    V = G e^{i theta} (C_s V_s + C_ref V_r) / (C_s + C_ref + C_par), with the
    sample excitation V_s = A1 10^(-att_s/20) and the reference
    V_r = A2 10^(-att_r/20) exp(i (phi2 - phi1)). This is linear in the balance
    vector, as the original algorithm assumes, so balancing converges to
    C_s/C_ref exactly as with the real bridge.
    """

    def __init__(self, excitation, electrodes, attenuation, *, sample=None, gain=1.0,
                 phase_offset_deg=27.0, parasitic=50.0, noise_rms=0.0, seed=None):
        self._excitation = excitation
        self._electrodes = electrodes
        self._attenuation = attenuation
        self.sample = sample or SimulatedSample()
        self.gain = float(gain)
        self.phase_offset = math.radians(phase_offset_deg)
        self.parasitic = float(parasitic)
        self.noise_rms = float(noise_rms)
        self._rng = np.random.default_rng(seed)

    def __call__(self, time_constant=0.1):
        state = self._excitation()
        sample_atten, ref_atten = self._attenuation()
        vt, vb, vs = self._electrodes()
        cs = self.sample.complex_capacitance(vt, vb, vs)
        a1 = state["ch1_amplitude"] if state.get("ch1_enabled", True) else 0.0
        a2 = state["ch2_amplitude"] if state.get("ch2_enabled", True) else 0.0
        relative = math.radians(state.get("ch2_phase", 0.0) - state.get("ch1_phase", 0.0))
        v_sample = a1 * 10 ** (-sample_atten / 20.0)
        v_reference = a2 * 10 ** (-ref_atten / 20.0) * complex(math.cos(relative), math.sin(relative))
        node = (cs * v_sample + 1.0 * v_reference) / (cs + 1.0 + self.parasitic)
        value = self.gain * complex(math.cos(self.phase_offset), math.sin(self.phase_offset)) * node
        if self.noise_rms > 0:
            scale = self.noise_rms * math.sqrt(0.1 / max(float(time_constant), 1e-5))
            value += complex(*self._rng.normal(0, scale, 2))
        return value
