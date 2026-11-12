"""Electrode coordinates, sweep plans and voltage limits shared by the gate-map tabs.

The capacitance-bridge and four-probe transport tabs sweep the same three
electrodes (top gate, bottom gate, sample). This module has no Qt or instrument
dependency. The conventions are those of the qcodes-qpg scripts
(``capacitance_measurement.py`` and ``include/rotated_basis.py``):

    n0 = (1 + delta) (Vt - Vs) + (1 - delta) (Vb - Vs)   normalized density, V
    p0 = (1 + delta) (Vt - Vs) - (1 - delta) (Vb - Vs)   normalized field, V
    delta = (Ct - Cb) / (Ct + Cb)                        gate asymmetry

One electrode is held at ``v_fixed`` and the other two follow from (n0, p0),
or, in direct mode, are swept directly. Every physical target of a sweep is
built and checked before anything moves; nothing is clipped.
"""
from dataclasses import dataclass, field
import math

import numpy as np

ELECTRODES = ("top", "bottom", "sample")
ELECTRODE_LABELS = {"top": "Top gate", "bottom": "Bottom gate", "sample": "Sample"}
ELECTRODE_SYMBOLS = {"top": "V_top", "bottom": "V_bottom", "sample": "V_sample"}

COORDINATE_MODES = ("density", "direct")


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
        raise ValueError("A gate map is limited to 1,000,000 points.")
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
