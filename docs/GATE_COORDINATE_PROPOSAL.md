# Proposed gate coordinates ? not implemented

Define what the channels APPLY directly, instead of asking users to define a
matrix that the program later inverts. Explicitly assign channel 1/2 to Top and
Bottom in the setup. Let u and v be two named sweep coordinates in volts:

```
Vtop    = Vtop_offset    + a*u + b*v
Vbottom = Vbottom_offset + c*u + d*v
```

The table headings should be "Applied top voltage per +1 V of u", etc.
Entering 0.5 as a coefficient means half of that sweep coordinate is applied;
no inverse or hidden normalization is performed. Offsets are explicit and
start at zero; choosing a preset must not cause hardware movement. Arbitrary
coefficients remain allowed, even if a direction is redundant; warn about a
redundant two-dimensional sweep without treating it as a hardware error.

## Presets and concrete examples

| Preset | Applied top voltage | Applied bottom voltage | Example |
|---|---|---|---|
| Independent physical gates | u | v | u=10, v=0 -> 10 V top, 0 V bottom |
| Equal voltage / difference | u+v | u-v | u=10, v=0 -> 10 V on each gate |
| Custom, one coordinate | a*u | c*u | a=c=0.5, u=10 -> 5 V on each gate |

In the equal-voltage preset, u is the common voltage and v is HALF of the
voltage difference. Label that fact directly. Never label u as charge density
unless the capacitance calibration is explicitly supplied.

## Optional hBN thickness preset

In the geometric-capacitance approximation, equal induced charge contributions
satisfy Ct*(Vtop-Vtop0) = Cb*(Vbottom-Vbottom0). For equal dielectric constants,
this gives (Vtop-Vtop0)/dtop = (Vbottom-Vbottom0)/dbottom. Thus capacitance is
inverse in thickness, while the voltage required for equal charge is
proportional to thickness. With zero offsets this is exactly the requested
relation: dtop*Vbottom = dbottom*Vtop.

For an easy voltage scale, define D=max(dtop, dbottom), kt=dtop/D, kb=dbottom/D:

```
Vtop    = Vtop0    + kt*(u+v)
Vbottom = Vbottom0 + kb*(u-v)
```

Example: dtop=20 nm, dbottom=40 nm, u=10 V, v=0 -> top=5 V,
bottom=10 V. Equal-voltage mode remains available separately. For unequal
permittivities, replace thicknesses by d/epsilon; alternatively allow measured
capacitances. Charge offsets/traps and quantum capacitance can require a more
complete sample model. This preset is an optional geometric approximation.

The density/displacement decomposition above follows the capacitance-weighted
sum/difference used in the primary research paper
[Direct measurement of ferroelectric polarization in a tunable semimetal](https://www.nature.com/articles/s41467-021-25587-3).
The direct software mapping and normalized presets here are our proposed UI
conventions, not claims from that paper.

## Required preview and protection

Before Start, show the actual top and bottom minimum/maximum over the complete
sweep, the largest absolute top-bottom difference, the selected direction and
physical start/end values. Keep target and measured physical voltages visible
next to the plotted coordinates. Reject the entire sweep if any physical
channel target exceeds its code-only cap; never clip or rescale silently.
Both coordinates and offsets can add, even when each coefficient is <=1.
Check the physical trajectory during sequential channel ramps too; a future
sample-specific differential/field limit would need its own code setting.

Store the mapping direction/version, coefficients, channel assignment,
offsets, optional thickness/permittivity/calibration, physical grids and actual
readbacks in each file. Older files must preserve their recorded grids and
legacy convention rather than being reinterpreted under the new model.

## To do

- Review this proposal before implementing or migrating any gate-coordinate UI.
- Keep the current transform unchanged until that separate change is requested.

## Potentially dangerous

The current transform is still active. The 14 V per-channel cap bounds its
output but does not impose a top-bottom difference or sample electric-field
limit. Neither this proposed model nor 100 nA compliance guarantees that a
particular sample can withstand a selected voltage.
