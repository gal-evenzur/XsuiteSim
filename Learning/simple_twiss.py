import numpy as np
import matplotlib.pyplot as plt

import xobjects as xo
import xtrack as xt

## Generate a simple line
line = xt.Line(
    elements=[xt.Drift(length=0.5), # 5cm
              xt.Quadrupole(k1=2.0, length=0.2),
              xt.Bend(length=0.2, angle=np.pi/12),
              xt.Quadrupole(k1=-2.0, length=0.2),
              xt.Drift(length=1.)],
    element_names=['drift_0', 'quad_0', 'bend', 'quad_1', 'drift_1'])

## Attach a reference particle to the line (optional)
## (defines the reference mass, charge and energy)
line.particle_ref = xt.Particles(p0c=1e6, #eV
                                 q0=1, mass0=xt.PROTON_MASS_EV)

## Transfer lattice on context and compile tracking code
line.build_tracker()

## Compute lattice functions
tw = line.twiss(
    method='4d',
    init=xt.TwissInit(
        x=0, px = 0, y = 0, py = 0,
    ),
    start='drift_0',
    end='_end_point',
)
tw.cols['s betx bety'].show()
