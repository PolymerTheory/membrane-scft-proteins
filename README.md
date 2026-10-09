# Arcs SCFT

CUDA/MPI self-consistent field and string calculations for membranes with
arc-shaped proteins. This is a source-release candidate; full manuscript
free-energy validation is in progress. Scientific reproduction datasets and
final paper/release identifiers will be added after that validation.

## Requirements and installation

Solver: Linux, NVIDIA CUDA-capable GPU(s), CUDA/CUFFT, MPI, C++ compiler and
HDF5 with zlib. CPU protein-geometry demo: C++ compiler and Python 3 with
NumPy and h5py; it does not require CUDA. Tests additionally use NumPy/h5py.

The source-release solver was built and run on JUPITER: Linux aarch64, RHEL 9.8,
GCC 14.3.0, ParaStationMPI 5.13.0-1, CUDA 13.0 (nvcc 13.0.48),
HDF5 1.14.6 serial. CPU preview tested on macOS 27.0.1 arm64, Apple clang 21.0.0,
Python 3.12.9, NumPy 2.5.0, h5py 3.13.0 (HDF5 1.14.6). A CUDA desktop build has not yet been tested. A JUPITER binary
cannot be used directly on macOS or an x86 desktop.

Build the small geometry demo (in an environment with a C++ compiler):

```sh
python -m pip install -r requirements.txt
make -C src protein_preview SCFT_STRN=4
mkdir -p build/geometry
cp examples/geometry/* build/geometry/
(cd build/geometry && ../../src/protein_preview input.dat)
```

Expected: `proteins.h5` and `proteins.xdmf`, containing finite protein fields
`prot1`, `prot2`, `prot3` for four 32x32x32 replicas. Additive mode yields
z-reflection symmetry. Use `--vtk` to export twelve VTK files instead.
The clean preview build took 1.48 seconds and this demo took 1.17 seconds
on the tested Mac, with the compiler and Python dependencies already installed.
Dependency installation is additional and was not timed in a fresh environment.
Expected field maxima and sums are in `examples/geometry/expected_output.json`;
small floating-point differences across platforms are possible.
Preview uses unit chi; solver fields include the input interaction strength.

Build the GPU demo (adapt MPI compiler and library locations to your system):

```sh
make -C src SCFT_STRN=4 USE_HDF5=1 NVCC='nvcc -ccbin mpicxx'
mkdir -p build/gpu-smoke
cp examples/gpu-smoke/* build/gpu-smoke/
(cd build/gpu-smoke && mpiexec -n 1 ../../src/scft_gpu input.dat > solver.log 2>&1)
python examples/gpu-smoke/check_output.py build/gpu-smoke
```

Expected: finite `FEs`, restart `wins.h5`, density `concentrations.h5`, protein
`proteins.h5`, corresponding visualization metadata and normal `Done` exit.
One GPU/rank can own the four replicas. The clean public-source GPU smoke
ran in 17 seconds of solver time (20.53 seconds including launch) on one
NVIDIA GH200 GPU on JUPITER; NVIDIA driver 595.71.05. Full output arrays
and four free-energy rows were finite, and the solver exited normally.
Reference output statistics and free-energy rows are in
`examples/gpu-smoke/expected_output.json` for inspection; numerical values
may differ across platforms. This five-iteration analytical-start demo verifies execution; it is not a
converged membrane calculation or a physical-density reference solution.

## Inputs and use

Run from a directory containing `input.dat` and `prot_input.dat`; example
positions/orientations are provided in `P0s` and `qtns`. Use separate run
folders because output files are written in the current directory.
Replica count is compile-time `SCFT_STRN`; it must match the input arrays.
For a new calculation, copy an example into a fresh directory and edit the
keyed values (comments after `#` are allowed). `m` gives the three grid sizes;
`D` gives the three box lengths in the model's reduced length units. These
are not automatically nanometres. Preserve the paper-specific unit conversion
when comparing physical radii or free energies. `chi`, `f`, `N`, `Vc` and `ens`
define the interaction, chain and ensemble setup; keep them consistent with
the selected reference calculation. Patch offsets are radians.

Provide one row per compiled replica in `P0s` (`x y z theta`) and `qtns`
(quaternion `w x y z`); the supplied examples use normalized quaternions.
Set `readwin=0` for the analytical initial field, or `readwin=1` for a
three-field restart. With HDF5 enabled, `wins.h5` datasets `W1`, `W2`, `Wp`
must have shape `(SCFT_STRN, mx, my, mz)` matching the grid and replica count.
Use a restart from the same model and compatible geometry; translating a
periodic field or changing protein construction requires explicit scientific
validation. `dostring` selects string updates rather than independent images.

Outputs `concentrations.h5` (`rhoA`, `rhoB`, `rhoH`) and `proteins.h5`
(`prot1`, `prot2`, `prot3`) use the same replica-first shape. Block A is the
hydrophobic membrane component. `FEs` contains replica free energies in model
units; manuscript normalization/conversion is a separate processing step.
`max_outer` bounds work rather than certifying convergence. Restart from the
latest intact fields for further bounded segments and verify finite full
arrays, composition/pressure residuals, energy stability and pathway morphology.
Larger calculations require restart fields and adequate GPU memory.

In keyed `prot_input.dat`:

| Input | Values | Omitted default |
|---|---|---|
| protein_cap_mode | 0: exterior caps off; 1: aligned caps on | 1 |
| protein_symmetrize | 0: single field; 1: original plus reflected field; 2: lower-z half copied onto upper-z half | 0 |
| protein_pivot_align | 0: no shift; 1: preserve arc midpoint for y-axis tilts | 0 |

The aligned body is the same with caps off/on. No legacy cap implementation
is included. Reflection mode 2 retains half-domain replacement for optional
use; mode 1 is the additive two-protein construction used for corrected paired
calculations, including doubled values at reflection planes.
`examples/single-protein` disables reflection; `examples/geometry` and
`examples/gpu-smoke` use additive reflection. Explicitly set the switches.

Other controls include protein enable/movement, surface versus patch mode,
patch offsets (radians), patch widths, arc extent, position, radius and pitch.
Alignment is specific to y-axis tilts. Cap centers use the aligned production
formula; anisotropic cap axes retain its existing treatment.

`input.dat` sets grid/box/chain/interaction parameters, restart loading,
string controls and checkpoint precision (32 or 64). `max_outer` defaults
1000; `field_iterations` defaults 100. The small GPU demo bounds these to
1 and 5. Finite outputs or normal exit alone do not establish convergence;
inspect residuals, free-energy stability and morphology.

## Tests and attribution

```sh
python -m unittest discover -s tests
```

Tests exercise caps off/on, all three reflection modes, finite output,
additive identity, half-copy semantics and rejection of unsupported cap modes.
See LICENSE.txt and NOTICE.md for MIT license and prior-paper attribution.
