# openCOSMO-RS-Phi

openCOSMO-RS-Phi is the equation-of-state extension of openCOSMO-RS. It keeps
the openCOSMO-RS contribution and adds the Phi/EOS machinery needed for
vapor-liquid calculations, vapor pressures, densities, critical/azeotropic
searches, flash calculations, and excess or mixing properties.

The core model is an implementation of COSMO-SAC-Phi in C++ and exposed to Python through pybind11 as
the `openCOSMORS` module. The Python scripts in `bindings/` are the recommended
example entry point for everyday calculations.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `code/` | Current openCOSMO-RS-Phi C++ source and Python binding. |
| `bindings/` | Example scripts, COSMO files, and locally built Python extension files. |
| `eigen/`, `pybind11/` | External dependencies. |

## Requirements

- CMake 3.10 or newer.
- A C++14 compiler. On Windows, we have tested MSVC from Visual Studio 2022.
- Python with NumPy. The bundled Visual Studio Python project is configured for
  the conda environment `base311`. But this can easily be changed.
- After cloning, initialize dependencies:

```bash
git submodule update --init --recursive
```

If this checkout was copied without registered submodule metadata, make sure
`eigen/` and `pybind11/` are populated before building.

## Build

### Python Extension

The default CMake target builds the `openCOSMORS` Python extension.

```bash
cmake -S . -B build -DOPENCOMSORS_SIMD=AUTO
cmake --build build --config Release
```

On Windows with a conda environment:

```powershell
conda activate base311
cmake -S . -B build -DOPENCOMSORS_SIMD=FMA -DPython_EXECUTABLE="$env:CONDA_PREFIX\python.exe"
cmake --build build --config Release
```

If your CMake version does not know the Visual Studio 2022 generator, run from a
Visual Studio developer shell or after `vcvars64.bat` and use NMake:

```powershell
cmake -S . -B build-msvc-nmake -G "NMake Makefiles" -DOPENCOMSORS_SIMD=FMA -DPython_EXECUTABLE="$env:CONDA_PREFIX\python.exe"
cmake --build build-msvc-nmake
```

### Visual Studio Solution

`openCOSMORS.sln` contains:

- `openCOSMORS-Phi`: the C++/pybind11 project.
- `openCOSMORS_python_example`: the Python example project.

The Visual Studio project uses `/arch:AVX2` and defines
`OPENCOMSORS_SIMD_FMA` for its x64 configurations. The produced module is still
named `openCOSMORS` so existing Python imports continue to work.

### Standalone CLI

The CMake CLI target is available with:

```bash
cmake -S . -B build-cli -DBINARY=1 -DOPENCOMSORS_SIMD=AUTO
cmake --build build-cli --config Release
```

The Python API is the primary interface for openCOSMO-RS-Phi workflows. The CLI
entry point is less complete than the Python binding for the current Phi/EOS
feature set.

## SIMD Backend

The SIMD backend is selected with `OPENCOMSORS_SIMD`:

| Value | Meaning |
| --- | --- |
| `AUTO` | Selects a backend from the target architecture. |
| `SCALAR` | Portable fallback for debugging or old CPUs. |
| `SSE3` | x86 SSE3 backend. |
| `AVX` | x86 AVX backend. |
| `FMA` | x86 AVX/FMA backend. |
| `NEON` | AArch64/ARM64 NEON backend. |

Examples:

```bash
cmake -S . -B build -DOPENCOMSORS_SIMD=SCALAR
cmake -S . -B build -DOPENCOMSORS_SIMD=FMA
cmake -S . -B build -DOPENCOMSORS_SIMD=NEON
```

## Running Examples

The current examples live in `bindings/`. Run them from that directory or add
the directory containing the compiled `.pyd`/shared library to `PYTHONPATH`.

```powershell
conda activate base311
cd bindings
python run_phi_pure_with_densities.py
```

If you built into `build/Release`, use:

```powershell
$env:PYTHONPATH = "$PWD\build\Release;$env:PYTHONPATH"
python bindings\run_phi_pure_with_densities.py
```

### Pure-Component Vapor Pressure

`bindings/run_phi_pure.py` is the minimal pure-component vapor-pressure example.
It calculates the vapor pressure of methanol at one user-provided temperature.

### Vapor Pressure With Saturated Densities

`bindings/run_phi_pure_with_densities.py` extends the pure-component workflow.
It exposes:

```python
calculate_vapor_pressure_and_densities(
    orcacosmo_file,
    pure_component_parameters,
    temperatures,
    pressure_guess=101325.0,
)
```

The function returns one dictionary per temperature with:

- `temperature_K`
- `vapor_pressure_Pa`
- `vapor_molar_volume_m3_per_mol`
- `liquid_molar_volume_m3_per_mol`
- `vapor_molar_density_mol_per_m3`
- `liquid_molar_density_mol_per_m3`
- `vapor_mass_density_kg_per_m3`
- `liquid_mass_density_kg_per_m3`

The script also contains a runnable methanol example:

```python
methanol_pc_parameters = [
    56.08186814344738,
    1.3518314651065213,
    3.3225493224786162,
    8.713083189565843,
]
methanol_temperatures = np.array([280.0, 298.15, 320.0], np.float64)
results = calculate_vapor_pressure_and_densities(
    "methanol.orcacosmo",
    methanol_pc_parameters,
    methanol_temperatures,
)
```

### Mixture Workflows

`bindings/run_phi_mixture.py` contains examples and helper functions for common
binary-mixture workflows, including:

- isothermal VLE diagrams
- isobaric VLE diagrams
- dew-pressure calculations
- pT flash calculations
- activity coefficients
- excess and mixing properties
- critical and azeotropic point searches

The file is intentionally script-oriented: uncomment the calculation block you
want to run, set component COSMO files and pure-component parameters, then run
the script from `bindings/`.

## Python API Shape

The pybind11 API is dictionary-based:

```python
import openCOSMORS

openCOSMORS.initialize()
openCOSMORS.loadMolecules(options, parameters, components)
openCOSMORS.loadCalculations(calculations)
calculations = openCOSMORS.calculate(parameters, calculations, False)
```

The result arrays are allocated in Python and mapped by the C++ layer. Keep
arrays C-contiguous, `dtype=np.float64`, and alive until `calculate()` returns.
For Phi/EOS calculations, pure-component saturated vapor and liquid states are
stored as paired rows: for `n` temperatures, vapor rows are `0:n` and liquid
rows are `n:2*n`.

## Related Projects

- [openCOSMO-RS_py](https://github.com/TUHH-TVT/openCOSMO-RS_py)
- [LVPP sigma profile database](https://github.com/lvpp/sigma)
- [Benchmark COSMO-SAC implementation](https://github.com/usnistgov/COSMOSAC)
- [Pysac](https://github.com/lvpp/pysac)
