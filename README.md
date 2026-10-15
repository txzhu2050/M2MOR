# M2MOR

This repo contains the source code of M2MOR, associated with the TCAD paper:

> **M2MOR: Multi-scale Thermo-mechanical Coupling Simulation of Chiplet Heterogeneous Integration Systems Based on Model Order Reduction**

M2MOR performs thermal and mechanical co-simulation of micro-structure arrays in chiplet heterogeneous integration systems. In the offline stage, detailed finite-element models of local structures are reduced to compact reduced-order models (ROMs). In the online stage, these ROMs are assembled according to a floorplan and solved in parallel at the system scale.

M2MOR is the improved version of MORE-Stress.

## Repository Layout

```text
M2MOR/
├── src/                  # Offline/online solvers and supporting modules
├── cases/
│   ├── tsv/              # Through-silicon-via example
│   ├── mbump/            # Micro-bump example
│   ├── c4bump/           # C4-bump example
│   ├── hbt/              # Hybrid-bonding example
│   ├── GPUHBM/           # GPU-HBM system example
│   └── generate_flp.py   # Floorplan-generation script
└── README.md
```

Each example directory is self-contained and includes a `sim.config` file together with the XML structure definitions, material library, and floorplan used by that case.

## Environment Setup

The code has been tested with **FEniCSx 0.11.0** in a Conda environment. The main dependencies are:

- FEniCSx 0.11.0 (including DOLFINx, UFL, Basix, PETSc, and mpi4py)
- an MPI implementation (the tested environment uses MPICH)
- PyVista
- MPI-enabled h5py
- Gmsh and its Python interface
- NumPy and SciPy
- tqdm
- psutil
- Matplotlib (used by the floorplan-generation and visualization scripts)

Most dependencies can be installed from `conda-forge`. First create and activate a Conda environment:

```bash
conda create -n fenicsx python=3.12
conda activate fenicsx
```

Install FEniCSx using the command recommended on the [FEniCS Project download page](https://fenicsproject.org/download/):

```bash
conda install -c conda-forge fenics-dolfinx mpich pyvista
```

To reproduce the version tested with M2MOR exactly, use the following version-pinned variant instead:

```bash
conda install -c conda-forge fenics-dolfinx=0.11.0 mpich pyvista
```

Then install the additional packages required by M2MOR:

```bash
conda install -c conda-forge \
    "h5py=*=mpi_mpich*" \
    gmsh python-gmsh \
    numpy scipy tqdm psutil matplotlib
```

The h5py build must support MPI because M2MOR uses the `mpio` HDF5 driver in both the offline and online stages. Verify the installation with:

```bash
python -c "import dolfinx, h5py; print('DOLFINx:', dolfinx.__version__); print('h5py MPI:', h5py.get_config().mpi)"
```

The expected output includes DOLFINx version `0.11.0` and `h5py MPI: True`. All MPI-dependent packages must use the same MPI implementation; do not mix MPICH and Open MPI packages in one environment.

## Running an Example

The following commands use the TSV example. Run them from the repository root after activating the Conda environment.

### 1. Offline stage

The offline stage generates meshes and reduced-order models for all structures referenced by `sim.config`:

```bash
cd cases/tsv
python ../../src/Main_offline.py -config sim.config
```

The generated mesh and ROM files are written to the directory specified by `model_dir` in `sim.config`.

### 2. Online stage

After the offline stage has completed, run the system-level simulation with the desired number of MPI processes:

```bash
mpirun -np <N> python ../../src/Main_online.py -config sim.config
```

Replace `<N>` with the number of MPI processes to use, for example:

```bash
mpirun -np 16 python ../../src/Main_online.py -config sim.config
```

The online results are written to the directory specified by `result_dir` in `sim.config`.

The same workflow can be used for the other examples by entering `cases/mbump`, `cases/c4bump`, or `cases/hbt` before running the commands.

## Input Files

The main entry point for a simulation is `sim.config`. Paths in the configuration are interpreted relative to the directory from which the program is launched, so the recommended practice is to run M2MOR from the selected case directory.

### `sim.config`

The configuration file controls the simulation workflow and references all other input files. Its main sections include:

| Section | Purpose |
| --- | --- |
| `[Simulation]` | Simulation mode, interpolation order, ROM directory, and result directory. |
| `[MeshConfig]` | Default mesh size, internal unit scaling, and geometric precision. |
| `[SolverConfig]` | Online solver type, tolerances, iteration limit, and solver output. |
| `[Temperature]` | Reference and ambient temperatures. |
| `[Structure]` | Paths to the XML structure files and the default structure name. |
| `[Floorplan]` | Path to the floorplan JSON file. |
| `[Boundary]` | Optional origins and paths to prescribed temperature or displacement data. |
| `[Matlib]` | Path to the material library. |
| `[Postprocess]` | Requested output fields and post-processing resolution. |
| `[Solid]` | Names of material properties required by the solid model. |

For example:

```ini
[Structure]
path = ./tsv.xml, ./dummy.xml
default = tsv

[Floorplan]
path = ./flp.json

[Matlib]
path = ./material.config
```

### XML structure files

An XML file defines one local structure as an ordered collection of geometric shapes. Each `Shape` specifies its geometry, material, and whether it acts as a heat source. Supported shape types include boxes, cylinders, and spheres.

```xml
<Shape Name="3" Type="cylinder" FillingMaterial="Cu" PowerOn="True">
    <Geometry Radius="10e-6" Height="10e-6"
              OriginXYZ="(15e-6,15e-6,0.0)"/>
</Shape>
```

Shape order is significant when geometries overlap: **a later shape overrides an earlier shape** in the overlapping region.

- `PowerOn="True"` marks the shape as power-generating and creates a corresponding power input in the ROM.
- `PowerOn="False"` marks the shape as non-power-generating.
- `FillingMaterial` must match a material name defined in the material file.

If a structure contains multiple power-generating shapes, their power-density inputs appear in the same order as the corresponding `PowerOn="True"` shapes in the XML file.

### Material files

The material file uses INI syntax. Each section defines one material and its thermo-mechanical properties, for example:

```ini
[Cu]
type = Solid

young modulus (N/m2) = 121e9
poisson ratio (1) = 0.34
thermal expansion coefficient (1/K) = 16.3e-6
thermal conductivity (W/m-K) = 400
```

Every material referenced by `FillingMaterial` in an XML file must be present in the material library and must provide the properties listed under `[Solid]` in `sim.config`.

### Floorplan files

The floorplan (`flp.json`) defines the two-dimensional arrangement of local structures. The top-level `size` field specifies the array dimensions. Each numbered item specifies a location, a structure type, and its power-density values:

```json
{
  "size": [2, 2],
  "0": {
    "loc": [0, 0],
    "structure": "tsv",
    "power": [2.5e14]
  },
  "1": {
    "loc": [1, 0],
    "structure": "dummy",
    "power": []
  }
}
```

- `loc` is the integer array coordinate `[x, y]`.
- `structure` must match the stem of an XML file listed under `[Structure]` in `sim.config`.
- `power` contains the power density for each power-generating shape in that structure. Its length must equal the number of `PowerOn="True"` shapes in the corresponding XML file.

The script [`cases/generate_flp.py`](cases/generate_flp.py) provides an example for generating a floorplan and a preview image. Edit the array dimensions, structure name, population ratio, and power distribution near the beginning of the script, then run it from the target case directory:

```bash
cd cases/tsv
python ../generate_flp.py
```

## License

This project is released under the [MIT License](LICENSE).

## Citation

If you use M2MOR in your research, please cite:

```bibtex
@ARTICLE{11670304,
  author={Zhu, Tianxiang and Wang, Qipan and Lin, Yibo and Wang, Runsheng},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  title={M2MOR: Multi-scale Thermo-mechanical Coupling Simulation of Chiplet Heterogeneous Integration Systems Based on Model Order Reduction},
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Modeling; Arrays; Reduced order systems; Simulation; Temperature; Measurement units; Couplings; Joining processes; Stress; Integrated circuits; Chiplet Heterogeneous Integration; Thermo-Mechanical Simulation; Model Order Reduction; Multi-scale Simulation},
  doi={10.1109/TCAD.2026.3728619}
}
```

M2MOR builds on our earlier work on model-order-reduction-based thermal stress simulation of TSV arrays. Please also consider citing:

```bibtex
@INPROCEEDINGS{10993234,
  author={Zhu, Tianxiang and Wang, Qipan and Lin, Yibo and Wang, Runsheng and Huang, Ru},
  booktitle={2025 Design, Automation & Test in Europe Conference (DATE)},
  title={MORE-Stress: Model Order Reduction based Efficient Numerical Algorithm for Thermal Stress Simulation of TSV Arrays in 2.5D/3D IC},
  year={2025},
  volume={},
  number={},
  pages={1-7},
  keywords={Computational modeling; Software algorithms; Memory management; Software; Finite element analysis; Numerical models; Through-silicon vias; Stress; Thermal stresses; Convergence},
  doi={10.23919/DATE64628.2025.10993234}
}
```

## Contact

For questions or feedback, please contact the author at [txzhu@pku.edu.cn](mailto:txzhu@pku.edu.cn).
