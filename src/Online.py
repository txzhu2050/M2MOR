from dolfinx import fem, mesh, la, default_scalar_type
from dolfinx.fem.petsc import create_matrix, create_vector
from dolfinx.la.petsc import create_vector_wrap
import ufl
from pseudo import create_pseudo_element, assemble_matrix_, assemble_vector_
from utils import build_nullspace
from petsc4py import PETSc
from Parser import DataCollection
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import griddata
from mpi4py import MPI
import h5py
import json
from pathlib import Path
from typing import Optional
import sys

class FlpManager:
    def __init__(self, simdb: DataCollection, rom_dir: Path, flp_path: Path, comm: MPI.Comm) -> None:
        self.simdb = simdb
        self.rom_dir = rom_dir
        self.comm = comm
        
        with open(flp_path, 'r') as f:
            self.flp: dict = json.load(f)
        self._parse_flp()

        self.domain = mesh.create_box(
            comm, 
            [[0, 0, 0], [self.size_x, self.size_y, 1]], 
            [self.size_x, self.size_y, 1], 
            cell_type=mesh.CellType.hexahedron, 
            dtype=default_scalar_type
        )
    
    def _parse_flp(self):
        power_num_dict: dict[str, int] = {}
        exist_structures = []
        for rom_path in self.rom_dir.rglob('*.h5'):
            if rom_path.stem in self.simdb.structure_names:
                exist_structures.append(rom_path.stem)
                with h5py.File(rom_path, 'r', driver='mpio', comm=self.comm) as f:
                    power_num = f['rom_t']['mode_p'].shape[0]
                    power_num_dict[rom_path.stem] = power_num
        for name in self.simdb.structure_names:
            if not name in exist_structures:
                print(f"ERROR: Reduced-order models of structure {(name)} not found!")
                sys.exit(1)
        
        self.size_x = int(self.flp['size'][0])
        self.size_y = int(self.flp['size'][1])
        self.loc_names = [[self.simdb.default_name for _ in range(self.size_x)] for _ in range(self.size_y)]
        self.loc_powers = [[np.zeros(power_num_dict[self.simdb.default_name]) for _ in range(self.size_x)] 
                           for _ in range(self.size_y)]
        for key in self.flp.keys():
            if key != 'size':
                loc_x, loc_y = self.flp[key]['loc']
                name = self.flp[key]['structure']
                power = np.array(self.flp[key]['power']) / (10**(self.simdb.unit*2))
                if power.shape[0] != power_num_dict[name]:
                    print(f"ERROR: Invalid power values for item {key} in flp file!")
                    sys.exit(1)
                self.loc_names[loc_y][loc_x] = name
                self.loc_powers[loc_y][loc_x] = power


class BaseOnlineSolver:
    def __init__(self, flp_manager: FlpManager, bcs: Optional[list[fem.DirichletBC]]) -> None:
        self.flp_manager = flp_manager
        self.simdb = flp_manager.simdb
        self.comm = flp_manager.comm

        self.interp_num = self.simdb.interp_num
        self.interp_pt_num = self.simdb.interp_num**3 - (self.simdb.interp_num-2)**3

        self.stiffness_dict: dict[str, NDArray] = {}
        self.load_dict: dict[str, NDArray] = {}

        self.bcs = bcs
        
        self.dx = ufl.Measure("dx", self.flp_manager.domain)

        self.solution_func: Optional[fem.Function] = None

        self._solver = None
        self._A = None
        self._b = None
        self._x = None

    def _read_rom(self) -> None:
        pass

    def _set_up_bcs(self) -> None:
        pass
    
    @staticmethod
    def _get_cell_loc(domain: mesh.Mesh, local_idx) -> tuple[int, int]:
        coords = domain.geometry.x
        cell_coords = coords[domain.geometry.dofmap[local_idx], :]
        center_x = int(sum(cell_coords)[0]/8)
        center_y = int(sum(cell_coords)[1]/8)
        return (center_x, center_y)
    
    def _assemble_matrix_vector(self) -> None:
        pass

    def solve(self) -> fem.Function:
        # Initialize the PETSc solver
        self._solver = PETSc.KSP().create(self.comm)
        # Give PETSc solver options a unique prefix
        problem_prefix = f"dolfinx_solve_{id(self)}"
        self._solver.setOptionsPrefix(problem_prefix)
        # Set PETSc options
        opts = PETSc.Options()
        opts.prefixPush(problem_prefix)
        
        '''
        # Create the nullspace and attach it to the PETSc matrix
        ns = build_nullspace(self.solution_func.function_space)
        if not self.bcs:
            self._A.setNullSpace(ns)

        if self.simdb.solver_type == 0:
            self._A.setNearNullSpace(ns)
            opts["ksp_type"] = "cg"
            if self.simdb.rtol is not None:
                opts["ksp_rtol"] = self.simdb.rtol
            if self.simdb.atol is not None:
                opts["ksp_atol"] = self.simdb.atol
            if self.simdb.max_it is not None:
                opts["ksp_max_it"] = self.simdb.max_it
            opts["pc_type"] = "gamg"
            opts["pc_gamg_type"] = "agg"
            opts["pc_gamg_coarse_eq_limit"] = 1000
            opts["pc_gamg_square_graph"] = 2
            opts["pc_gamg_threshold"] = 0.02
            # Use Chebyshev smoothing for multigrid
            opts["mg_levels_ksp_type"] = "chebyshev"
            opts["mg_levels_pc_type"] = "jacobi"
            # Improve estimate of eigenvalues for Chebyshev smoothing
            opts["mg_levels_ksp_chebyshev_esteig_steps"] = 10
        elif self.simdb.solver_type == 1:
            opts["ksp_type"] = "preonly"
            opts["pc_type"] = "lu"
            opts["pc_factor_mat_solver_type"] = "mumps"
        if self.simdb.is_display:
                opts["-ksp_monitor_true_residual"] = None
                opts["-ksp_converged_reason"] = None
        opts.prefixPop()
        '''

        if self.simdb.solver_type == 0:
            opts["ksp_type"] = "gmres"
            opts["ksp_norm_type"] = 'unpreconditioned'
            opts["pc_type"] = "hypre"
            opts["ksp_rtol"] = self.simdb.rtol
            opts["ksp_atol"] = self.simdb.atol
            opts["ksp_max_it"] = self.simdb.max_it
        elif self.simdb.solver_type == 1:
            opts["ksp_type"] = "preonly"
            opts["ksp_norm_type"] = 'unpreconditioned'
            opts["pc_type"] = "lu"
            opts["pc_factor_mat_solver_type"] = "mumps"
        if self.simdb.is_display:
            opts["ksp_monitor"] = None
            opts["ksp_converged_reason"] = None
        opts.prefixPop()

        # Set matrix operator
        self._solver.setOperators(self._A)
        self._solver.setFromOptions()

        if self.comm.rank == 0:
            print("Start solving ...")
        self._x = create_vector_wrap(self.solution_func.x)
        self._solver.solve(self._b, self._x)
        if self.comm.rank == 0:
            print("Finish solving.")
            print()

        # Scatter forward the solution vector to update ghost values
        self.solution_func.x.scatter_forward()

        return self.solution_func


    def __del__(self) -> None:
        if self._A is not None:
            self._A.destroy()
        if self._b is not None:
            self._b.destroy()
        if self._x is not None:
            self._x.destroy()
        if self._solver is not None:
            self._solver.destroy()

class ThermalOnlineSolver(BaseOnlineSolver):
    def __init__(self, flp_manager: FlpManager, bcs: Optional[list[fem.DirichletBC]]):
        super().__init__(flp_manager, bcs)

        pseudo_element = create_pseudo_element(self.interp_num, 1)
        func_space = fem.functionspace(self.flp_manager.domain, pseudo_element)
        if self.comm.rank == 0:
            print(f"# of DoFs: {func_space.dofmap.index_map.size_global * func_space.dofmap.index_map_bs}")
        self.solution_func = fem.Function(func_space)

        self._read_rom()
        if self.bcs is None:
            self._set_up_bcs()
        self._assemble_matrix_vector()

    def _read_rom(self):
        for rom_path in self.flp_manager.rom_dir.rglob('*.h5'):
            if rom_path.stem in self.simdb.structure_names:
                with h5py.File(rom_path, 'r', driver='mpio', comm=self.comm) as f:
                    rom_t = f["rom_t"]
                    self.load_dict[rom_path.stem] = rom_t["load"][:]
                    self.stiffness_dict[rom_path.stem] = rom_t["stiffness"][:]

    def _set_up_bcs(self):
        func_space = self.solution_func.function_space

        top_marker = lambda x: (
            np.isclose(x[2], 1)
        )
        top_bc_func = fem.Function(func_space)
        top_bc_func.x.array[:] = self.simdb.amb_temp
        top_bc_func.x.scatter_forward()
        top_constraint = fem.dirichletbc(
                top_bc_func,
                fem.locate_dofs_geometrical(func_space, top_marker)
                )
        bot_marker = lambda x: (
            np.isclose(x[2], 0)
        )
        bot_bc_func = fem.Function(func_space)
        bot_bc_func.x.array[:] = self.simdb.amb_temp
        bot_bc_func.x.scatter_forward()
        bot_constraint = fem.dirichletbc(
                bot_bc_func,
                fem.locate_dofs_geometrical(func_space, bot_marker)
                )     
        self.bcs = [top_constraint, bot_constraint]


    def _assemble_matrix_vector(self):
        if self.comm.rank == 0:
            print("Start assembling ...")
        
        func_space = self.solution_func.function_space
        size_local = func_space.dofmap.index_map.size_local
        num_cells = self.flp_manager.domain.topology.index_map(3).size_local
        dofmap = func_space.dofmap.list.astype(np.dtype(PETSc.IntType))[:num_cells, :]

        u = ufl.TrialFunction(func_space); v = ufl.TestFunction(func_space)
        f = fem.Constant(self.flp_manager.domain, default_scalar_type(1))
        bilinear_form = fem.form(ufl.inner(u, v)*self.dx, dtype=default_scalar_type)
        linear_form = fem.form(f*v*self.dx, dtype=default_scalar_type)
        self._A = create_matrix(bilinear_form)
        self._A.zeroEntries()
        self._b = create_vector(func_space)
        with self._b.localForm() as b_loc:
            b_loc.set(0)

        element_stiffness = np.zeros((num_cells, self.interp_pt_num, self.interp_pt_num), dtype=default_scalar_type)
        for idx, _ in enumerate(dofmap):
            center_x, center_y = BaseOnlineSolver._get_cell_loc(self.flp_manager.domain, idx)
            element_stiffness[idx] = self.stiffness_dict[self.flp_manager.loc_names[center_y][center_x]]
        assemble_matrix_(self._A, element_stiffness, dofmap)
        self._A.assemble()
        for bc in self.bcs:
            dofs, _ = bc._cpp_object.dof_indices()
            self._A.zeroRowsLocal(dofs, diag=1)

        element_loads = np.zeros((num_cells, self.interp_pt_num), dtype=default_scalar_type)
        for idx, _ in enumerate(dofmap):
            center_x, center_y = BaseOnlineSolver._get_cell_loc(self.flp_manager.domain, idx)
            element_loads[idx] = self.flp_manager.loc_powers[center_y][center_x]@self.load_dict[self.flp_manager.loc_names[center_y][center_x]]
        assemble_vector_(self._b, element_loads, dofmap)
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        for bc in self.bcs:
            bc.set(self._b.array_w)

        if self.comm.rank == 0:
            func_space = self.solution_func.function_space
            total_num_dof = func_space.dofmap.index_map.size_global * func_space.dofmap.index_map_bs
            print(f"Total number of Dofs: {total_num_dof}")
            print("Finish assembling.\n")


class MechanicalOnlineSolver(BaseOnlineSolver):
    def __init__(self, 
                 flp_manager: FlpManager,
                 bcs: Optional[list[fem.DirichletBC]],
                 t_func: Optional[fem.Function]):
        super().__init__(flp_manager, bcs)
        
        self.t_func = t_func

        pseudo_element = create_pseudo_element(self.interp_num, 3)
        func_space = fem.functionspace(self.flp_manager.domain, pseudo_element)
        if self.comm.rank == 0:
            print(f"# of DoFs: {func_space.dofmap.index_map.size_global * func_space.dofmap.index_map_bs}")
        self.solution_func = fem.Function(func_space)

        self._read_rom()
        if self.bcs is None:
            self._set_up_bcs()
        self._assemble_matrix_vector()
    
    def _read_rom(self):
        for rom_path in self.flp_manager.rom_dir.rglob('*.h5'):
            if rom_path.stem in self.simdb.structure_names:
                with h5py.File(rom_path, 'r', driver='mpio', comm=self.comm) as f:
                    rom_u = f["rom_u"]
                    self.load_dict[rom_path.stem] = rom_u["load"][:]
                    self.stiffness_dict[rom_path.stem] = rom_u["stiffness"][:]

    def _set_up_bcs(self):
        func_space = self.solution_func.function_space

        top_marker = lambda x: (
            np.isclose(x[2], 1)
        )
        top_bc_func = fem.Function(func_space)
        top_constraint = fem.dirichletbc(
                top_bc_func,
                fem.locate_dofs_geometrical(func_space, top_marker)
                )

        bot_marker = lambda x: (
            np.isclose(x[2], 0)
        )
        bot_bc_func = fem.Function(func_space)
        bot_constraint = fem.dirichletbc(
                bot_bc_func,
                fem.locate_dofs_geometrical(func_space, bot_marker)
                )
        
        self.bcs = [top_constraint, bot_constraint]

    def _assemble_matrix_vector(self):
        if self.comm.rank == 0:
            print("Start assembling ...")

        func_space = self.solution_func.function_space
        size_local = func_space.dofmap.index_map.size_local
        num_cells = self.flp_manager.domain.topology.index_map(3).size_local
        dofmap = func_space.dofmap.list.astype(np.dtype(PETSc.IntType))[:num_cells, :]

        u = ufl.TrialFunction(func_space); v = ufl.TestFunction(func_space)
        f = fem.Constant(self.flp_manager.domain, default_scalar_type((1, 1, 1)))
        bilinear_form = fem.form(ufl.inner(u, v)*self.dx, dtype=default_scalar_type)
        linear_form = fem.form(ufl.inner(f,v)*self.dx, dtype=default_scalar_type)
        self._A = create_matrix(bilinear_form)
        self._A.zeroEntries()
        self._b = create_vector(func_space)
        with self._b.localForm() as b_loc:
            b_loc.set(0)

        element_stiffness = np.zeros((num_cells, self.interp_pt_num*3, self.interp_pt_num*3), dtype=default_scalar_type)
        for idx, _ in enumerate(dofmap):
            center_x, center_y = BaseOnlineSolver._get_cell_loc(self.flp_manager.domain, idx)
            element_stiffness[idx] = self.stiffness_dict[self.flp_manager.loc_names[center_y][center_x]]
        assemble_matrix_(self._A, element_stiffness, dofmap)
        self._A.assemble()
        for bc in self.bcs:
            dofs, _ = bc._cpp_object.dof_indices()
            self._A.zeroRowsLocal(dofs, diag=1)
        
        element_loads = np.zeros((num_cells, self.interp_pt_num*3), dtype=default_scalar_type)
        if self.t_func is None:
            assert self.simdb.mode == 1
            coeffs = np.full(self.interp_pt_num, self.simdb.amb_temp-self.simdb.ref_temp)
            for idx, pos in enumerate(dofmap):
                center_x, center_y = BaseOnlineSolver._get_cell_loc(self.flp_manager.domain, idx)
                element_loads[idx] = coeffs@self.load_dict[self.flp_manager.loc_names[center_y][center_x]][:self.interp_pt_num, :]
        else:
            assert self.simdb.mode == 2
            for idx, pos in enumerate(dofmap):
                center_x, center_y = BaseOnlineSolver._get_cell_loc(self.flp_manager.domain, idx)
                coeffs = np.concatenate((self.t_func.x.array[pos] - self.simdb.ref_temp,
                                         self.flp_manager.loc_powers[center_y][center_x]))
                element_loads[idx] = coeffs@self.load_dict[self.flp_manager.loc_names[center_y][center_x]]
        assemble_vector_(self._b, element_loads, dofmap)
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        for bc in self.bcs:
            bc.set(self._b.array_w)
        

        if self.comm.rank == 0:
            func_space = self.solution_func.function_space
            total_num_dof = func_space.dofmap.index_map.size_global * func_space.dofmap.index_map_bs
            print(f"Total number of Dofs: {total_num_dof}")
            print("Finish assembling.\n")


        