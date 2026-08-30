from dolfinx import fem, geometry, default_scalar_type
import ufl
import numpy as np
import h5py
from pathlib import Path
from collections import defaultdict
from typing import Optional

from Offline import MeshManager
from Online import FlpManager
from utils import epsilon, unfold_idx_by_dim, locate_grid_points

class BasePost:
    def __init__(
            self,
            mesh_manager_dict: dict[str, MeshManager],
            flp_manager: FlpManager,
        ) -> None:
        self.mesh_manager_dict = mesh_manager_dict
        self.simdb = list(mesh_manager_dict.values())[0].simdb

        self.flp_manager = flp_manager
        self.comm = flp_manager.comm


        self.interp_num = self.simdb.interp_num
        self.interp_pt_num = self.simdb.interp_num**3 - (self.simdb.interp_num-2)**3

    @staticmethod
    def _assign_properties(mesh_manager: MeshManager, func: fem.Function) -> None:
        """
        Assign materials to the mesh according to the cell tags.

        Args:
            mesh_manager: MeshManager instance.
            mat_func: Dolfinx function of the material.
            name: name of the material.
        """
        name = func.name
        assert name in ['stiffness', 'cte', 'ctc']
        func.x.array[:] = 0.0
        func.x.scatter_forward()

        size = func.function_space.value_size
        for shape in mesh_manager.shape_list:
            cells = mesh_manager.cell_tags.find(shape.tag)
            func.x.array[unfold_idx_by_dim(cells, size)] = np.concatenate(
                [mesh_manager.simdb.mat_dict[shape.filling_mat][name].flatten() for _ in cells],
                dtype=default_scalar_type)

    def _read_rom(self) -> None:
        pass

    def _restore_local_values(self, idx: int) -> None:
        pass

    def set_up_values(self, value_names: list[str]) -> dict[str, tuple[fem.Function, Optional[fem.Expression]]]:
        pass

    def extract_plane_values(self, z_coord: float, local_reso: int, value_dict: dict[str, dict[str, tuple[fem.Function, Optional[fem.Expression]]]], dir_name: Path) -> None:
        local_info_dict = defaultdict(list)
        for structure_name in self.mesh_manager_dict:
            mesh_manager = self.mesh_manager_dict[structure_name]
            local_bb_tree = geometry.bb_tree(mesh_manager.msh, 3)
            pts = locate_grid_points(
                mesh_manager.bb_box[0], 
                mesh_manager.bb_box[1], 
                z_coord,
                mesh_manager.bb_box[3] - mesh_manager.bb_box[0],
                mesh_manager.bb_box[4] - mesh_manager.bb_box[1],
                local_reso,
                local_reso
                )
            cell_candidates = geometry.compute_collisions_points(local_bb_tree, pts.T)
            colliding_cells = geometry.compute_colliding_cells(mesh_manager.msh, cell_candidates, pts.T)
            cells = [colliding_cells.links(i)[0] for i in range(len(pts.T))]
            
            local_info_dict[structure_name].extend([pts, cells])

        idx = 0
        while True:
            rst_path = (dir_name/("values"+str(idx))).with_suffix(".h5")
            if not rst_path.exists():
                break
            else:
                idx += 1
        with h5py.File(rst_path, 'a', driver="mpio", comm=self.comm) as f:
            value_handle_dict = {}
            for value_name in value_dict.keys():
                size = value_dict[value_name][self.simdb.default_name][0].function_space.value_size
                tmp = f.create_dataset(value_name,
                                       shape=(self.flp_manager.size_y, self.flp_manager.size_x, size*(local_reso**2)),
                                       dtype=default_scalar_type)
                value_handle_dict[value_name] = tmp
                
            num_cells = self.flp_manager.domain.topology.index_map(3).size_local
            coords = self.flp_manager.domain.geometry.x
            for i in range(num_cells):
                cell_coords = coords[self.flp_manager.domain.geometry.dofmap[i], :]
                center_x = int(sum(cell_coords)[0]/8)
                center_y = int(sum(cell_coords)[1]/8)
                name = self.flp_manager.loc_names[center_y][center_x]
                pts, cells = local_info_dict[name]

                self._restore_local_values(i, center_x, center_y)

                for value_name in value_dict.keys():
                    value_func, value_expr = value_dict[value_name][name]
                    if value_expr is not None:
                        value_func.interpolate(value_expr)
                    values = value_func.eval(pts.T, cells)
                    if value_name == "disp":
                        value_handle_dict[value_name][center_y, center_x, :] = values.flatten() / (10**self.simdb.unit)
                    else:
                        value_handle_dict[value_name][center_y, center_x, :] = values.flatten()


class CouplePost(BasePost):
    def __init__(
            self, 
            mesh_manager_dict: dict[str, MeshManager], 
            flp_manager: FlpManager,
            t_func: fem.Function,
            u_func: fem.Function
        ):
        super().__init__(mesh_manager_dict, flp_manager)

        self.global_t_func = t_func
        self.local_t_func_dict: dict[str, fem.Function] = {}
        self.ctc_dict: dict[str, fem.Function] = {}

        self.global_u_func = u_func
        self.local_u_func_dict: dict[str, fem.Function] = {}
        self.ref_temp_dict: dict[str, fem.Function] = {}
        self.stiffness_dict: dict[str, fem.Function] = {}
        self.cte_dict: dict[str, fem.Function] = {}

        for name in mesh_manager_dict:
            mesh_manager = mesh_manager_dict[name]

            self.local_t_func_dict[name] = fem.Function(fem.functionspace(mesh_manager.msh, ("Lagrange", 2)))
            self.ctc_dict[name] = fem.Function(fem.functionspace(mesh_manager.msh, ("DG", 0, (3,3))), name="ctc")
            BasePost._assign_properties(mesh_manager, self.ctc_dict[name])
            
            if self.simdb.mode == 2:
                self.local_u_func_dict[name] = fem.Function(fem.functionspace(mesh_manager.msh, ("Lagrange", 2, (3,))))
                self.ref_temp_dict[name] = fem.Constant(mesh_manager.msh, self.simdb.ref_temp)
                self.stiffness_dict[name] = fem.Function(fem.functionspace(mesh_manager.msh, ("DG", 0, (6,6))), name="stiffness")
                BasePost._assign_properties(mesh_manager, self.stiffness_dict[name])
                self.cte_dict[name] = fem.Function(fem.functionspace(mesh_manager.msh, ("DG", 0, (6,))), name="cte")
                BasePost._assign_properties(mesh_manager, self.cte_dict[name])


        self.mode_t_dict = {}
        self.mode_p_dict = {}
        self.mode_u_dict = {}
        self.mode_c_dict = {}
        self._read_rom()
    
    def _read_rom(self) -> None:
        for rom_path in self.flp_manager.rom_dir.rglob('*.h5'):
            if rom_path.stem in self.simdb.structure_names:
                with h5py.File(rom_path, 'r', driver='mpio', comm=self.comm) as f:
                    rom = f["rom_t"]
                    self.mode_t_dict[rom_path.stem] = rom["mode"][:]
                    self.mode_p_dict[rom_path.stem] = rom["mode_p"][:]
                    if self.simdb.mode == 2:
                        rom = f["rom_u"]
                        self.mode_u_dict[rom_path.stem] = rom["mode"][:]
                        self.mode_c_dict[rom_path.stem] = rom["mode_c"][:]

    def set_up_values(self, value_names: list[str]) -> dict[str, tuple[fem.Function, Optional[fem.Expression]]]:
        value_dict: dict[str, dict[str, tuple[fem.Function, Optional[fem.Expression]]]] = defaultdict(dict)

        for name in self.local_t_func_dict.keys():
            local_t_func = self.local_t_func_dict[name]
            ctc = self.ctc_dict[name]

            local_u_func = self.local_u_func_dict[name]
            ref_temp = self.ref_temp_dict[name]
            stiffness = self.stiffness_dict[name]
            cte = self.cte_dict[name]
            strain = epsilon(local_u_func)
            stress = ufl.dot(stiffness, (strain - cte * (local_t_func-ref_temp)))

            local_msh = self.mesh_manager_dict[name].msh

            if "temp" in value_names:
                value_dict["temp"][name] = (local_t_func, None)

            if "h_flux" in value_names:
                heat_flux = -ctc*ufl.grad(local_t_func)
                V_flux = fem.functionspace(local_msh, ("DG", 1, (3,)))
                flux_expr = fem.Expression(heat_flux, V_flux.element.interpolation_points)
                heat_flux_func = fem.Function(V_flux)
                value_dict["h_flux"][name] = (heat_flux_func, flux_expr)

            if self.simdb.mode == 2:
                if "disp" in value_names:
                    value_dict["disp"][name] = (local_u_func, None)

                if "strain" in value_names:
                    V_strain = fem.functionspace(local_msh, ("DG", 1, (6,)))
                    strain_expr = fem.Expression(strain, V_strain.element.interpolation_points)
                    strain_func = fem.Function(V_strain)
                    value_dict["strain"][name] = (strain_func, strain_expr)

                if "stress" in value_names:
                    V_stress = fem.functionspace(local_msh, ("DG", 1, (6,)))
                    stress_expr = fem.Expression(stress, V_stress.element.interpolation_points)
                    stress_func = fem.Function(V_stress)
                    value_dict["stress"][name] = (stress_func, stress_expr)

        return value_dict
    
    def _restore_local_values(self, idx: int, center_x: int, center_y: int) -> None:
        name = self.flp_manager.loc_names[center_y][center_x]
        power_values = self.flp_manager.loc_powers[center_y][center_x]

        dofmap = self.global_t_func.function_space.dofmap.list
        node_t_values = self.global_t_func.x.array[dofmap[idx]]
        local_t_values = node_t_values@self.mode_t_dict[name] + power_values@self.mode_p_dict[name]
        self.local_t_func_dict[name].x.array[:] = local_t_values

        if self.simdb.mode == 2:
            node_u_values = self.global_u_func.x.array[unfold_idx_by_dim(dofmap[idx], 3)]
            local_u_values = (node_u_values@self.mode_u_dict[name]
                            + np.concatenate(((node_t_values-self.simdb.ref_temp), power_values))@self.mode_c_dict[name])
            self.local_u_func_dict[name].x.array[:] = local_u_values



class MechanicalPost(BasePost):
    def __init__(
            self, 
            mesh_manager_dict: dict[str, MeshManager], 
            flp_manager: FlpManager, 
            u_func: fem.Function):
        super().__init__(mesh_manager_dict, flp_manager)

        self.global_u_func = u_func
        self.local_u_func_dict: dict[str, fem.Function] = {}
        self.ref_temp_dict: dict[str, fem.Function] = {}
        self.amb_temp_dict: dict[str, fem.Function] = {}
        self.stiffness_dict: dict[str, fem.Function] = {}
        self.cte_dict: dict[str, fem.Function] = {}

        for name in mesh_manager_dict:
            mesh_manager = mesh_manager_dict[name]

            self.local_u_func_dict[name] = fem.Function(fem.functionspace(mesh_manager.msh, ("Lagrange", 2, (3,))))
            self.ref_temp_dict[name] = fem.Constant(mesh_manager.msh, self.simdb.ref_temp)
            self.amb_temp_dict[name] = fem.Constant(mesh_manager.msh, self.simdb.amb_temp)
            self.stiffness_dict[name] = fem.Function(fem.functionspace(mesh_manager.msh, ("DG", 0, (6,6))), name="stiffness")
            BasePost._assign_properties(mesh_manager, self.stiffness_dict[name])
            self.cte_dict[name] = fem.Function(fem.functionspace(mesh_manager.msh, ("DG", 0, (6,))), name="cte")
            BasePost._assign_properties(mesh_manager, self.cte_dict[name])

        self.mode_u_dict = {}
        self.mode_c_dict = {}
        self._read_rom()
    
    def _read_rom(self) -> None:
        for rom_path in self.flp_manager.rom_dir.rglob('*.h5'):
            if rom_path.stem in self.simdb.structure_names:
                with h5py.File(rom_path, 'r', driver='mpio', comm=self.comm) as f:
                    rom = f["rom_u"]
                    self.mode_u_dict[rom_path.stem] = rom["mode"][:]
                    self.mode_c_dict[rom_path.stem] = rom["mode_c"][:]

    def set_up_values(self, value_names: list[str]) -> dict[str, tuple[fem.Function, Optional[fem.Expression]]]:
        value_dict: dict[str, dict[str, tuple[fem.Function, Optional[fem.Expression]]]] = defaultdict(dict)

        for name in self.local_t_func_dict.keys():
            local_u_func = self.local_u_func_dict[name]
            ref_temp = self.ref_temp_dict[name]
            amb_temp = self.amb_temp_dict[name]
            stiffness = self.stiffness_dict[name]
            cte = self.cte_dict[name]
            strain = epsilon(local_u_func)
            stress = ufl.dot(stiffness, (strain - cte * (amb_temp-ref_temp)))

            local_msh = self.mesh_manager_dict[name].msh

            if "disp" in value_names:
                value_dict["disp"][name] = (local_u_func, None)

            if "strain" in value_names:
                    V_strain = fem.functionspace(local_msh, ("DG", 1, (6,)))
                    strain_expr = fem.Expression(strain, V_strain.element.interpolation_points)
                    strain_func = fem.Function(V_strain)
                    value_dict["strain"][name] = (strain_func, strain_expr)

            if "stress" in value_names:
                V_stress = fem.functionspace(local_msh, ("DG", 1, (6,)))
                stress_expr = fem.Expression(stress, V_stress.element.interpolation_points)
                stress_func = fem.Function(V_stress)
                value_dict["stress"][name] = (stress_func, stress_expr)

        return value_dict
    
    def _restore_local_values(self, idx: int, center_x: int, center_y: int) -> None:
        name = self.flp_manager.loc_names[center_y][center_x]

        dofmap = self.global_u_func.function_space.dofmap.list
        node_u_values = self.global_u_func.x.array[unfold_idx_by_dim(dofmap[idx], 3)]
        local_u_values = (node_u_values@self.mode_u_dict[name]
                          + (self.simdb.amb_temp - self.simdb.ref_temp)*np.sum(self.mode_c_dict[name][:self.interp_pt_num, :], 0))
        self.local_u_func_dict[name].x.array[:] = local_u_values
        


        


