from dolfinx import fem, mesh

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import griddata
from pathlib import Path
from typing import Optional

from Online import FlpManager
from pseudo import create_pseudo_element
from utils import unfold_idx_by_dim

class BaseMapper:
    def __init__(self, 
                 flp_manager: FlpManager, 
                 substrate_box: tuple[float, float, float, float, float, float]) -> None:
        self.flp_manager = flp_manager
        self.simdb = flp_manager.simdb
        self.comm = flp_manager.comm

        self.x_origin = self.simdb.x_origin*10**self.simdb.unit
        self.y_origin = self.simdb.y_origin*10**self.simdb.unit
        self.z_origin = self.simdb.z_origin*10**self.simdb.unit

        self.substrate_box = substrate_box

        self.array_box = (
            self.x_origin,
            self.y_origin,
            self.z_origin,
            self.flp_manager.size_x * (substrate_box[3] - substrate_box[0]) + self.x_origin,
            self.flp_manager.size_y * (substrate_box[4] - substrate_box[1]) + self.y_origin,
            (substrate_box[5] - substrate_box[2]) + self.z_origin
        )

        tdim = self.flp_manager.domain.topology.dim
        self.flp_manager.domain.topology.create_connectivity(tdim-1, tdim)
        self.boundary_facets = mesh.exterior_facet_indices(self.flp_manager.domain.topology)

    @staticmethod
    def _get_convex_hull_idx(coords: NDArray, box: tuple[float, float, float, float, float, float]) -> NDArray:
        idx = []
        for i, coord in enumerate(coords):
            x, y, z = coord
            if x >= box[0] - 10*(box[3] - box[0]) and x <= box[3] + 10*(box[3] - box[0]):
                if y >= box[1] - 10*(box[4] - box[1]) and y <= box[4] + 10*(box[4] - box[1]):
                    if z >= box[2] - 0.5*(box[5] - box[2]) and z <= box[5] + 0.5*(box[5] - box[2]):
                        idx.append(i)

        return np.array(idx)


    def get_bcs(self, rst_file: Optional[Path]) -> list[fem.DirichletBC]:
        pass

class ThermalMapper(BaseMapper):
    def __init__(self, 
                 flp_manager: FlpManager, 
                 substrate_box: tuple[float, float, float, float, float, float]):
        super().__init__(flp_manager, substrate_box)

        pseudo_element = create_pseudo_element(self.simdb.interp_num, 1)
        self.bc_func = fem.Function(fem.functionspace(flp_manager.domain, pseudo_element))

    def get_bcs(self, rst_file: Path) -> list[fem.DirichletBC]:
        rst_data = np.load(rst_file)

        pts = rst_data[:, :3] * (10**self.simdb.unit)
        selected_idx = BaseMapper._get_convex_hull_idx(pts, self.array_box)
        pts = pts[selected_idx, :]
        temp_values = rst_data[selected_idx, 3]

        func_space = self.bc_func.function_space
        size_local = func_space.dofmap.index_map.size_local
        coords = func_space.tabulate_dof_coordinates()
        boundary_dofs = fem.locate_dofs_topological(func_space, 2, self.boundary_facets)
        boundary_dofs = boundary_dofs[boundary_dofs < size_local]

        boundary_coords = coords[boundary_dofs, :]

        boundary_coords[:, 0] = boundary_coords[:, 0] * (self.substrate_box[3] - self.substrate_box[0]) + self.x_origin
        boundary_coords[:, 1] = boundary_coords[:, 1] * (self.substrate_box[4] - self.substrate_box[1]) + self.y_origin
        boundary_coords[:, 2] = boundary_coords[:, 2] * (self.substrate_box[5] - self.substrate_box[2]) + self.z_origin

        bc_values = griddata(pts, temp_values, boundary_coords, method='linear')
        self.bc_func.x.array[boundary_dofs] = bc_values

        self.bc_func.x.scatter_forward()

        bc = fem.dirichletbc(self.bc_func, boundary_dofs)

        return [bc]

    
class MechanicalMapper(BaseMapper):
    def __init__(self, 
                 flp_manager: FlpManager, 
                 substrate_box: tuple[float, float, float, float, float, float]):
        super().__init__(flp_manager, substrate_box)

        pseudo_element = create_pseudo_element(self.simdb.interp_num, 3)
        self.bc_func = fem.Function(fem.functionspace(flp_manager.domain, pseudo_element))

    def get_bcs(self, rst_file: Path) -> list[fem.DirichletBC]:
        rst_data = np.load(rst_file)

        pts = rst_data[:, :3] * (10**self.simdb.unit)
        selected_idx = BaseMapper._get_convex_hull_idx(pts, self.array_box)
        pts = pts[selected_idx, :]
        
        u_values = rst_data[selected_idx, 3]
        v_values = rst_data[selected_idx, 4]
        w_values = rst_data[selected_idx, 5]

        func_space = self.bc_func.function_space
        size_local = func_space.dofmap.index_map.size_local
        coords = func_space.tabulate_dof_coordinates()
        boundary_dofs = fem.locate_dofs_topological(func_space, 2, self.boundary_facets)
        boundary_dofs = boundary_dofs[boundary_dofs < size_local]

        boundary_coords = coords[boundary_dofs, :]

        boundary_coords[:, 0] = boundary_coords[:, 0] * (self.substrate_box[3] - self.substrate_box[0]) + self.x_origin
        boundary_coords[:, 1] = boundary_coords[:, 1] * (self.substrate_box[4] - self.substrate_box[1]) + self.y_origin
        boundary_coords[:, 2] = boundary_coords[:, 2] * (self.substrate_box[5] - self.substrate_box[2]) + self.z_origin

        bc_values = np.vstack((griddata(pts, u_values, boundary_coords, method='linear'),
                                griddata(pts, v_values, boundary_coords, method='linear'),
                                griddata(pts, w_values, boundary_coords, method='linear')))
        bc_values = bc_values.T.flatten()
        self.bc_func.x.array[unfold_idx_by_dim(boundary_dofs, 3)] = bc_values

        self.bc_func.x.scatter_forward()

        bc = fem.dirichletbc(self.bc_func, boundary_dofs)

        return [bc]

