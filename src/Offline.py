import gmsh
from dolfinx import fem, mesh, la, io, common, default_scalar_type
from dolfinx.fem.petsc import LinearProblem, create_vector, assemble_matrix, assemble_vector
from dolfinx.la.petsc import create_vector_wrap
from petsc4py import PETSc
import ufl
from basix import create_element, ElementFamily, CellType, LagrangeVariant
import numpy as np
from xml.etree.ElementTree import Element
from mpi4py import MPI
import h5py
from pathlib import Path
from tqdm import tqdm
import sys
from typing import Optional

from Parser import DataCollection
from utils import is_containing, unfold_idx_by_dim, epsilon

class Shape:
    """
    Represents a physical shape in the local model.
    """
    def __init__(self, content: Element, simdb: DataCollection) -> None:
        self.simdb = simdb
        self._u = simdb.unit
        self._p = simdb.precision
        self._eps = simdb.eps

        self.content = content

        if "Name" in content.attrib:
            self.name = content.attrib['Name']
        else:
            print(f"ERROR: 'Name' must be specified for all Shapes!")
            sys.exit(1)

        if "Type" in content.attrib:
            component_type = content.attrib["Type"].lower()
            if component_type in ('box', 'cylinder', 'sphere'):
                self.type = component_type
            else:
                print(f"ERROR: Type of Shape {self.name} must be one of: "
                    "<box, cylinder, sphere>!")
                sys.exit(1)
        else:
            print(f"ERROR: Missing 'Type' attribute for Shape {self.name}!")
            sys.exit(1)

        if "FillingMaterial" in content.attrib:
            self.filling_mat = content.attrib["FillingMaterial"]
        else:
            print(f"ERROR: Missing 'FillingMaterial' attribute for Shape {self.name}!")
            sys.exit(1)
        
        self.power_dens: Optional[float] = None
        if "PowerOn" in content.attrib:
            if content.attrib["PowerOn"].lower() == "true":
                self.power_dens = 0.0
            else:
                self.power_dens = None
        else:
            self.power_dens = None
            
        
        flag = False
        for element in self.content:
            tag = element.tag.lower()
            if tag == 'geometry':
                flag = True
                self._set_geometry(element)
                break
        if not flag:
            print(f"ERROR: Missing 'Geometry' element for Shape {self.name}!")
            sys.exit(1)

        self.tag: Optional[int] = None


    def _set_geometry(self, content: Element) -> None:
        if self.type == 'box':
            try:
                self.length: float = round(float(content.attrib['Length'])*10**self._u, self._p-self._u)
                self.width: float = round(float(content.attrib['Width'])*10**self._u, self._p-self._u)
                self.height: float = round(float(content.attrib['Height'])*10**self._u, self._p-self._u)
                self.origin = content.attrib['OriginXYZ'].lower()

                if self.origin == 'OriginXYZ':
                    self.x_origin, self.y_origin, self.z_origin = 0.0, 0.0, 0.0
                elif self.origin.startswith('('):
                    try:
                        x_str, y_str, z_str = self.origin[1:-1].split(",")
                        self.x_origin: float = round(float(x_str.strip())*10**self._u, self._p-self._u)
                        self.y_origin: float = round(float(y_str.strip())*10**self._u, self._p-self._u)
                        self.z_origin: float = round(float(z_str.strip())*10**self._u, self._p-self._u)
                    except ValueError:
                        print(f"ERROR: Invalid coordinate format in OriginXYZ for Shape {self.name}!")
                        sys.exit(1)
                else:
                    print(f"ERROR: OriginXYZ of Shape {self.name} must be 'origin' or '(X,Y,Z)'!")
                    sys.exit(1)
            except:
                print(f"ERROR: <Length, Width, Height, OriginXYZ> must be provided with valid values for Shape {self.name}!")
                sys.exit(1)

            self.bb_box = (
                self.x_origin,
                self.y_origin,
                self.z_origin,
                self.x_origin + self.length,
                self.y_origin + self.width,
                self.z_origin + self.height
            )

        elif self.type == 'cylinder':
            try:
                self.radius: float = round(float(content.attrib['Radius'])*10**self._u, self._p-self._u)
                self.height: float = round(float(content.attrib['Height'])*10**self._u, self._p-self._u)
                self.origin = content.attrib['OriginXYZ'].lower()

                if self.origin == 'OriginXYZ':
                    self.x_origin, self.y_origin, self.z_origin = 0.0, 0.0, 0.0
                elif self.origin.startswith('('):
                    try:
                        x_str, y_str, z_str = self.origin[1:-1].split(",")
                        self.x_origin: float = round(float(x_str.strip())*10**self._u, self._p-self._u)
                        self.y_origin: float = round(float(y_str.strip())*10**self._u, self._p-self._u)
                        self.z_origin: float = round(float(z_str.strip())*10**self._u, self._p-self._u)
                    except ValueError:
                        print(f"ERROR: Invalid coordinate format in OriginXYZ for Shape {self.name}!")
                        sys.exit(1)
                else:
                    print(f"ERROR: OriginXYZ of Shape {self.name} must be 'origin' or '(X,Y,Z)'!")
                    sys.exit(1)
            except:
                print(f"ERROR: <Radius, Height, OriginXYZ> must be provided with valid values for Shape {self.name}!")
                sys.exit(1)

            self.bb_box = (
                self.x_origin - self.radius,
                self.y_origin - self.radius,
                self.z_origin,
                self.x_origin + self.radius,
                self.y_origin + self.radius,
                self.z_origin + self.height
            )
    
        elif self.type == 'sphere':
            try:
                self.radius: float = round(float(content.attrib['Radius'])*10**self._u, self._p-self._u)
                self.origin = content.attrib['OriginXYZ'].lower()

                if self.origin == 'OriginXYZ':
                    self.x_origin, self.y_origin, self.z_origin = 0.0, 0.0, 0.0
                elif self.origin.startswith('('):
                    try:
                        x_str, y_str, z_str = self.origin[1:-1].split(",")
                        self.x_origin: float = round(float(x_str.strip())*10**self._u, self._p-self._u)
                        self.y_origin: float = round(float(y_str.strip())*10**self._u, self._p-self._u)
                        self.z_origin: float = round(float(z_str.strip())*10**self._u, self._p-self._u)
                    except ValueError:
                        print(f"ERROR: Invalid coordinate format in OriginXYZ for Shape {self.name}!")
                        sys.exit(1)
                else:
                    print(f"ERROR: OriginXYZ of Shape {self.name} must be 'origin' or '(X,Y,Z)'!")
                    sys.exit(1)
            except:
                print(f"ERROR: <Radius, OriginXYZ> must be provided with valid values for Shape {self.name}!")
                sys.exit(1)

            self.bb_box = (
                self.x_origin - self.radius,
                self.y_origin - self.radius,
                self.z_origin - self.radius,
                self.x_origin + self.radius,
                self.y_origin + self.radius,
                self.z_origin + self.radius
            )

class MeshManager:
    """
    Manages local mesh creation.
    """
    def __init__(self, elements: list[Element], simdb: DataCollection, comm: MPI.Comm) -> None:
        self.simdb = simdb
        self._u = self.simdb.unit
        self._p = self.simdb.precision
        self._eps = self.simdb.eps

        self.comm = comm
        self.msh = None
        self.cell_tags = None
        self.bb_box: tuple[float] = None

        self.shape_list: list[Shape] = []
        for element in elements:
            shape = Shape(element, self.simdb)
            shape.tag = len(self.shape_list) + 1
            self.shape_list.append(shape)
        
        self.power_item: list[int] = []
        for idx, shape in enumerate(self.shape_list):
            if shape.power_dens is not None:
                self.power_item.append(idx)
    
    def load_mesh(self, mesh_path: Path) -> None:
        """
        Load mesh of the structure from the file and distribute it across ranks.
        """
        def _load_mesh() -> gmsh.model:
            gmsh.model.add('structure')
            gmsh.option.setNumber("General.Verbosity", 2)

            gmsh.merge(str(mesh_path))

            bb_box = gmsh.model.get_bounding_box(-1, -1)
            self.bb_box = tuple([round(coord, self._p-self._u) for coord in bb_box])

            return gmsh.model
        
        model = None
        gmsh.initialize()
        if self.comm.rank == 0:
            model = _load_mesh()
        self.comm.barrier()
        mesh_data = io.gmsh.model_to_mesh(model, self.comm, 0, 3, dtype=default_scalar_type)
        self.msh = mesh_data.mesh
        self.cell_tags = mesh_data.cell_tags
        gmsh.finalize()
        self.comm.barrier()
        

        
    def generate_mesh(self, mesh_path: Path, mesh_size: float) -> None:
        """
        Generate mesh of the structure and distribute it across ranks.
        """
        def _generate_mesh() -> gmsh.model:
            """
            Generate the Gmsh model.\n
            Must call gmsh.initialize() before calling this function.
            """
            gmsh.model.add('structure')
            gmsh.option.setNumber("General.Verbosity", 2)

            for shape in self.shape_list:
                if shape.type == 'box':
                    gmsh.model.occ.addBox(shape.x_origin, shape.y_origin, shape.z_origin, shape.length, shape.width, shape.height)
                elif shape.type == 'cylinder':
                    gmsh.model.occ.addCylinder(shape.x_origin, shape.y_origin, shape.z_origin, 0, 0, shape.height, shape.radius)
                elif shape.type == 'sphere':
                    gmsh.model.occ.addSphere(shape.x_origin, shape.y_origin, shape.z_origin, shape.radius)
            if len(self.shape_list) > 1:
                gmsh.model.occ.fragment(gmsh.model.occ.getEntities(3), [])
            gmsh.model.occ.synchronize()

            dimtags = gmsh.model.occ.getEntities(3)
            tags = [tag for _, tag in dimtags]
            tag_box_dict = {}
            for tag in tags:
                tag_box_dict[tag] = gmsh.model.getBoundingBox(3, tag)

            bb_box = gmsh.model.get_bounding_box(-1, -1)
            self.bb_box = tuple([round(coord, self._p-self._u) for coord in bb_box])

            for shape in self.shape_list[::-1]:
                shape_tags = []
                for tag in list(tag_box_dict.keys()):
                    if is_containing(shape.bb_box, tag_box_dict[tag], self._p - self._u):
                        shape_tags.append(tag)
                        tag_box_dict.pop(tag)
                gmsh.model.addPhysicalGroup(3, shape_tags, tag=shape.tag)

            # set mesh size
            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), round(mesh_size*10**self._u, self._p-self._u))
            
            gmsh.model.mesh.generate(3)
            gmsh.write(str(mesh_path))

            # obtain mesh information

            element_types = gmsh.model.mesh.getElementTypes(dim=3)
            assert len(element_types) == 1
            element_type = element_types[0]

            name, _, _, _, _, _ = gmsh.model.mesh.getElementProperties(element_type)
            element_tags, _ = gmsh.model.mesh.getElementsByType(element_type)
            print(f"Element type: {name.split(' ')[0]}")
            print(f"Total number of elements: {len(element_tags)}")
            
            element_nodes, _, _ = gmsh.model.mesh.getNodes(dim=3, includeBoundary=True, returnParametricCoord=False)
            element_nodes = np.unique(element_nodes)
            print(f"Total number of nodes: {len(element_nodes)}")

            print(f"Mesh quality:")
            q = gmsh.model.mesh.getElementQualities(element_tags, "minSICN")
            print(f"SICN: {np.max(q):.6f} (max) {np.min(q):.6f} (min) {np.mean(q):.6f} (avg)")

            return gmsh.model
        
        model = None
        gmsh.initialize()
        if self.comm.rank == 0:
            print(f"Start meshing ...")
            model = _generate_mesh()
            print(f"Finish meshing.\n")
        self.comm.barrier()
        model = self.comm.bcast(model, root=0)
        mesh_data = io.gmsh.model_to_mesh(model, self.comm, 0, 3, dtype=default_scalar_type)
        self.msh = mesh_data.mesh
        self.cell_tags = mesh_data.cell_tags
        gmsh.finalize()
        self.comm.barrier()

class BaseRomBuilder:
    def __init__(self, mesh_manager: MeshManager, rom_path: Path) -> None:
        self._u = mesh_manager._u
        self._p = mesh_manager._p
        self._eps = mesh_manager._eps

        self.mesh_manager = mesh_manager
        self.simdb = mesh_manager.simdb
        self.msh = mesh_manager.msh
        self.cell_tags = mesh_manager.cell_tags
        self.comm = mesh_manager.comm

        self.dx = ufl.Measure("dx", self.msh, subdomain_data=self.cell_tags)
        self.solution_func: Optional[fem.Function] = None

        self.rom_path = rom_path

        self._solver = None
        self._a = None
        self._A = None
        self._L = None
        self._b = None
        self._x = None

        self.interp_pt_num = self.simdb.interp_num**3 - (self.simdb.interp_num-2)**3
        
        self.boundary_dofs = None
        self.bc_values = None
        self._get_boundary_values()

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
        assert name in ['stiffness', 'cte', 'ctc', 'power']
        func.x.array[:] = 0.0
        func.x.scatter_forward()

        if name in ['stiffness', 'cte', 'ctc']:
            size = func.function_space.value_size
            for shape in mesh_manager.shape_list:
                cells = mesh_manager.cell_tags.find(shape.tag)
                func.x.array[unfold_idx_by_dim(cells, size)] = np.concatenate(
                    [mesh_manager.simdb.mat_dict[shape.filling_mat][name].flatten() for _ in cells],
                    dtype=default_scalar_type)
        elif name == 'power':
            for shape in mesh_manager.shape_list:
                if shape.power_dens is not None:
                    cells = mesh_manager.cell_tags.find(shape.tag)
                    func.x.array[cells] = np.full_like(cells, shape.power_dens, dtype=default_scalar_type)

    def _get_boundary_values(self) -> None:
        func_space = fem.functionspace(self.msh, ("Lagrange", 2))
        tdim = self.msh.topology.dim
        self.msh.topology.create_connectivity(tdim-1, tdim)
        boundary_facets = mesh.exterior_facet_indices(self.msh.topology)
        boundary_dofs = fem.locate_dofs_topological(func_space, tdim-1, boundary_facets)
        boundary_dof_coords = func_space.tabulate_dof_coordinates()[boundary_dofs]
        
        bb_box = self.mesh_manager.bb_box
        boundary_dof_coords[:, 0] = (boundary_dof_coords[:, 0] - bb_box[0]) / (bb_box[3] - bb_box[0])
        boundary_dof_coords[:, 1] = (boundary_dof_coords[:, 1] - bb_box[1]) / (bb_box[4] - bb_box[1])
        boundary_dof_coords[:, 2] = (boundary_dof_coords[:, 2] - bb_box[2]) / (bb_box[5] - bb_box[2])

        lagrange = create_element(ElementFamily.P, CellType.hexahedron, self.simdb.interp_num-1, LagrangeVariant.equispaced)
        self.bc_values = lagrange.tabulate(0, boundary_dof_coords)[0, :, :, 0]
        self.boundary_dofs = boundary_dofs

    def problem(self) -> tuple[ufl.Form, ufl.Form]:
        return None, None
    
    def _pre_decomposition(self, a: ufl.Form, L: ufl.Form) -> None:
        pass
    
    def build_rom(self, a: ufl.Form, L: ufl.Form, rom_path: Path) -> None:
        pass

    def __del__(self) -> None:
        if self._A is not None:
            self._A.destroy()
        if self._b is not None:
            self._b.destroy()
        if self._x is not None:
            self._x.destroy()
        if self._solver is not None:
            self._solver.destroy()


class ThermalRomBuilder(BaseRomBuilder):
    def __init__(self, mesh_manager: MeshManager, rom_path: Path):
        super().__init__(mesh_manager, rom_path)

        V = fem.functionspace(self.msh, ("Lagrange", 2))
        self.solution_func = fem.Function(V)
        print(f"# of DoFs: {V.dofmap.index_map.size_global*V.dofmap.index_map_bs}")
        self._x = create_vector_wrap(self.solution_func.x)

        self.ctc = fem.Function(fem.functionspace(self.msh, ("DG", 0, (3, 3))), name="ctc")
        BaseRomBuilder._assign_properties(self.mesh_manager, self.ctc)
        self.power = fem.Function(fem.functionspace(self.msh, ("DG", 0)), name="power")
        BaseRomBuilder._assign_properties(self.mesh_manager, self.power)

    def problem(self) -> tuple[ufl.Form, ufl.Form]:
        func_space = self.solution_func.function_space
        t = ufl.TrialFunction(func_space); delta_t = ufl.TestFunction(func_space)
        weak_form = (ufl.inner(ufl.dot(self.ctc, ufl.grad(t)), ufl.grad(delta_t)) * self.dx
                    - self.power * delta_t * self.dx)
        
        return ufl.system(weak_form)
    
    def _pre_decomposition(self, a, L):
        self._a = fem.form(a, dtype = PETSc.ScalarType)
        self._L = fem.form(L, dtype = PETSc.ScalarType)
        self._A = assemble_matrix(self._a)
        self._A.assemble()
        self._A.zeroRowsLocal(self.boundary_dofs, diag=1.0)
        self._b = create_vector(self.solution_func.function_space)

        opts = PETSc.Options()
        opts["ksp_type"] = "preonly"
        opts["pc_type"] = "lu"
        opts["pc_factor_mat_solver_type"] = "mumps"

        self._solver = PETSc.KSP().create(self.comm)
        self._solver.setFromOptions()
        self._solver.setOperators(self._A)
        self._solver.setUp()

    
    def build_rom(self, a: ufl.Form, L: ufl.Form):
        if self.comm.rank == 0:
            print("Start LU decomposition ...")
        self._pre_decomposition(a, L)
        if self.comm.rank == 0:
            print("Finish LU decomposition.")

        imap = self.solution_func.function_space.dofmap.index_map
        size_global = imap.size_global
        print(f"# of DoFs: {size_global}")
        size_local = imap.size_local
        local_range = imap.local_range

        with h5py.File(self.rom_path, "a", driver="mpio", comm=self.comm) as f:
            f.attrs["interp_num"] = self.simdb.interp_num
            
            rom = f.create_group("rom_t")
            load = rom.create_dataset(
                "load", 
                shape=(len(self.mesh_manager.power_item), 
                       self.interp_pt_num), 
                dtype=default_scalar_type
            )
            stiffness = rom.create_dataset(
                "stiffness",
                shape=(self.interp_pt_num, 
                       self.interp_pt_num),
                dtype=default_scalar_type
            )
            mode = rom.create_dataset(
                "mode",
                shape=(self.interp_pt_num, 
                       size_global),
                dtype=default_scalar_type
            )
            mode_p = rom.create_dataset(
                "mode_p",
                shape=(len(self.mesh_manager.power_item), 
                       size_global),
                dtype=default_scalar_type
            )

            if self.comm.rank == 0:
                print("Start calculating thermal modes ...")
            for pt in tqdm(range(self.interp_pt_num)):
                self._b.zeroEntries()
                assemble_vector(self._b, self._L)
                self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                self._b.setValuesLocal(self.boundary_dofs, np.zeros(self.boundary_dofs.shape), PETSc.InsertMode.INSERT_VALUES)
                self._b.setValuesLocal(self.boundary_dofs, self.bc_values[:, pt], PETSc.InsertMode.INSERT_VALUES)
                self._b.assemble()
                self._b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                self._solver.solve(self._b, self._x)
                mode[pt, local_range[0]:local_range[1]] = self.solution_func.x.array[:size_local]
                self.comm.barrier()
            if self.comm.rank == 0:
                print("Finish calculating thermal modes.")

            if self.comm.rank == 0:
                print("Start calculating thermal stiffness ...")
            basis = mode[:]
            local_A = fem.assemble_matrix(self._a).to_scipy(ghosted=False)
            local_sum = basis[:, local_range[0]:local_range[1]]@local_A@basis.T
            if self.comm.rank == 0:
                total_sum = np.zeros((self.interp_pt_num, 
                                      self.interp_pt_num), 
                                     dtype=default_scalar_type)
            else:
                total_sum = None
            self.comm.Reduce(local_sum, total_sum, op=MPI.SUM, root=0)
            if total_sum is not None:
                stiffness[:] = total_sum
            if self.comm.rank == 0:
                print("Finish calculating thermal stiffness.")
            self.comm.barrier()
                
            if self.comm.rank == 0:
                print("Start calculating power modes and loads...")
            for i, idx in tqdm(enumerate(self.mesh_manager.power_item), total=len(self.mesh_manager.power_item)):
                self.mesh_manager.shape_list[idx].power_dens = 1.0
                BaseRomBuilder._assign_properties(self.mesh_manager, self.power)

                self._b.zeroEntries()
                assemble_vector(self._b, self._L)
                self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

                local_b = self._b.array_r[:size_local]
                local_sum = basis[:, local_range[0]:local_range[1]]@local_b
                if self.comm.rank == 0:
                    total_sum = np.zeros(self.interp_pt_num, dtype=default_scalar_type)
                else:
                    total_sum = None
                self.comm.Reduce(local_sum, total_sum, op=MPI.SUM, root=0)
                if total_sum is not None:
                    load[i, :] = total_sum

                self._b.setValuesLocal(self.boundary_dofs, np.zeros(self.boundary_dofs.shape), PETSc.InsertMode.INSERT_VALUES)
                self._b.assemble()
                self._b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                self._solver.solve(self._b, self._x)
                mode_p[i, local_range[0]:local_range[1]] = self.solution_func.x.array[:size_local]

                self.mesh_manager.shape_list[idx].power_dens = 0.0

                self.comm.barrier()
            if self.comm.rank == 0:
                print("Finish calculating power modes and loads.\n")

    def __del__(self) -> None:
        return super().__del__()


class MechanicalRomBuilder(BaseRomBuilder):
    def __init__(self, mesh_manager: MeshManager, rom_path: Path):
        super().__init__(mesh_manager, rom_path)

        V = fem.functionspace(self.msh, ("Lagrange", 2, (3,)))
        self.solution_func = fem.Function(V)
        print(f"# of DoFs: {V.dofmap.index_map.size_global*V.dofmap.index_map_bs}")
        self._x = create_vector_wrap(self.solution_func.x)
        self.t_func = fem.Function(fem.functionspace(self.msh, ("Lagrange", 2)))

        self.stiffness = fem.Function(fem.functionspace(self.msh, ("DG", 0, (6, 6))), name="stiffness")
        BaseRomBuilder._assign_properties(self.mesh_manager, self.stiffness)
        self.cte = fem.Function(fem.functionspace(self.msh, ("DG", 0, (6,))), name="cte")
        BaseRomBuilder._assign_properties(self.mesh_manager, self.cte)

    def problem(self) -> tuple[ufl.Form, ufl.Form]:
        func_space = self.solution_func.function_space
        disp = ufl.TrialFunction(func_space); delta_disp = ufl.TestFunction(func_space)
        strain = epsilon(disp); delta_strain = epsilon(delta_disp)
        stress = ufl.dot(self.stiffness, (strain - self.cte * self.t_func))
        weak_form = ufl.inner(stress, delta_strain) * self.dx

        return ufl.system(weak_form)
    
    def _pre_decomposition(self, a, L):
        self._a = fem.form(a, dtype = PETSc.ScalarType)
        self._L = fem.form(L, dtype = PETSc.ScalarType)
        self._A = assemble_matrix(self._a)
        self._A.assemble()
        self._A.zeroRowsLocal(self.boundary_dofs*3, diag=1.0)
        self._A.zeroRowsLocal(self.boundary_dofs*3+1, diag=1.0)
        self._A.zeroRowsLocal(self.boundary_dofs*3+2, diag=1.0)
        self._b = create_vector(self.solution_func.function_space)

        opts = PETSc.Options()
        opts["ksp_type"] = "preonly"
        opts["pc_type"] = "lu"
        opts["pc_factor_mat_solver_type"] = "mumps"

        self._solver = PETSc.KSP().create(self.comm)
        self._solver.setFromOptions()
        self._solver.setOperators(self._A)
        self._solver.setUp()
    
    def build_rom(self, a: ufl.Form, L: ufl.Form):
        if self.comm.rank == 0:
            print("Start LU decomposition ...")
        self._pre_decomposition(a, L)
        if self.comm.rank == 0:
            print("Finish LU decomposition.")

        imap = self.solution_func.function_space.dofmap.index_map
        size_global = imap.size_global
        size_local = imap.size_local
        local_range = imap.local_range

        with h5py.File(self.rom_path, "a", driver="mpio", comm=self.comm) as f:
            f.attrs["interp_num"] = self.simdb.interp_num
            
            rom = f.create_group("rom_u")
            stiffness = rom.create_dataset(
                "stiffness",
                shape=(self.interp_pt_num*3, 
                       self.interp_pt_num*3),
                dtype=default_scalar_type
            )
            mode = rom.create_dataset(
                "mode",
                shape=(self.interp_pt_num*3, 
                       size_global*3),
                dtype=default_scalar_type
            )

            if self.comm.rank == 0:
                print("Start calculating mechanical modes ...")
            for pt in tqdm(range(self.interp_pt_num)):
                for dim in range(3):
                    self._b.zeroEntries()
                    assemble_vector(self._b, self._L)
                    self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                    self._b.setValuesLocal(self.boundary_dofs*3, np.zeros(self.boundary_dofs.shape), PETSc.InsertMode.INSERT_VALUES)
                    self._b.setValuesLocal(self.boundary_dofs*3+1, np.zeros(self.boundary_dofs.shape), PETSc.InsertMode.INSERT_VALUES)
                    self._b.setValuesLocal(self.boundary_dofs*3+2, np.zeros(self.boundary_dofs.shape), PETSc.InsertMode.INSERT_VALUES)
                    self._b.setValuesLocal(self.boundary_dofs*3+dim, self.bc_values[:, pt], PETSc.InsertMode.INSERT_VALUES)
                    self._b.assemble()
                    self._b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    self._solver.solve(self._b, self._x)
                    mode[pt*3+dim, local_range[0]*3:local_range[1]*3] = self.solution_func.x.array[:size_local*3]

                    self.comm.barrier()
            if self.comm.rank == 0:
                print("Finish calculating mechanical modes ...")

            if self.comm.rank == 0:
                print("Start calculating mechanical stiffness ...")
            basis = mode[:]
            local_A = fem.assemble_matrix(self._a).to_scipy(ghosted=False)
            local_sum = basis[:, local_range[0]*3:local_range[1]*3]@local_A@basis.T
            if self.comm.rank == 0:
                total_sum = np.zeros((self.interp_pt_num*3, self.interp_pt_num*3), dtype=default_scalar_type)
            else:
                total_sum = None
            self.comm.Reduce(local_sum, total_sum, op=MPI.SUM, root=0)
            if total_sum is not None:
                stiffness[:] = total_sum
            if self.comm.rank == 0:
                print("Finish calculating mechanical stiffness.")

            self.comm.barrier()


            t_values = np.vstack((f["rom_t"]["mode"][:, local_range[0]:local_range[1]], 
                                    f["rom_t"]["mode_p"][:, local_range[0]:local_range[1]]))
            load = rom.create_dataset(
                "load", 
                shape=(t_values.shape[0], 
                        self.interp_pt_num*3), 
                dtype=default_scalar_type
            )
            mode_c = rom.create_dataset(
                "mode_c",
                shape = (t_values.shape[0], 
                         size_global*3),
                dtype=default_scalar_type
            )
            if self.comm.rank == 0:
                    print("Start calculating coupling modes and loads ...")
            for i, t_value in tqdm(enumerate(t_values), total=t_values.shape[0]):
                self.t_func.x.array[:size_local] = t_value
                self.t_func.x.scatter_forward()

                self._b.zeroEntries()
                assemble_vector(self._b, self._L)
                self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

                local_b = self._b.array_r[:size_local*3]
                local_sum = basis[:, local_range[0]*3:local_range[1]*3]@local_b
                if self.comm.rank == 0:
                    total_sum = np.zeros(self.interp_pt_num*3, dtype=default_scalar_type)
                else:
                    total_sum = None
                self.comm.Reduce(local_sum, total_sum, op=MPI.SUM, root=0)
                if total_sum is not None:
                    load[i, :] = total_sum

                self._b.setValuesLocal(self.boundary_dofs*3, np.zeros(self.boundary_dofs.shape), PETSc.InsertMode.INSERT_VALUES)
                self._b.setValuesLocal(self.boundary_dofs*3+1, np.zeros(self.boundary_dofs.shape), PETSc.InsertMode.INSERT_VALUES)
                self._b.setValuesLocal(self.boundary_dofs*3+2, np.zeros(self.boundary_dofs.shape), PETSc.InsertMode.INSERT_VALUES)
                self._b.assemble()
                self._b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                self._solver.solve(self._b, self._x)
                mode_c[i, local_range[0]*3:local_range[1]*3] = self.solution_func.x.array[:size_local*3]

                self.comm.barrier()
            if self.comm.rank == 0:
                print("Finish calculating coupling modes and loads.\n")
                    

    def __del__(self) -> None:
        return super().__del__()


def extract_thermal_properties(mesh_manager: MeshManager, a: ufl.Form, L: ufl.Form, prop_path: Path):
    with h5py.File(prop_path, 'a', driver='mpio', comm=mesh_manager.comm) as f:
        prop = f.create_group("prop")
        kappa = f.create_dataset("kappa", shape=(3,), dtype=default_scalar_type)

        func_space = fem.functionspace(mesh_manager.msh, ("Lagrange", 2))
        solution = fem.Function(func_space)
        for i in range(3):
            solution.x.array[:] = 0
            solution.x.scatter_forward()

            marker_0 = lambda x: np.isclose(x[i], mesh_manager.bb_box[i])
            bc_func_0 = fem.Function(func_space)
            constraint_0 = fem.dirichletbc(
                bc_func_0,
                fem.locate_dofs_geometrical(func_space, marker_0)
            )
            marker_1 = lambda x: np.isclose(x[i], mesh_manager.bb_box[i+3])
            bc_func_1 = fem.Function(func_space)
            bc_func_1.x.array[:] = 1.0
            bc_func_1.x.scatter_forward()
            constraint_1 = fem.dirichletbc(
                bc_func_1,
                fem.locate_dofs_geometrical(func_space, marker_1)
            )
            bcs = [constraint_0, constraint_1]
            LinearProblem(a, 
                          L, 
                          bcs, 
                          solution, 
                          petsc_options={"ksp_type":"preonly", "pc_type":"lu", "pc_factor_mat_solver_type":"mumps"})
            

def extract_mechanical_properties(mesh_manager: MeshManager, a: ufl.Form, L: ufl.Form, prop_path: Path):
    with h5py.File(prop_path, 'a', driver='mpio', comm=mesh_manager.comm) as f:
        prop = f.create_group("prop")
        young_modulus = f.create_dataset("young_modulus", shape=(3,), dtype=default_scalar_type)
        shear_modulus = f.create_dataset("shear_modulus", shape=(3,), dtype=default_scalar_type)
        poisson_ratio = f.create_dataset("poisson_ratio", shape=(3,), dtype=default_scalar_type)


        func_space = fem.functionspace(mesh_manager.msh, ("Lagrange", 2))
        solution = fem.Function(func_space)
        for i in range(3):
            solution.x.array[:] = 0
            solution.x.scatter_forward()

            marker_0 = lambda x: np.isclose(x[i], mesh_manager.bb_box[i])
            bc_func_0 = fem.Function(func_space)
            constraint_0 = fem.dirichletbc(
                bc_func_0,
                fem.locate_dofs_geometrical(func_space, marker_0)
            )
            marker_1 = lambda x: np.isclose(x[i], mesh_manager.bb_box[i+3])
            bc_func_1 = fem.Function(func_space)
            bc_func_1.x.array[:] = 1.0
            bc_func_1.x.scatter_forward()
            constraint_1 = fem.dirichletbc(
                bc_func_1,
                fem.locate_dofs_geometrical(func_space, marker_1)
            )
            bcs = [constraint_0, constraint_1]
            LinearProblem(a, 
                          L, 
                          bcs, 
                          solution, 
                          petsc_options={"ksp_type":"preonly", "pc_type":"lu", "pc_factor_mat_solver_type":"mumps"})
