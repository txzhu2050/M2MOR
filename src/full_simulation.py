import os
os.environ["OMP_NUM_THREADS"] = "1"

from dolfinx import fem, mesh, geometry, io, common, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl
import gmsh
import h5py
import numpy as np
from numpy.typing import NDArray
import json
from mpi4py import MPI
from pathlib import Path
from typing import Optional
import argparse
import psutil
import sys
sys.stdout.reconfigure(line_buffering=True)

from Parser import read_files, DataCollection
from Offline import Shape, MeshManager
from utils import unfold_idx_by_dim, is_containing, epsilon, locate_grid_points


def assign_properties(mesh_manager_dict: dict[str, MeshManager], flp: dict, cell_tags: mesh.MeshTags, func: fem.Function) -> None:
    names = list(mesh_manager_dict.keys())
    structure_name = None
    for name in names:
        if name != 'dummy':
            structure_name = name
    mesh_manager_ = mesh_manager_dict[structure_name]
    simdb = mesh_manager_.simdb
    substrate_boxs: list[Shape] = []
    for shape in mesh_manager_.shape_list:
        if shape.type == 'box':
            substrate_boxs.append(shape)

    name = func.name
    assert name in ['stiffness', 'cte', 'ctc', 'power']
    func.x.array[:] = 0.0
    func.x.scatter_forward()

    size_x = int(flp['size'][0])
    size_y = int(flp['size'][1])
    loc_names, loc_powers = parse_flp(mesh_manager_dict, flp)
    
    if name in ['stiffness', 'cte', 'ctc']:
        size = func.function_space.value_size
        for substrate_box in substrate_boxs:
            cells = cell_tags.find(substrate_box.tag)
            if cells.size > 0:
                func.x.array[unfold_idx_by_dim(cells, size)] = np.concatenate(
                    [simdb.mat_dict[substrate_box.filling_mat][name].flatten() for _ in cells],
                    dtype=default_scalar_type)

    for j in range(size_y):
        for i in range(size_x):
            idx = j*size_x + i

            s_name = loc_names[j][i]
            power_values = loc_powers[j][i]
            mesh_manager = mesh_manager_dict[s_name]

            if name in ['stiffness', 'cte', 'ctc']:
                size = func.function_space.value_size
                for shape in mesh_manager.shape_list:
                    if shape.type == 'box':
                        continue
                    cells = cell_tags.find(shape.tag + idx*10)
                    if cells.size > 0:
                        func.x.array[unfold_idx_by_dim(cells, size)] = np.concatenate(
                            [simdb.mat_dict[shape.filling_mat][name].flatten() for _ in cells],
                            dtype=default_scalar_type)
            elif name == 'power':
                tmp = 0
                for shape in mesh_manager.shape_list:
                    if shape.type == 'box':
                        continue
                    if shape.power_dens is not None:
                        cells = cell_tags.find(shape.tag + idx*10)
                        if cells.size > 0:
                            func.x.array[cells] = np.full_like(cells, power_values[tmp], dtype=default_scalar_type)
                            tmp+=1

def parse_flp(mesh_manager_dict: dict[str, MeshManager], flp: dict) -> tuple[list[list[str]], list[list[NDArray]]]:
    mesh_manager_ = list(mesh_manager_dict.values())[0]
    simdb = mesh_manager_.simdb

    size_x = int(flp['size'][0])
    size_y = int(flp['size'][1])
    loc_names = [[simdb.default_name for _ in range(size_x)] for _ in range(size_y)]
    loc_powers = [[np.zeros(len(mesh_manager_dict[simdb.default_name].power_item)) for _ in range(size_x)] 
                        for _ in range(size_y)]
    for key in flp.keys():
        if key != 'size':
            loc_x, loc_y = flp[key]['loc']
            name = flp[key]['structure']
            power = np.array(flp[key]['power']) / (10**(simdb.unit*2))
            assert power.shape[0] == len(mesh_manager_dict[name].power_item)
            loc_names[loc_y][loc_x] = name
            loc_powers[loc_y][loc_x] = power
    
    return loc_names, loc_powers

def generate_mesh(mesh_manager_dict: dict[str, MeshManager], flp: dict, mesh_path: Path, comm: MPI.Comm) -> tuple[mesh.Mesh, mesh.MeshTags]:
    def _generate_mesh():
        gmsh.model.add('structure')
        gmsh.option.setNumber("General.Verbosity", 1)

        names = list(mesh_manager_dict.keys())
        structure_name = None
        for name in names:
            if name != 'dummy':
                structure_name = name
        mesh_manager_ = mesh_manager_dict[structure_name]
        simdb = mesh_manager_.simdb
        
        size_x = int(flp['size'][0])
        size_y = int(flp['size'][1])
        loc_names, _ = parse_flp(mesh_manager_dict, flp)

        #add substrate
        sub_length = 0.0
        sub_width = 0.0
        for shape in mesh_manager_.shape_list:
            if shape.type == 'box':
                gmsh.model.occ.addBox(shape.x_origin, shape.y_origin, shape.z_origin, shape.length*size_x, shape.width*size_y, shape.height)
                sub_length = shape.length
                sub_width = shape.width

        for j in range(size_y):
            for i in range(size_x):
                x_offset = i*sub_length
                y_offset = j*sub_width

                name = loc_names[j][i]
                mesh_manager = mesh_manager_dict[name]

                for shape in mesh_manager.shape_list:
                    if shape.type == 'box':
                        continue
                    elif shape.type == 'cylinder':
                        gmsh.model.occ.addCylinder(shape.x_origin + x_offset, shape.y_origin + y_offset, shape.z_origin, 0, 0, shape.height, shape.radius)
                    elif shape.type == 'sphere':
                        gmsh.model.occ.addSphere(shape.x_origin + x_offset, shape.y_origin + y_offset, shape.z_origin, shape.radius)

        gmsh.model.occ.fragment(gmsh.model.occ.getEntities(3), [])
        gmsh.model.occ.synchronize()

        dimtags = gmsh.model.occ.getEntities(3)
        tags = [tag for _, tag in dimtags]
        tag_box_dict: dict[int, tuple[float, float, float, float, float, float]] = {}
        for tag in tags:
            tag_box_dict[tag] = gmsh.model.getBoundingBox(3, tag)

        for j in range(size_y):
            for i in range(size_x):
                idx = j*size_x + i

                x_offset = i*sub_length
                y_offset = j*sub_width

                name = loc_names[j][i]
                mesh_manager = mesh_manager_dict[name]

                for shape in mesh_manager.shape_list[::-1]:
                    if shape.type == 'box':
                        continue
                    bb_box = (
                        shape.bb_box[0] + x_offset,
                        shape.bb_box[1] + y_offset,
                        shape.bb_box[2],
                        shape.bb_box[3] + x_offset,
                        shape.bb_box[4] + y_offset,
                        shape.bb_box[5]
                    )
                    shape_tags = []
                    for tag in list(tag_box_dict.keys()):
                        if is_containing(bb_box, tag_box_dict[tag], mesh_manager_._p-mesh_manager_._u):
                            shape_tags.append(tag)
                            tag_box_dict.pop(tag)
                    gmsh.model.addPhysicalGroup(3, shape_tags, tag=shape.tag + idx*10)
        
        for shape in mesh_manager_.shape_list[::-1]:
            if shape.type == 'box':
                shape_tags = []
                bb_box = (shape.x_origin, shape.y_origin, shape.z_origin,
                        shape.x_origin + shape.length*size_x, shape.y_origin + shape.width*size_y, shape.z_origin + shape.height)
                for tag in list(tag_box_dict.keys()):
                    if is_containing(bb_box, tag_box_dict[tag], mesh_manager_._p-mesh_manager_._u):
                        shape_tags.append(tag)
                        tag_box_dict.pop(tag)
                        break
                gmsh.model.addPhysicalGroup(3, shape_tags, tag=shape.tag)

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), round(simdb.mesh_size*10**simdb.unit, simdb.precision-simdb.unit))
        gmsh.model.mesh.generate(3)
        gmsh.write(str(mesh_path))
    
    if comm.rank == 0:
        gmsh.initialize()
        _generate_mesh()
        #gmsh.merge(str(mesh_path))
        msh, cell_tags, _ = io.gmshio.model_to_mesh(gmsh.model, comm, 0, 3, dtype=default_scalar_type)
        gmsh.finalize()
    else:
        msh, cell_tags, _ = io.gmshio.model_to_mesh(gmsh.model, comm, 0, 3, dtype=default_scalar_type)
    
    return msh, cell_tags

def thermal_problem(func_space: fem.FunctionSpace, 
                    ctc: fem.Function, 
                    power: fem.Function) -> tuple[ufl.Form, ufl.Form]: 
    t = ufl.TrialFunction(func_space); delta_t = ufl.TestFunction(func_space)
    weak_form = (ufl.inner(ufl.dot(ctc, ufl.grad(t)), ufl.grad(delta_t)) - power * delta_t) * ufl.dx
    
    return ufl.system(weak_form)

def mechanical_problem(func_space: fem.FunctionSpace,
                       stiffness: fem.Function,
                       cte: fem.Function,
                       t_func: fem.Function,
                       ref_temp: fem.Constant) -> tuple[ufl.Form, ufl.Form]:
    disp = ufl.TrialFunction(func_space); delta_disp = ufl.TestFunction(func_space)
    strain = epsilon(disp); delta_strain = epsilon(delta_disp)
    stress = ufl.dot(stiffness, (strain - cte * (t_func-ref_temp)))
    weak_form = ufl.inner(stress, delta_strain) * ufl.dx

    return ufl.system(weak_form)

def extract_plane_values(msh: dict[str, MeshManager],
                         pts: NDArray,
                         value_dict: dict[str, tuple[fem.Function, Optional[fem.Expression]]], 
                         dir_name: Path, 
                         comm: MPI.Comm) -> None:
    bb_tree = geometry.bb_tree(msh, 3)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts.T)
    local_cells = []
    local_pts = []
    local_idx = []
    for idx, pt in enumerate(pts.T):
        if len(colliding_cells.links(idx)) > 0:
            local_idx.append(idx)
            local_cells.append(colliding_cells.links(idx)[0])
            local_pts.append(pt)

    rst_path = dir_name / "values.h5"
    with h5py.File(rst_path, 'a', driver="mpio", comm=comm) as f:
        value_handle_dict = {}
        for value_name in value_dict.keys():
            size = value_dict[value_name][0].function_space.value_size
            tmp = f.create_dataset(value_name,
                                    shape=(len(pts.T), size),
                                    dtype=default_scalar_type)
            value_handle_dict[value_name] = tmp

        for value_name in value_dict.keys():
            value_func, value_expr = value_dict[value_name]
            if value_expr is not None:
                value_func.interpolate(value_expr)
            values = value_func.eval(local_pts, local_cells)
            if value_name == "disp":
                value_handle_dict[value_name][local_idx, :] = values / (10**simdb.unit)
            else:
                value_handle_dict[value_name][local_idx, :] = values



if __name__ == "__main__":
    with common.Timer('1. Parsing'):
        parser = argparse.ArgumentParser()
        parser.add_argument('-config', action='store')
        parser.add_argument('-flp', action='store')
        parser.add_argument('--result_dir', action='store', default='.')

        parser_args = parser.parse_args()
        config_file = Path(parser_args.config)
        flp_file = Path(parser_args.flp)
        result_dir = Path(parser_args.result_dir)

        comm = MPI.COMM_WORLD

        if comm.rank == 0:
            if not result_dir.exists():
                result_dir.mkdir(parents=True)
        comm.barrier()

        with open(flp_file, 'r') as f:
            flp: dict = json.load(f)
        size_x = int(flp['size'][0])
        size_y = int(flp['size'][1])

        system_xml_dict, config, matlib = read_files(config_file)
        simdb = DataCollection(config, matlib)

        mesh_manager_dict: dict[str, MeshManager] = {}
        for structure_name in system_xml_dict:
            system_xml = system_xml_dict[structure_name]
            shapes = system_xml.find_elements('Shape')
            mesh_manager = MeshManager(shapes, simdb, comm=MPI.COMM_SELF)
            mesh_manager_dict[structure_name] = mesh_manager
        
        names = list(mesh_manager_dict.keys())
        structure_name = None
        for name in names:
            if name != 'dummy':
                structure_name = name
        mesh_manager_ = mesh_manager_dict[structure_name]
        simdb = mesh_manager_.simdb
        bb_box = [np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf]
        for shape in mesh_manager_.shape_list:
            if shape.type == 'box':
                bb_box[0] = min(bb_box[0], shape.bb_box[0])
                bb_box[1] = min(bb_box[1], shape.bb_box[1])
                bb_box[2] = min(bb_box[2], shape.bb_box[2])
                bb_box[3] = max(bb_box[3], shape.bb_box[0] + shape.length * size_x)
                bb_box[4] = max(bb_box[4], shape.bb_box[1] + shape.width * size_y)
                bb_box[5] = max(bb_box[5], shape.bb_box[2] + shape.height)

    with common.Timer('2. Meshing'):
        mesh_path = result_dir / 'array.msh'
        msh, cell_tags = generate_mesh(mesh_manager_dict, flp, mesh_path, comm)

    with common.Timer('3. Solving'):
        opts = {}
        if simdb.solver_type == 0:
            opts["ksp_type"] = "cg"
            opts["ksp_norm_type"] = 'unpreconditioned'
            opts["pc_type"] = "hypre"
            opts["ksp_rtol"] = simdb.rtol
            opts["ksp_atol"] = simdb.atol
            opts["ksp_max_it"] = simdb.max_it
        elif simdb.solver_type == 1:
            opts["ksp_type"] = "preonly"
            opts["pc_type"] = "lu"
            opts["pc_factor_mat_solver_type"] = "mumps"
        if simdb.is_display:
            opts["ksp_monitor"] = None
            opts["ksp_converged_reason"] = None

        top_marker = lambda x: (
                    np.isclose(x[2], bb_box[5])
                )
        bot_marker = lambda x: (
            np.isclose(x[2], bb_box[0])
        )

        with common.Timer("3.1 Thermal"):
            ctc = fem.Function(fem.functionspace(msh, ("DG", 0, (3, 3))), name="ctc")
            assign_properties(mesh_manager_dict, flp, cell_tags, ctc)
            power = fem.Function(fem.functionspace(msh, ("DG", 0)), name="power")
            assign_properties(mesh_manager_dict, flp, cell_tags, power)
            V = fem.functionspace(msh, ("Lagrange", 2))
            if comm.rank == 0:
                print(f"# of DoFs: {V.dofmap.index_map.size_global*V.dofmap.index_map_bs}")
            comm.barrier()
            a, L = thermal_problem(V, ctc, power)
            t_func = fem.Function(V)

            top_bc_func = fem.Function(V)
            top_bc_func.x.array[:] = simdb.amb_temp
            top_bc_func.x.scatter_forward()
            top_constraint = fem.dirichletbc(
                    top_bc_func,
                    fem.locate_dofs_geometrical(V, top_marker)
                    )
            bot_bc_func = fem.Function(V)
            bot_bc_func.x.array[:] = simdb.amb_temp
            bot_bc_func.x.scatter_forward()
            bot_constraint = fem.dirichletbc(
                    bot_bc_func,
                    fem.locate_dofs_geometrical(V, bot_marker)
                    )     
            bcs = [top_constraint, bot_constraint]

            problem = LinearProblem(a, L, bcs, t_func, opts)

            problem.solve()

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        total_rss = comm.reduce(memory_info.rss, root=0)
        total_vms = comm.reduce(memory_info.vms, root=0)
        if comm.rank == 0:
            print("Solving stage:")
            print(f"RSS: {total_rss / 1024 / 1024:.3f} MB")
            print(f"VMS: {total_vms / 1024 / 1024:.3f} MB")
            print()
        comm.barrier()

        with common.Timer("3.2 Mechanical"):
            stiffness = fem.Function(fem.functionspace(msh, ("DG", 0, (6, 6))), name="stiffness")
            assign_properties(mesh_manager_dict, flp, cell_tags, stiffness)
            cte = fem.Function(fem.functionspace(msh, ("DG", 0, (6,))), name="cte")
            assign_properties(mesh_manager_dict, flp, cell_tags, cte)
            ref_temp = fem.Constant(msh, simdb.ref_temp)
            V = fem.functionspace(msh, ("Lagrange", 2, (3,)))
            if comm.rank == 0:
                print(f"# of DoFs: {V.dofmap.index_map.size_global*V.dofmap.index_map_bs}")
            comm.barrier()
            a, L = mechanical_problem(V, stiffness, cte, t_func, ref_temp)
            u_func = fem.Function(V)

            top_bc_func = fem.Function(V)
            top_constraint = fem.dirichletbc(
                    top_bc_func,
                    fem.locate_dofs_geometrical(V, top_marker)
                    )
            bot_bc_func = fem.Function(V)
            bot_constraint = fem.dirichletbc(
                    bot_bc_func,
                    fem.locate_dofs_geometrical(V, bot_marker)
                    )
            bcs = [top_constraint, bot_constraint]

            problem = LinearProblem(a, L, bcs, u_func, opts)
            problem.solve()

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        total_rss = comm.reduce(memory_info.rss, root=0)
        total_vms = comm.reduce(memory_info.vms, root=0)
        if comm.rank == 0:
            print("Solving stage:")
            print(f"RSS: {total_rss / 1024 / 1024:.3f} MB")
            print(f"VMS: {total_vms / 1024 / 1024:.3f} MB")
            print()
        comm.barrier()

    with common.Timer("4. Postprocessing"):
        local_reso = 50
        pts = locate_grid_points(
            bb_box[0],
            bb_box[1],
            (bb_box[2] + bb_box[5])/2,
            bb_box[3] - bb_box[0],
            bb_box[4] - bb_box[1],
            local_reso*size_x,
            local_reso*size_y
        )

        value_dict: dict[str, tuple[fem.Function, Optional[fem.Expression]]] = {}

        value_dict["temp"] = (t_func, None)

        heat_flux = -ctc*ufl.grad(t_func)
        V_flux = fem.functionspace(msh, ("DG", 1, (3,)))
        flux_expr = fem.Expression(heat_flux, V_flux.element.interpolation_points())
        heat_flux_func = fem.Function(V_flux)
        value_dict["h_flux"] = (heat_flux_func, flux_expr)
        strain = epsilon(u_func)
        stress = ufl.dot(stiffness, (strain - cte * (t_func-ref_temp)))

        value_dict["disp"] = (u_func, None)

        V_strain = fem.functionspace(msh, ("DG", 1, (6,)))
        strain_expr = fem.Expression(strain, V_strain.element.interpolation_points())
        strain_func = fem.Function(V_strain)
        value_dict["strain"] = (strain_func, strain_expr)

        V_stress = fem.functionspace(msh, ("DG", 1, (6,)))
        stress_expr = fem.Expression(stress, V_stress.element.interpolation_points())
        stress_func = fem.Function(V_stress)
        value_dict["stress"] = (stress_func, stress_expr)

        extract_plane_values(msh, pts, value_dict, result_dir, comm)

    memory_info = process.memory_info()
    total_rss = comm.reduce(memory_info.rss, root=0)
    total_vms = comm.reduce(memory_info.vms, root=0)
    if comm.rank == 0:
        print("Postprocessing stage:")
        print(f"RSS: {total_rss / 1024 / 1024:.3f} MB")
        print(f"VMS: {total_vms / 1024 / 1024:.3f} MB")
        print()
    comm.barrier()

    common.list_timings(comm, [common.TimingType.wall])






    


