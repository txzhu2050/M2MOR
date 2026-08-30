import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

from dolfinx import common
from mpi4py import MPI

from Parser import read_files, DataCollection
from Offline import MeshManager
from Mapper import ThermalMapper, MechanicalMapper
from Online import FlpManager, ThermalOnlineSolver, MechanicalOnlineSolver
from Postprocessor import CouplePost, MechanicalPost

import argparse
from pathlib import Path
import psutil
import sys
sys.stdout.reconfigure(line_buffering=True)

def kernel(config_file: Path, comm: MPI.Comm) -> None:
    with common.Timer("1. Parsing"):
        system_xml_dict, config, matlib = read_files(config_file)
        simdb = DataCollection(config, matlib)
        mode = simdb.mode
        model_dir = simdb.model_dir
        result_dir = simdb.result_dir
        flp_path = simdb.flp_path

        if comm.rank == 0:
            if not model_dir.exists():
                print(f"ERROR: {model_dir} not found!")
                comm.Abort(1)
            if not result_dir.exists():
                result_dir.mkdir(parents=True)
        comm.barrier()

        flp_manager = FlpManager(simdb, model_dir, flp_path, comm)

        mesh_manager_dict: dict[str, MeshManager] = {}
        for structure_name in system_xml_dict:
            mesh_path = (model_dir/structure_name).with_suffix(".msh")
            system_xml = system_xml_dict[structure_name]
            shapes = system_xml.find_elements('Shape')
            mesh_manager = MeshManager(shapes, simdb, comm=MPI.COMM_SELF)
            mesh_manager.load_mesh(mesh_path)
            mesh_manager_dict[structure_name] = mesh_manager
        bb_box = list(mesh_manager_dict.values())[0].bb_box
        comm.barrier()
    
    thermal_bcs = None
    mechanical_bcs = None
    with common.Timer("2. Mapping"):
        if mode == 0 or mode == 2:
            if simdb.temp_path:
                thermal_mapper = ThermalMapper(flp_manager, bb_box)
                thermal_bcs = thermal_mapper.get_bcs(simdb.temp_path)
        if mode == 1 or mode == 2:
            if simdb.disp_path:
                mechanical_mapper = MechanicalMapper(flp_manager, bb_box)
                mechanical_bcs = mechanical_mapper.get_bcs(simdb.disp_path)

    t_func = None
    u_func = None
    with common.Timer("3. Thermal solving"):
        if mode == 0 or mode == 2:
            thermal_solver = ThermalOnlineSolver(flp_manager, thermal_bcs)
            t_func = thermal_solver.solve()

    with common.Timer("4. Mechanical solving"):
        if mode == 1 or mode == 2:
            mechanical_solver = MechanicalOnlineSolver(flp_manager, mechanical_bcs, t_func)
            u_func = mechanical_solver.solve()
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    total_rss = comm.reduce(memory_info.rss, root=0)
    total_vms = comm.reduce(memory_info.vms, root=0)
    if comm.rank == 0:
        print("Memory usage (solving):")
        print(f"RSS: {total_rss / 1024 / 1024:.3f} MB")
        print(f"VMS: {total_vms / 1024 / 1024:.3f} MB")
        print()
    comm.barrier()
            
    with common.Timer("5. Postprocessing"):
        z_coord = simdb.z_coord * 10**simdb.unit
        if mode == 0 or mode == 2:
            couple_post = CouplePost(mesh_manager_dict, flp_manager, t_func, u_func)
            value_dict = couple_post.set_up_values(simdb.value_names)
            couple_post.extract_plane_values(
                z_coord,
                simdb.reso,
                value_dict,
                result_dir
            )
        elif mode == 1:
            mechanical_post = MechanicalPost(mesh_manager_dict, flp_manager, u_func)
            value_dict = mechanical_post.set_up_values(simdb.value_names)
            mechanical_post.extract_plane_values(
                z_coord,
                simdb.reso,
                value_dict,
                result_dir
            )
        comm.barrier()

    memory_info = process.memory_info()
    total_rss = comm.reduce(memory_info.rss, root=0)
    total_vms = comm.reduce(memory_info.vms, root=0)
    if comm.rank == 0:
        print("Memory usage (postprocessing):")
        print(f"RSS: {total_rss / 1024 / 1024:.3f} MB")
        print(f"VMS: {total_vms / 1024 / 1024:.3f} MB")
        print()
    comm.barrier()

    common.list_timings(comm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', action='store')

    parser_args = parser.parse_args()
    config_file = Path(parser_args.config)

    comm = MPI.COMM_WORLD
    
    kernel(
        config_file = config_file,
        comm = comm
    )