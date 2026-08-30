import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["BLIS_NUM_THREADS"] = "1"

from dolfinx import common

from Parser import read_files, DataCollection
from Offline import MeshManager, ThermalRomBuilder, MechanicalRomBuilder

import argparse
from pathlib import Path
from mpi4py import MPI
import sys
sys.stdout.reconfigure(line_buffering=True)


def kernel(config_file: Path, comm: MPI.Comm) -> None:
    with common.Timer("1. Parsing"):
        system_xml_dict, config, matlib = read_files(config_file)
        simdb = DataCollection(config, matlib)

        model_dir = simdb.model_dir
        if comm.rank == 0:
            if not model_dir.exists():
                model_dir.mkdir(parents=True)
        comm.barrier()

        rom_path_dict: dict[str, Path] = {}
        mesh_path_dict: dict[str, Path] = {}
        mesh_size_dict: dict[str, float] = {}
        for structure_name in system_xml_dict:
            rom_path = (model_dir/structure_name).with_suffix(".h5")
            mesh_path = (model_dir/structure_name).with_suffix(".msh")
            if rom_path.exists():
                rom_path.unlink()
            if mesh_path.exists():
                mesh_path.unlink()
            rom_path_dict[structure_name] = rom_path
            mesh_path_dict[structure_name] = mesh_path

            mesh_size_dict[structure_name] = simdb.mesh_size
            system_xml = system_xml_dict[structure_name]
            mesh_size = system_xml.find_elements('MeshSize')
            if mesh_size:
                if 'Size' in mesh_size[0].attrib:
                    mesh_size_dict[structure_name] = float(mesh_size[0].attrib['Size'])
    comm.barrier()
    
    for structure_name in system_xml_dict:
        print(f"Start generating rom for {structure_name}:\n")
        system_xml = system_xml_dict[structure_name]
        rom_path = rom_path_dict[structure_name]
        mesh_path = mesh_path_dict[structure_name]
        mesh_size = mesh_size_dict[structure_name]

        shapes = system_xml.find_elements('Shape')
        mesh_manager = MeshManager(shapes, simdb, comm=comm)

        with common.Timer(f"2. Meshing ({structure_name})"):
            mesh_manager.generate_mesh(mesh_path, mesh_size)
        
        with common.Timer(f"3. Rom generation ({structure_name})"):
            with common.Timer(f"3.1 Thermal rom ({structure_name})"):
                thermal_rom_builder = ThermalRomBuilder(mesh_manager, rom_path)
                a, L = thermal_rom_builder.problem()
                thermal_rom_builder.build_rom(a, L)

            with common.Timer(f"3.2 Mechanical rom ({structure_name})"):
                mechanical_rom_builder = MechanicalRomBuilder(mesh_manager, rom_path)
                a, L = mechanical_rom_builder.problem()
                mechanical_rom_builder.build_rom(a, L)
                print(f"Finish generating rom for {structure_name}.\n")
    
    common.list_timings(comm)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', action='store')

    parser_args = parser.parse_args()
    config_file = Path(parser_args.config)

    comm = MPI.COMM_WORLD

    kernel(
        config_file=config_file,
        comm=comm
    )


    