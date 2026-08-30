from dolfinx import fem, la, default_scalar_type
import ufl
from petsc4py import PETSc
import numpy as np
from numpy.typing import NDArray
from mpi4py import MPI
from typing import Tuple, Optional, Any

def unfold_idx_by_dim(idx: NDArray[np.int32], dim: int):
    """
    Unfold the original indexes using the specified dim.

    Agrs:
        idx: Original indexes.
        dim: Dim of the target indexes.
    """
    assert idx.ndim == 1

    target_idx = np.zeros(idx.shape[0]*dim, dtype=np.int32)
    cnt = np.arange(idx.shape[0])
    for d in range(dim):
        target_idx[dim*cnt + d] = dim*idx + d

    return target_idx


def is_overlapping(object_1: Tuple[float, ...], object_2: Tuple[float, ...], precision: int) -> bool:
    """
    Judge whether two rectangles/boxes overlap.
    """
    object_1 = tuple([round(coord, precision) for coord in object_1])
    object_2 = tuple([round(coord, precision) for coord in object_2])
    if len(object_1) == 4 and len(object_2) == 4:
        x_min_1, y_min_1, x_max_1, y_max_1 = object_1
        x_min_2, y_min_2, x_max_2, y_max_2 = object_2
        if max(x_min_1, x_min_2) < min(x_max_1, x_max_2) and max(y_min_1, y_min_2) < min(y_max_1, y_max_2):
            return True
        else:
            return False
    elif len(object_1) == 6 and len(object_2) == 6:
        x_min_1, y_min_1, z_min_1, x_max_1, y_max_1, z_max_1 = object_1
        x_min_2, y_min_2, z_min_2, x_max_2, y_max_2, z_max_2 = object_2
        if max(x_min_1, x_min_2) < min(x_max_1, x_max_2) and max(y_min_1, y_min_2) < min(y_max_1, y_max_2) and max(z_min_1, z_min_2) < min(z_max_1, z_max_2):
            return True
        else:
            return False
    else:
        return False

    
def is_containing(object_1: Tuple[float, ...], object_2: Tuple[float, ...], precision: int) -> bool:
    """
    Judge whether rectangle/box 1 contains rectangle/box 2.
    """
    object_1 = tuple([round(coord, precision) for coord in object_1])
    object_2 = tuple([round(coord, precision) for coord in object_2])
    if len(object_1) == 4 and len(object_2) == 4:
        x_min_1, y_min_1, x_max_1, y_max_1 = object_1
        x_min_2, y_min_2, x_max_2, y_max_2 = object_2
        if x_min_1 <= x_min_2 and x_max_1 >= x_max_2 and y_min_1 <= y_min_2 and y_max_1 >= y_max_2:
            return True
        else:
            return False
    elif len(object_1) == 6 and len(object_2) == 6:
        x_min_1, y_min_1, z_min_1, x_max_1, y_max_1, z_max_1 = object_1
        x_min_2, y_min_2, z_min_2, x_max_2, y_max_2, z_max_2 = object_2
        if x_min_1 <= x_min_2 and x_max_1 >= x_max_2 and y_min_1 <= y_min_2 and y_max_1 >= y_max_2 and z_min_1 <= z_min_2 and z_max_1 >= z_max_2:
            return True
        else:
            return False
    else:
        return False
    
def is_coinciding(object_1: Tuple[float, ...], object_2: Tuple[float, ...], precision: int) -> bool:
    object_1 = tuple([round(coord, precision) for coord in object_1])
    object_2 = tuple([round(coord, precision) for coord in object_2])
    if len(object_1) == 4 and len(object_2) == 4:
        x_min_1, y_min_1, x_max_1, y_max_1 = object_1
        x_min_2, y_min_2, x_max_2, y_max_2 = object_2
        if x_min_1 == x_min_2 and x_max_1 == x_max_2 and y_min_1 == y_min_2 and y_max_1 == y_max_2:
            return True
        else:
            return False
    elif len(object_1) == 6 and len(object_2) == 6:
        x_min_1, y_min_1, z_min_1, x_max_1, y_max_1, z_max_1 = object_1
        x_min_2, y_min_2, z_min_2, x_max_2, y_max_2, z_max_2 = object_2
        if x_min_1 == x_min_2 and x_max_1 == x_max_2 and y_min_1 == y_min_2 and y_max_1 == y_max_2 and z_min_1 == z_min_2 and z_max_1 == z_max_2:
            return True
        else:
            return False
    else:
        return False
    
def epsilon(disp: Any) -> Any:
    """
    Calc strain vector in Voigt notation:
    (xx, yy, zz, yz, xz, xy)

    Args:
        disp: Displacement.

    Returns:
        A strain vector.
    """
    u, v, w = disp[0], disp[1], disp[2]
    return ufl.as_vector(
            [u.dx(0), v.dx(1), w.dx(2), v.dx(2)+w.dx(1), u.dx(2)+w.dx(0), u.dx(1)+v.dx(0)]
        )

def build_nullspace(func_space: fem.FunctionSpace) -> Any:
        """
        Build PETSc nullspace.

        Args:
            func_space: Functionspace to build nullspace on.

        Returns:
            PETSc nullspace.
        """
        _x = fem.Function(func_space)

        length = func_space.dofmap.index_map.size_local
        bs = func_space.dofmap.index_map_bs

        if bs == 1:
            nullspace_basis = [la.vector(func_space.dofmap.index_map, bs, dtype=default_scalar_type)]
            basis = [b.array for b in nullspace_basis]

            dofs = func_space.dofmap.list.flatten()

            basis[0][dofs] = 1.0
            
            for b in nullspace_basis:
                b.scatter_forward()

            la.orthonormalize(nullspace_basis)
            assert la.is_orthonormal(nullspace_basis, float(np.finfo(_x.x.array.dtype).eps))

            basis_petsc = [PETSc.Vec().createWithArray(x[:bs*length], bsize=bs, comm=func_space.mesh.comm) for x in basis]

            return PETSc.NullSpace().create(comm=func_space.mesh.comm, vectors=basis_petsc)
        elif bs == 3:
            nullspace_basis = [la.vector(func_space.dofmap.index_map, bs, dtype=default_scalar_type) for _ in range(6)]
            basis = [b.array for b in nullspace_basis]

            dofs = [func_space.sub(i).dofmap.list.flatten() for i in range(3)]

            #set the three translational rigid body modes
            for i in range(3):
                basis[i][dofs[i]] = 1.0
            
            # set the three rotational rigid body modes
            x = func_space.tabulate_dof_coordinates()
            dofs_block = func_space.dofmap.list.flatten()
            x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
            basis[3][dofs[0]] = -x1
            basis[3][dofs[1]] = x0
            basis[4][dofs[0]] = x2
            basis[4][dofs[2]] = -x0
            basis[5][dofs[2]] = x1
            basis[5][dofs[1]] = -x2
            
            for b in nullspace_basis:
                b.scatter_forward()

            la.orthonormalize(nullspace_basis)
            assert la.is_orthonormal(nullspace_basis, float(np.finfo(_x.x.array.dtype).eps))

            basis_petsc = [PETSc.Vec().createWithArray(x[:bs*length], bsize=bs, comm=func_space.mesh.comm) for x in basis]

            return PETSc.NullSpace().create(comm=func_space.mesh.comm, vectors=basis_petsc)
        

def locate_grid_points(x0: float, y0: float, z0: float, dx: float, dy: float, nx: int, ny: int):
    """
    Calc the coordinates of grid points parallel to xy plane.

    Args:
        x0: Start coord of x.
        y0: Start coord of y.
        z0: Coord of z.
        dx: Length.
        dy: Width.
        nx: Number of points on the x direction.
        ny: Numner of points on the y direction.

    Returns:
        Coords of points in the shape of (3, num).
    """
    interval_x = dx/nx; interval_y = dy/ny
    x = np.arange(0.5, 0.5+nx)*interval_x + x0; y = np.arange(0.5, 0.5+ny)*interval_y + y0
    xx, yy = np.meshgrid(x, y)
    points = np.vstack((
        xx.reshape((1, -1)), 
        yy.reshape((1, -1)),
        np.full((1, nx*ny), fill_value=z0)
    ))

    return points


def gather_distributed_values(local_value: NDArray, value_idx: NDArray, comm: MPI.Comm, rank :int = 0) -> Optional[NDArray]:
    """
    Gather distributed values on different ranks and sort them using the indexes.

    Args:
        local_value: Local values to gather.
        value_idx: Global indexes of the local values.
        Comm: MPI communicator.
        rank: Target rank.

    Returns:
        Gathered and sorted values on the target rank (None on the others).
    """
    local_size = value_idx.size
    local_value = np.ascontiguousarray(local_value)
    value_idx = np.ascontiguousarray(value_idx)
    if comm.rank == rank:
        gathered_size = [0 for _ in range(comm.size)]
        gathered_size[rank] = local_size
        for i in range(comm.size):
            if i != rank:
                gathered_size[i] = comm.recv(source=i)
    else:
        comm.send(local_size, dest=rank)

    if comm.rank == rank:
        gathered_idx = [value_idx]
        for i in range(comm.size):
            if i != rank:
                recv_idx = np.empty((gathered_size[i],), dtype=np.int32)
                comm.Recv(recv_idx, source=i)
                gathered_idx.append(recv_idx)
        gathered_idx = np.concatenate(gathered_idx)
    else:
        comm.Send(value_idx, dest=rank)
        
    if comm.rank == rank:
        gathered_value = [local_value]
        for i in range(comm.size):
            if i != rank:
                recv_value = np.empty((gathered_size[i], local_value.shape[1]), dtype=default_scalar_type)
                comm.Recv(recv_value, source=i)
                gathered_value.append(recv_value)
        gathered_value = np.concatenate(gathered_value)
    else:
        comm.Send(local_value, dest=rank)

    if comm.rank == 0:
        _, perm = np.unique(gathered_idx, return_index=True)
        gathered_value = gathered_value[perm, :]
        return gathered_value
    else:
        return None
