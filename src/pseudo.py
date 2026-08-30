import basix
import basix.ufl
from basix import CellType, PolynomialType, ElementFamily, LagrangeVariant, LatticeType, MapType, PolysetType, SobolevSpace
from petsc4py import PETSc
import numpy as np

def create_pseudo_element(interp_num, dim):
    pts, wts = basix.make_quadrature(CellType.hexahedron, (interp_num-1)*2)
    poly = basix.tabulate_polynomials(PolynomialType.legendre, CellType.hexahedron, interp_num-1, pts)

    lagrange = basix.create_element(ElementFamily.P, CellType.hexahedron, interp_num-1, LagrangeVariant.equispaced)
    f = lagrange.tabulate(0, pts)[0, :, :interp_num**3-(interp_num-2)**3, 0]

    wcoeffs = f.T@((poly*wts).T)

    geometry = basix.geometry(basix.CellType.hexahedron)
    topology = basix.topology(basix.CellType.hexahedron)
    x = [[], [], [], []]
    M = [[], [], [], []]

    for v in topology[0]:
        x[0].append(np.array(geometry[v]))
        M[0].append(np.array([[[[1.0]]]]))

    if interp_num > 2:
        for e in topology[1]:
            v0 = geometry[e[0]]
            v1 = geometry[e[1]]
            pts = basix.create_lattice(CellType.interval, interp_num-1, LatticeType.equispaced, False)
            edge_pts = np.array([v0 + p * (v1-v0) for p in pts])
            x[1].append(edge_pts)
            mat = np.eye(pts.shape[0], pts.shape[0]).reshape(pts.shape[0], 1, pts.shape[0], 1)
            M[1].append(mat)

        for f in topology[2]:
            v0 = geometry[f[0]]
            v1 = geometry[f[1]]
            v2 = geometry[f[2]]
            pts = basix.create_lattice(CellType.quadrilateral, interp_num-1, LatticeType.equispaced, False)
            face_pts = np.array([v0+p1*(v1-v0)+p2*(v2-v0) for p1,p2 in pts])
            x[2].append(face_pts)
            mat = np.eye(pts.shape[0], pts.shape[0]).reshape(pts.shape[0], 1, pts.shape[0], 1)
            M[2].append(mat)
    else:
        for e in topology[1]:
            x[1].append(np.zeros((0, 3)))
            M[1].append(np.zeros((0, 1, 0, 1)))
        for f in topology[2]:
            x[2].append(np.zeros((0, 3)))
            M[2].append(np.zeros((0, 1, 0, 1)))

    x[3].append(np.zeros((0, 3)))
    M[3].append(np.zeros((0, 1, 0, 1)))

    if dim > 1:
        pseudo_element = basix.ufl.blocked_element(basix.ufl.custom_element(
            CellType.hexahedron,
            [],
            wcoeffs,
            x,
            M,
            0,
            MapType.identity,
            SobolevSpace.H1,
            False,
            1,
            interp_num-1,
            PolysetType.standard,
        ), (dim,))
    else:
        pseudo_element = basix.ufl.custom_element(
            CellType.hexahedron,
            [],
            wcoeffs,
            x,
            M,
            0,
            MapType.identity,
            SobolevSpace.H1,
            False,
            1,
            interp_num-1,
            PolysetType.standard,
        )

    return pseudo_element

def assemble_matrix_(A, element_stiffness, dofmap):
    for idx, pos in enumerate(dofmap):
        A.setValuesBlockedLocal(pos, pos, element_stiffness[idx], PETSc.InsertMode.ADD_VALUES) 

def assemble_vector_(b, element_loads, dofmap):
    for idx, pos in enumerate(dofmap):
        b.setValuesBlockedLocal(pos, element_loads[idx], PETSc.InsertMode.ADD_VALUES)
    b.assemble()



if __name__ == "__main__":
    from dolfinx import fem, mesh, default_scalar_type
    from basix import create_element
    from mpi4py import MPI
    '''
    Nx = 10; Ny = 10
    comm = MPI.COMM_WORLD
    pseudo_element = create_pseudo_element(6, 1)
    domain = mesh.create_box(comm, [(0.0, 0.0, 0.0), (Nx, Ny, 1)], 
                             [Nx, Ny, 1], cell_type=mesh.CellType.hexahedron, dtype=default_scalar_type)
    
    V = fem.functionspace(domain, pseudo_element)
    func = fem.Function(V)

    local_cells = domain.topology.index_map(3).size_local
    coords = V.tabulate_dof_coordinates()
    dofmap = V.dofmap.list
    node_coords_ = coords[dofmap[0]]
    center = np.sum(node_coords_, axis=0) / node_coords_.shape[0]
    node_coords_ = node_coords_ - center
    for cell in range(local_cells):
        node_coords = coords[dofmap[cell]]
        center = np.sum(node_coords, axis=0) / node_coords.shape[0]
        node_coords = node_coords - center
        if not np.isclose(node_coords_, node_coords).all():
            print(cell)
    '''
    Nx = 10; Ny = 10
    comm = MPI.COMM_WORLD
    pseudo_element = create_pseudo_element(6, 1)
    domain = mesh.create_box(comm, [(0.0, 0.0, 0.0), (Nx, Ny, 1)], 
                             [Nx, Ny, 1], cell_type=mesh.CellType.hexahedron, dtype=default_scalar_type)
    
    V = fem.functionspace(domain, pseudo_element)
    func = fem.Function(V)
    local_cells = domain.topology.index_map(3).size_local
    coords = V.tabulate_dof_coordinates()
    dofmap = V.dofmap.list
    node_coords = coords[dofmap[0]]
    center = np.sum(node_coords, 0)/node_coords.shape[0]
    node_coords = node_coords - center + 1/2

    lagrange = create_element(ElementFamily.P, CellType.hexahedron, 5, LagrangeVariant.equispaced)
    values = lagrange.tabulate(0, node_coords)[0, :, :, 0]
    diag = np.diag(values)
    print(diag)
    print(diag.shape)
        