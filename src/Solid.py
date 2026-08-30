import numpy as np
from numpy.typing import NDArray
from configparser import ConfigParser
import sys

def define_material_properties(
    label: str,
    config: ConfigParser
) -> dict[str, NDArray[np.float64]]:
    """Define thermal properties (linear or non-linear) for a material.
    
    Args:
        label: Name/identifier of the material.
        config: Configuration parser object containing material properties.
        
    Returns:
        stiffness matrix, thermal expansion coefficient vector, and thermal conductivity matrix of the material.
    """
    # Get material parameters from config section
    try:
        params = config._sections[label]
    except KeyError:
        print(f"ERROR: Material '{label}' not found in configuration!")
        sys.exit(1)
    
    stiffness = np.zeros((6, 6)) #xx, yy, zz, yz, xz, xy
    cte = np.zeros((6,))
    ctc = np.zeros((3, 3))

    # Parse young's modulus values
    try:
        E = _parse_values(params["young modulus (n/m2)"])
    except (KeyError, ValueError):
        print(f"ERROR: Invalid young's modulus for material '{label}'!")
        sys.exit(1)
    
    if len(E) == 1:
        try:
            v = _parse_values(params["poisson ratio (1)"])
            if len(v) != 1:
                raise ValueError
        except (KeyError, ValueError):
            print(f"ERROR: Invalid poisson's ratio for material '{label}'!")
            sys.exit(1)
        E = E[0]; v = v[0]
        tmp = E / (1 - 2*v) / (1 + v)
        stiffness[0][0] = tmp * (1 - v); stiffness[0][1] = tmp * v; stiffness[0][2] = tmp * v
        stiffness[1][0] = stiffness[0][1]; stiffness[1][1] = tmp * (1 - v); stiffness[1][2] = tmp * v
        stiffness[2][0] = stiffness[0][2]; stiffness[2][1] = stiffness[1][2]; stiffness[2][2] = tmp * (1 - v)
        stiffness[3][3] = tmp * (1 - 2*v) / 2
        stiffness[4][4] = tmp * (1 - 2*v) / 2
        stiffness[5][5] = tmp * (1 - 2*v) / 2

    elif len(E) == 3:
        try:
            G = _parse_values(params["shear modulus (n/m2)"])
            if len(G) != 3:
                raise ValueError
        except (KeyError, ValueError):
            print(f"ERROR: Invalid shear modulus for material '{label}'!")
            sys.exit(1)
        try:
            v = _parse_values(params["poisson ratio (1)"])
            if len(v) != 3:
                raise ValueError
        except (KeyError, ValueError):
            print(f"ERROR: Invalid poisson ratio for material '{label}'!")
            sys.exit(1)
        E1, E2, E3 = E; G12, G13, G23 = G; v12, v13, v23 = v
        v21 = v12 / E1 * E2; v31 = v13 / E1 * E3; v32 = v23 / E2 * E3
        tmp = np.linalg.det(np.array([[1/E1, -v21/E2, -v31/E3], 
                                      [-v12/E1, 1/E2, -v32/E3], 
                                      [-v13/E1, -v23/E2, 1/E3]]))
        stiffness[0][0] = (1 - v23*v32) / (E2*E3*tmp); stiffness[0][1] = (v21 + v23*v31) / (E2*E3*tmp); stiffness[0][2] = (v31 + v21*v32) / (E2*E3*tmp)
        stiffness[1][0] = stiffness[0][1]; stiffness[1][1] = (1 - v13*v31) / (E1*E3*tmp); stiffness[1][2] = (v32 + v12*v31) / (E1*E3*tmp)
        stiffness[2][0] = stiffness[0][2]; stiffness[2][1] = stiffness[1][2]; stiffness[2][2] = (1 - v12*v21) / (E1*E2*tmp)
        stiffness[3][3] = G23
        stiffness[4][4] = G13
        stiffness[5][5] = G12
    
    # Parse thermal expansion coefficient values
    try:
        alpha = _parse_values(params["thermal expansion coefficient (1/k)"])
    except (KeyError, ValueError):
        print(f"ERROR: Invalid thermal expansion coefficient for material '{label}'!")
        sys.exit(1)
    
    if len(alpha) == 1:
        cte[0] = alpha[0]; cte[1] = alpha[0]; cte[2] = alpha[0]

    elif len(alpha) == 3:
        cte[0] = alpha[0]; cte[1] = alpha[1]; cte[2] = alpha[2]


    # Parse thermal conductivity values
    try: 
        k = _parse_values(params["thermal conductivity (w/m-k)"])
    except (KeyError, ValueError):
        print(f"ERROR: Invalid thermal conductivity for material '{label}'!")
        sys.exit(1)

    if len(k) == 1:
        ctc[0][0] = k[0]; ctc[1][1] = k[0]; ctc[2][2] = k[0]

    elif len(k) == 3:
        ctc[0][0] = k[0]; ctc[1][1] = k[1]; ctc[2][2] = k[2]


    return {
            "stiffness": stiffness,
            "cte": cte,
            "ctc": ctc
    }




def _parse_values(
    material_str: str
) -> list[float]:
    """Parse values from string.
    
    Args:
        material_str: Space-separated values.
        
    Returns:
        list of values.
    """
    values = material_str.split()
    
    if len(values) == 1:
        # Isotropic material (same in all directions)
        value = float(values[0])
        return [value]
    elif len(values) == 3:
        # Orthotropic material (different x, y, z)
        return [float(values[0]), float(values[1]), float(values[2])]
    else:
        raise ValueError