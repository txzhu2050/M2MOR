import matplotlib.pyplot as plt
import h5py
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

def plot_map(value_info: dict[str, int], reso: int, size_x: int, size_y: int, rst_path: Path, mode='mor'):
    dir_name = rst_path.parent
    with h5py.File(rst_path, 'r') as f:
        for value_name in value_info:
            value_idx = value_info[value_name]

            value: NDArray = f[value_name][:]
            if value_name == 'disp':
                value = value * 1e6
            if value_name == 'stress':
                value = value / 1e6

            if mode == 'mor':
                value = value.reshape((size_y, size_x, reso**2, -1)).transpose((3, 0, 1, 2))[value_idx, :, :, :]
                value = value.reshape((size_y, size_x, reso, reso))
                data = []
                for j in range(size_y):
                    rows = []
                    for i in range(size_x):
                        rows.append(value[j, i, :, :])
                    data.append(np.hstack(rows))
                value_map = np.vstack(data)
            elif mode == 'full':
                value = value.transpose(1, 0)[value_idx, :]
                value_map = value.reshape((size_y*reso, size_x*reso))

            np.savetxt(dir_name/(value_name+str(value_idx)+'.txt'), value_map, fmt='%.6f', delimiter=',')
            print(value_name, value_idx, np.min(value_map), np.max(value_map))
            
            plt.clf()
            plt.imshow(value_map, origin='lower')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(dir_name/(value_name+str(value_idx)+'.png'))


if __name__ == "__main__":
    value_info = {'temp':0, 'h_flux': 0, 'disp': 0, 'stress': 0}
    #value_info = {'temp': 0, 'disp': 0}
    reso = 50
    size_x = 10
    size_y = 10
    rst_path = Path(f'./mor/values0.h5')
    if rst_path.exists():
        print("mor:")
        plot_map(value_info, reso, size_x, size_y, rst_path, mode='mor')

    '''
    rst_path = Path(f'./full_{size_x}_{size_y}/values.h5')
    if rst_path.exists():
        print("Ground truth:")
        plot_map(value_info, reso, size_x, size_y, rst_path, mode='full')
    '''

