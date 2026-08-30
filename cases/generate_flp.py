from pathlib import Path
import numpy as np
import random
import json

np.random.seed(42)
random.seed(42)

structure = 'tsv'

size_x = 10
size_y = 10
flp_path = Path(f'./flp.json')
structure_ratio = 0.8
dummy_boundary = 0
power1 = np.random.uniform(2e14, 4e14, size=(size_x, size_y))
#power2 = ...
flp_data = {"size":(size_x+2*dummy_boundary, size_y+2*dummy_boundary)}

idx = 0
for i in range(size_x + 2*dummy_boundary):
    for j in range(size_y + 2*dummy_boundary):
        flp_data[str(idx)] = {"loc": (i,j), "structure": "dummy", "power":[]}
        if i >= dummy_boundary and i <= size_x - (1-dummy_boundary) and j >= dummy_boundary and j <= size_y - (1-dummy_boundary):
            flag = random.random()
            if flag < structure_ratio:
                #flp_data[str(idx)] = {"loc": (i,j), "structure": structure, "power": [power1[i-dummy_boundary][j-dummy_boundary], power2[i-dummy_boundary][j-dummy_boundary]]}
                flp_data[str(idx)] = {"loc": (i,j), "structure": structure, "power": [power1[i-dummy_boundary][j-dummy_boundary]]}
        idx += 1


with open(flp_path, 'w') as f:
    json.dump(flp_data, f, indent=2)

data = np.zeros((size_y+2*dummy_boundary, size_x+2*dummy_boundary))
for key in flp_data.keys():
    if key != "size":
        loc_x, loc_y = flp_data[key]["loc"]
        if flp_data[key]["structure"] == structure:
            data[loc_y][loc_x] = flp_data[key]["power"][0]

import matplotlib.pyplot as plt
plt_path = flp_path.with_suffix('.png')
plt.imshow(data, origin='lower')
plt.colorbar()
plt.savefig(plt_path)