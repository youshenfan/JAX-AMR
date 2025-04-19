import jax
import jax.numpy as jnp

Lx = 1.0
Ly = 1.0

Nx = 200
Ny = 200

n_block = [
    [1, 1],  # Level 0
    [20, 20], # Level 1
    [2, 2],  # Level 2
    [2, 2],  # Level 3
    [2, 2]   # Level 4
] # x-direction, y-direction

template_node_num = 1

buffer_num = 2

refinement_tolerance = {
    'density': 5.0,
    'velocity': 0.5
}


'''AUTO Computation'''
n_grid = [[Nx // n_block[0][0], Ny // n_block[0][1]]] 
dx = [Lx/Nx] 
dy = [Ly/Ny]
for i, (bx, by) in enumerate(n_block[1:], 1):
    px, py = n_grid[-1]
    mult = 1 if i == 1 else 2
    if (px * mult) % bx != 0 or (py * mult) % by != 0:
        raise ValueError(f"Initial grid not divisible: {(px * mult)}%{bx}={(py * mult)%bx}, {(py * mult)}%{by}={(py * mult)%by}")
        break

    n_grid.append([(px * mult // bx) , (py * mult// by) ])
    dx.append(Lx/Nx / (2.0**i))
    dy.append(Ly/Ny / (2.0**i))

grid_mask_buffer_kernel = ( 
    jnp.zeros((2 * buffer_num + 1, 2 * buffer_num + 1))
        .at[buffer_num, :].set(1)
        .at[:, buffer_num].set(1)
        .at[buffer_num, buffer_num].set(0)
)
