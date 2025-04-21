# Copyright © 2025 Haocheng Wen, Faxuan Luo
# SPDX-License-Identifier: MIT

import jax.numpy as jnp
import jaxamr.amr as amr

Lx = amr.Lx
Ly = amr.Ly
n_block = amr.n_block
n_grid = amr.n_grid

def plot_block_data(blk_data_component, blk_info, fig_handle, vrange):

    valid_blk_num = blk_info['number']
    level = blk_info['glob_index'].shape[1]//2 - 1
    for i in range(valid_blk_num):
        x_min = 0
        y_min = 0
        dx = Lx
        dy = Ly
        for j in range(level+1):
            dx = dx/n_block[j][0]
            dy = dy/n_block[j][1]
            idx = blk_info['glob_index'][i, 2*j]
            idy = blk_info['glob_index'][i, 2*j+1]
            x_min = x_min + dx * idx
            y_min = y_min + dy * idy
        x_max = x_min + dx
        y_max = y_min + dy

        if level == 0:
            nx = n_grid[level][0]
            ny = n_grid[level][1]
        else:
            nx = n_grid[level][0] * 2
            ny = n_grid[level][1] * 2

        x_edges = jnp.linspace(x_min, x_max, nx)
        y_edges = jnp.linspace(y_min, y_max, ny)

        X, Y = jnp.meshgrid(x_edges, y_edges)

        fig = fig_handle.pcolormesh(X, Y, blk_data_component[i].transpose(1,0), shading='auto', vmin=vrange[0], vmax=vrange[1])

    return fig


def get_N_level_block_data(level, blk_data, blk_info):

    nx = n_grid[0][0] * 2**level
    ny = n_grid[0][1] * 2**level
    nU = blk_data.shape[1]
    U = jnp.zeros((nU, nx, ny))

    reshape_dims = [nU]
    for i in range(1, level + 1):
        reshape_dims.extend([n_block[i][0]])
    reshape_dims.append(n_grid[level][0] * 2)
    for i in range(1, level + 1):
        reshape_dims.extend([n_block[i][1]])
    reshape_dims.append(n_grid[level][1] * 2)

    U_reshaped = U.reshape(reshape_dims)
    transpose_order = [0] 
    for i in range(1, level + 2):
        transpose_order.append(i)
        transpose_order.append(i + level + 1)
    U_transposed = U_reshaped.transpose(transpose_order)

    index_columns = blk_info['glob_index'][:blk_info['number'], 2:2 + 2 * level]
    index_tuple = tuple(index_columns[:, i] for i in range(2 * level))

    blk_data_processed = blk_data[:blk_info['number']].transpose(1, 0, 2, 3)

    U_updated = U_transposed.at[(slice(None),) + index_tuple].set(blk_data_processed)

    def get_inverse_order(order):
        inv_order = [0] * len(order)
        for i, pos in enumerate(order):
            inv_order[pos] = i
        return inv_order
    inv_transpose_order = get_inverse_order(transpose_order)
    U_restored = U_updated.transpose(inv_transpose_order).reshape((4, nx, ny))

    return U_restored
