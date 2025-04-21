# Copyright © 2025 Haocheng Wen, Faxuan Luo
# SPDX-License-Identifier: MIT

import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import cfd_solver
from cfd_solver import rhs
import jaxamr.amr as amr

template_node_num = amr.template_node_num

def initialize(nx, ny, gamma=1.4):
    x = jnp.linspace(0, 1, nx)
    y = jnp.linspace(0, 1, ny)
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    rho = jnp.where(jnp.sqrt((X - 0.5)**2 + (Y - 0.5)**2) < 0.15, 1.0, 0.125)
    u = jnp.zeros_like(X)
    v = jnp.zeros_like(X)
    p = jnp.where(jnp.sqrt((X - 0.5)**2 + (Y - 0.5)**2) < 0.15, 1.0, 0.1)
    E = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2)

    U = jnp.array([rho, rho * u, rho * v, E])
    return X, Y, U


@partial(jit, static_argnames=('level'))
def rk2(level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info):

    num = template_node_num

    ghost_blk_data = amr.get_ghost_block_data(ref_blk_data, ref_blk_info)
    blk_data1 = ghost_blk_data + 0.5 * dt * vmap(rhs, in_axes=(0, None, None))(ghost_blk_data, dx, dy)
    blk_data1 = amr.update_external_boundary(level, blk_data, blk_data1[..., num:-num, num:-num], ref_blk_info)


    ghost_blk_data1 = amr.get_ghost_block_data(blk_data1, ref_blk_info)
    blk_data2 = ghost_blk_data + dt * vmap(rhs, in_axes=(0, None, None))(ghost_blk_data1, dx, dy)
    blk_data2 = amr.update_external_boundary(level, blk_data, blk_data2[..., num:-num, num:-num], ref_blk_info)

    return blk_data2


@jit
def rk2_L0(blk_data, dx, dy, dt):

    U = blk_data[0]

    U1 = U + 0.5 * dt * rhs(U, dx, dy)
    U2 = U + dt * rhs(U1, dx, dy)

    return jnp.array([U2])
