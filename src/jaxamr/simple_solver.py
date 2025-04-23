# Copyright © 2025 Haocheng Wen, Faxuan Luo
# SPDX-License-Identifier: MIT

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

template_node_num = amr_config['template_node_num']

def flux(U, gamma=1.4):
    rho = U[0]
    u = U[1] / rho
    v = U[2] / rho
    E = U[3]
    p = (gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))

    F = jnp.array([
        rho * u,
        rho * u**2 + p,
        rho * u * v,
        u * (E + p)
    ])

    G = jnp.array([
        rho * v,
        rho * u * v,
        rho * v**2 + p,
        v * (E + p)
    ])

    return F, G


def lax_friedrichs_flux(U_L, U_R, gamma=1.4):
    F_L, G_L = flux(U_L, gamma)
    F_R, G_R = flux(U_R, gamma)

    # jnp.nanmax is used instead of jnp.max
    lambda_max = jnp.nanmax(jnp.abs(U_L[1]/U_L[0]) + jnp.sqrt(gamma * (gamma - 1) * (U_L[3]/U_L[0] - 0.5 * (U_L[1]**2 + U_L[2]**2)/U_L[0]**2)))

    return 0.5 * (F_L + F_R) - 0.5 * lambda_max * (U_R - U_L), \
           0.5 * (G_L + G_R) - 0.5 * lambda_max * (U_R - U_L)

@jit
def rhs(U, dx, dy, gamma=1.4):

    U_L_x = U[:, :-1, :] 
    U_R_x = U[:, 1:, :]
    F_LR_x, _ = lax_friedrichs_flux(U_L_x, U_R_x, gamma)
    F_x = jnp.zeros_like(U)
    F_x = F_x.at[:, 1:-1, :].set(- (F_LR_x[:, 1:, :] - F_LR_x[:, :-1, :]) / dx)

    U_L_y = U[:, :, :-1]
    U_R_y = U[:, :, 1:]
    _, G_LR_y = lax_friedrichs_flux(U_L_y, U_R_y, gamma)
    F_y = jnp.zeros_like(U)
    F_y = F_y.at[:, :, 1:-1].set(- (G_LR_y[:, :, 1:] - G_LR_y[:, :, :-1]) / dy)

    return F_x + F_y


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

