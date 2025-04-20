# Copyright © 2025 Haocheng Wen, Faxuan Luo
# SPDX-License-Identifier: MIT

import jax
import jax.numpy as jnp
from jax import jit

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

