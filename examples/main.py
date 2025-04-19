import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
import amrsolver as solver
import jaxamr as amr
import amraux as aux
import config
from config import dx, dy, n_grid

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'gpu')

X, Y, U = solver.initialize(n_grid[0][0], n_grid[0][1])

blk_data0 = jnp.array([U]) 

blk_info0 = {
      'number': 1,
      'index': jnp.array([0, 0, 0]),
      'glob_index': jnp.array([[0, 0]]),
      'neighbor_index': jnp.array([[-1, -1, -1, -1]])
        }

dt = 0.00006 * 8 

nt = 30

amr_update_step = 2

amr_initialized = False

for step in tqdm(range(nt), desc="Progress", unit="step"):

    if amr_initialized == False :

        blk_data1, blk_info1, max_blk_num1 = amr.initialize(1, blk_data0, blk_info0, 'density', dx[1], dy[1])
        blk_data2, blk_info2, max_blk_num2 = amr.initialize(2, blk_data1, blk_info1, 'density', dx[2], dy[2])
        blk_data3, blk_info3, max_blk_num3 = amr.initialize(3, blk_data2, blk_info2, 'density', dx[3], dy[3])

        amr_initialized = True

    elif (step % amr_update_step == 0):
        blk_data1, blk_info1, max_blk_num1 = amr.update(1, blk_data0, blk_info0, 'density', dx[1], dy[1], blk_data1, blk_info1, max_blk_num1)
        blk_data2, blk_info2, max_blk_num2 = amr.update(2, blk_data1, blk_info1, 'density', dx[2], dy[2], blk_data2, blk_info2, max_blk_num2)
        blk_data3, blk_info3, max_blk_num3 = amr.update(3, blk_data2, blk_info2, 'density', dx[3], dy[3], blk_data3, blk_info3, max_blk_num3)

    # Crossover advance
    for _ in range(2):
        for _ in range(2):
            for _ in range(2):
                blk_data3 = solver.rk2(3, blk_data2, dx[3], dy[3], dt/8.0, blk_data3, blk_info3)
            blk_data2 = solver.rk2(2, blk_data1, dx[2], dy[2], dt/4.0, blk_data2, blk_info2)
        blk_data1 = solver.rk2(1, blk_data0, dx[1], dy[1], dt/2.0, blk_data1, blk_info1)
    blk_data0 = solver.rk2_L0(blk_data0, dx[0], dy[0], dt)


    # Synchronous advance
    #blk_data3 = solver.rk2(3, blk_data2, dx[3], dy[3], dt/8.0, blk_data3, blk_info3)
    #blk_data2 = solver.rk2(2, blk_data1, dx[2], dy[2], dt/8.0, blk_data2, blk_info2)
    #blk_data1 = solver.rk2(1, blk_data0, dx[1], dy[1], dt/8.0, blk_data1, blk_info1)
    #blk_data0 = solver.rk2_L0(blk_data0, dx[0], dy[0], dt/8.0)


    blk_data2 = amr.interpolate_fine_to_coarse(3, blk_data2, blk_data3, blk_info3)
    blk_data1 = amr.interpolate_fine_to_coarse(2, blk_data1, blk_data2, blk_info2)
    blk_data0 = amr.interpolate_fine_to_coarse(1, blk_data0, blk_data1, blk_info1)



plt.figure(figsize=(10, 8))
ax = plt.gca()

component = 0
vrange = (0, 1)
fig = aux.plot_block_data(blk_data0[:, component], blk_info0, ax, vrange) # Level 0
fig = aux.plot_block_data(blk_data1[:, component], blk_info1, ax, vrange) # Level 1
fig = aux.plot_block_data(blk_data2[:, component], blk_info2, ax, vrange) # Level 2
fig = aux.plot_block_data(blk_data3[:, component], blk_info3, ax, vrange) # Level 3

plt.colorbar(fig, ax=ax, label='Density')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.axis('equal')
plt.show()
