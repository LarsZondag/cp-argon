import numpy as np
import math
import scipy.stats as stats
from numba import jit
import matplotlib.pyplot as plt

# L determines the number of FCC cells in each spatial direction.
# Each FCC cell contains 4 atoms.
L = 3
N = 4 * L ** 3
T = 3.1
density = 0.3
box_size = (N / density) ** (1 / 3)
dt = 0.004
relaxation_time = 500
Nt = 500 + relaxation_time
eps_kb = 125
e_kt = np.zeros(Nt)
mom_x = np.zeros(Nt)
mom_y = np.zeros(Nt)
mom_z = np.zeros(Nt)
e_pot = np.zeros(Nt)
temp = np.zeros(Nt)
cv = np.zeros(Nt)
pres = np.zeros(Nt)
diff = np.zeros(Nt)
msd = 0
bins = 150
drPC = 5 / bins
rPC = np.linspace(0.001, box_size * 0.5, bins)


@jit
def calc_forces(locations):
    rc2 = 9.0
    f = np.zeros((N, 3))
    virial = np.zeros(N)
    potential = 0
    nPC = np.zeros([len(rPC)])
    # These for-loops fill the distances array with the appropriate distance. Notice distances = -distances^T
    # In the loop a check is made to make sure the right images are used (periodic boundary conditions)
    for i in range(N):
        for j in range(i + 1, N):
            dis_vec = locations[i] - locations[j]
            dx = locations[i, 0] - locations[j, 0]
            dy = locations[i, 1] - locations[j, 1]
            dz = locations[i, 2] - locations[j, 2]
            dis_vec -= np.rint(dis_vec / box_size) * box_size
            dx -= np.rint(dx / box_size) * box_size
            dy -= np.rint(dy / box_size) * box_size
            dz -= np.rint(dz / box_size) * box_size
            r2 = np.sum(dis_vec ** 2)
            if r2 < rc2:
                ir2 = 1 / r2
                ir6 = ir2 * ir2 * ir2
                ir12 = ir6 * ir6
                # Implement Lennard-Jones
                common_force_factor = 24 * ir2 * (2 * ir12 - ir6) # * dis_vec
                fx = common_force_factor * dx
                fy = common_force_factor * dy
                fz = common_force_factor * dz
                f[i, 0] += fx
                f[i, 1] += fy
                f[i, 2] += fz
                f[j, 0] -= fx
                f[j, 1] -= fy
                f[j, 2] -= fz
                # f[j] -= common_force_factor
                common_potential = 4 * (ir12 - ir6)
                potential += common_potential
                virial[i] += np.dot(f[i], [[dx], [dy], [dz]])
                virial[j] -= np.dot(f[j], [[dx], [dy], [dz]])

    return f, potential, virial


def initiate():
    # velocities = np.zeros((N, 3))
    locations = np.zeros((N, 3))
    fcc = np.array([  # These are the locations available within one cell of the FCC structure
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5]])
    index = 0
    for i in range(L):
        for j in range(L):
            for k in range(L):
                for z in range(4):
                    locations[index] = np.add(fcc[z], np.array([i, j, k]))
                    index += 1
    locations *= box_size / L
    # a = math.sqrt(T)
    # speeds = stats.maxwell.rvs(loc=0, scale=a, size=N)  # This takes random maxwell-distributed values
    # # Take N normally distributed values, this is the base for our speeds.
    velocities = stats.norm.rvs(size=(N, 3))
    #
    # # After this we scale the ran_velos such that the vector is the same size as speeds.
    # # Then it will correspond to a maxwell distribution with random directions.
    # for i in range(N):
    #     normVelocity = np.linalg.norm(ran_velos[i, :])
    #     scaling = speeds[i] / normVelocity
    #     velocities[i] = ran_velos[i] * scaling
    #
    # # Now make sure there is no net-movement of the particles (take of the average)
    velocities -= np.mean(velocities, axis=0)
    energy_kinetic = 0.5 * np.sum(velocities * velocities)
    velocities *= math.sqrt(N * 3 * T / (2 * energy_kinetic))
    forces, potential, virial = calc_forces(locations)
    return locations, velocities, forces  # , potential, energy_kinetic, momentum


def make_time_step(locations, velocities, old_f):
    velocities += 0.5 * old_f * dt
    locations += velocities * dt
    locations = np.mod(locations, box_size)
    new_f, potential, virial = calc_forces(locations)
    velocities += 0.5 * new_f * dt
    return locations, velocities, new_f, potential, virial


locs, velos, forces = initiate()
for t in range(0, Nt):
    locs, velos, forces, e_pot[t], virial = make_time_step(locs, velos, forces)
    e_kt[t] = 0.5 * np.sum(velos * velos)
    if t < relaxation_time:
        # Optionally rescale the velocies in order to make temperature constant:
        velos *= math.sqrt(N * 3 * T / (2 * e_kt[t]))
        e_kt[t] = 0.5 * np.sum(velos * velos)
    mom_x[t] = sum(velos[:, 0])
    mom_y[t] = sum(velos[:, 1])
    mom_z[t] = sum(velos[:, 2])
    pres[t] = density / (3 * N) * (2 * e_kt[t] + np.sum(virial))
    msd += (velos * dt) ** 2
    diff[t] = 1 / 6 / N / (t + 1) * np.sum(msd)
temp = e_kt * 2 / (3 * N)
samples = 2
cv = np.zeros(samples)
for i in range(samples):
    interval_start = relaxation_time + i * math.floor((Nt - relaxation_time) / samples)
    interval_stop = interval_start + math.floor((Nt - relaxation_time) / samples)
    std_tmp = np.mean((temp[interval_start:interval_stop] - np.mean(temp[interval_start:interval_stop])) ** 2)
    cv[i] = ((2 / (3 * N) - std_tmp / (np.mean(temp[interval_start:interval_stop]) ** 2)) ** (-1)) / N * 2 / 3
print("Theoretical Cv = ", 3 / T)
print("Emperical Cv = ", np.mean(cv), ", with error: ", np.std(cv) / math.sqrt(samples))
print("mean temp", np.mean(temp[relaxation_time:]))

# print(avg_temp)
# print(temp)
# print(temp)
# print(e_kt)
# linee_pot, = plt.plot(range(Nt), e_pot, label="Potential energy")
# line_E, = plt.plot(range(Nt), e_kt + e_pot, label="Total energy")
# linee_kt, = plt.plot(range(Nt), e_kt, label="Kinetic energy")
# # lineMom, = plt.plot(range(Nt), mom, label="Momentum")
# line_temp, = plt.plot(range(Nt), temp, label="Temperature")
# # line_avg_temp, = plt.plot(range(len(avg_temp)), avg_temp, label="Average Temperature")
# # line_cv, = plt.plot(range(Nt), cv, label="Heat Capacity")
# # line_avg_cv, = plt.plot(range(len(avg_cv)), avg_cv, label="Average Heat Capacity")
# #
# #
# plt.legend(handles=[line_temp])
# plt.show()
