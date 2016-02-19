import numpy as np
import math
import scipy.stats as stats
from numba import jit
import matplotlib.pyplot as plt

# L determines the number of FCC cells in each spatial direction.
# Each FCC cell contains 4 atoms.
L = 2
T = 1
density = 0.8

# The time step and the number of time steps are defined here. Relaxation_time is the time amount of timesteps
# the system gets to reach a steady state (within this time the thermostat is used).
dt = 0.004
relaxation_time = 1000
Nt = 5000 + relaxation_time


# Initialize constants and variables needed for the statistics. Samples is the number of intervals
# the measurement will be divided up in. The mean of each quantity will be calculated over this number of
# samples and the error will be related to the variance between these samples
samples = 5
sample_index = 0
sample_length = math.floor((Nt - relaxation_time) / samples)
cv = np.zeros(samples)
distance = np.zeros(samples)
diff_c = np.zeros(samples)

# Initialize all the variables needed for the calculations
N = 4 * L ** 3
box_size = (N / density) ** (1 / 3)
eps_kb = 125
e_kt = np.zeros(Nt)
virial = np.zeros(Nt)
mom_x = np.zeros(Nt)
mom_y = np.zeros(Nt)
mom_z = np.zeros(Nt)
e_pot = np.zeros(Nt)
temp = np.zeros(Nt)
pres = np.zeros(Nt)
diff = np.zeros(Nt)
distance = np.zeros((N, 3))
bins = 150
drPC = 1 / bins
rPC = np.linspace(0.001, box_size * 0.5, bins)
nPC = np.zeros([len(rPC)])
nPCtot = np.zeros([len(rPC)])
PCF = np.zeros([len(rPC)])


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
            # dis_vec = locations[i] - locations[j]
            dx = locations[i, 0] - locations[j, 0]
            dy = locations[i, 1] - locations[j, 1]
            dz = locations[i, 2] - locations[j, 2]
            # dis_vec -= np.rint(dis_vec / box_size) * box_size
            dx -= np.rint(dx / box_size) * box_size
            dy -= np.rint(dy / box_size) * box_size
            dz -= np.rint(dz / box_size) * box_size
            # r2 = np.sum(dis_vec ** 2)
            r2 = dx * dx + dy * dy + dz * dz
            if r2 < rc2:
                ir2 = 1 / r2
                ir6 = ir2 * ir2 * ir2
                ir12 = ir6 * ir6
                # Implement Lennard-Jones
                common_force_factor = 24 * ir2 * (2 * ir12 - ir6)  # * dis_vec
                fx = common_force_factor * dx
                fy = common_force_factor * dy
                fz = common_force_factor * dz
                f[i, 0] += fx
                f[i, 1] += fy
                f[i, 2] += fz
                f[j, 0] -= fx
                f[j, 1] -= fy
                f[j, 2] -= fz
                potential += 4 * (ir12 - ir6)
                common_virial = fx * dx + fy * dy + fz * dz
                virial[i] += common_virial
                virial[j] -= common_virial
            for k in range(len(rPC)):
                if (r2 > rPC[k] * rPC[k]) and (r2 < ((rPC[k] + drPC) * (rPC[k] + drPC))):
                    nPC[k] += 1
    return f, potential, sum(virial), nPC


def initiate():
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
    # Take N normally distributed values, this is the base for our speeds.
    velocities = stats.norm.rvs(size=(N, 3))
    # Now make sure there is no net-movement of the particles (take of the average)
    velocities -= np.mean(velocities, axis=0)
    energy_kinetic = 0.5 * np.sum(velocities * velocities)
    velocities *= math.sqrt(N * 3 * T / (2 * energy_kinetic))
    forces, potential, virial, nPC = calc_forces(locations)
    return locations, velocities, forces


def make_time_step(locations, velocities, old_f):
    velocities += 0.5 * old_f * dt
    locations += velocities * dt
    locations = np.mod(locations, box_size)
    new_f, potential, virial, nPC = calc_forces(locations)
    velocities += 0.5 * new_f * dt
    return locations, velocities, new_f, potential, virial, nPC


locs, velos, forces = initiate()
for t in range(0, Nt):
    locs, velos, forces, e_pot[t], virial[t], nPC = make_time_step(locs, velos, forces)
    nPCtot += nPC
    e_kt[t] = 0.5 * np.sum(velos * velos)
    # Optionally rescale the velocies in order to make temperature constant:
    if t < relaxation_time:
        velos *= math.sqrt(N * 3 * T / (2 * e_kt[t]))
        e_kt[t] = 0.5 * np.sum(velos * velos)
    elif (t + 1 - relaxation_time) % sample_length == 0 and t > (relaxation_time + sample_length - samples): # Calculation of the self-diffusion coefficient:
        print((t + 1 - relaxation_time))
        dx = distance[:,0]
        dy = distance[:,1]
        dz = distance[:,2]
        d2 = sum(dx * dx) + sum(dy * dy) + sum(dz * dz)
        diff_c[sample_index] = d2 / (6 * N * sample_length * dt)
        distance = np.zeros((N,3))
        sample_index += 1
    else:
        distance += velos * dt

    mom_x[t] = sum(velos[:, 0])
    mom_y[t] = sum(velos[:, 1])
    mom_z[t] = sum(velos[:, 2])


# Calculating the pair correlation function
nPCavg = nPCtot / Nt
for p in range(len(rPC)):
    PCF[p] = 2 * nPCavg[p] / (4 * math.pi * rPC[p] * rPC[p] * drPC * density * (N - 1))
    

# Calculating the temperature and pressure
temp = e_kt * 2 / (3 * N)
pres = density / (3 * N) * (2 * e_kt + virial)
print(diff_c)

# Here we take a number of samples to determine the Cv over. Then a mean is calculated from these samples
# And the error is determined according to the standard deviation.

for i in range(samples):
    interval_start = relaxation_time + i * sample_length
    interval_stop = interval_start + sample_length - 1
    k = np.mean(e_kt[interval_start:interval_stop])
    dk2 = np.mean((e_kt[interval_start:interval_stop] - k) ** 2)
    k2 = k * k
    cv[i] = 3 * k2 / (2 * k2 - 3 * N * dk2)
print("Theoretical Cv = 1.5")
print("Emperical Cv = ", np.mean(cv), ", with error: ", np.std(cv) / math.sqrt(samples))
print("mean temp", np.mean(temp[relaxation_time:]))

# PLOTS

plt.plot(rPC, np.ones([len(rPC)]), '--', rPC, PCF)
plt.xlabel(r'r/$\sigma$')
plt.ylabel('g(r)')
plt.show()

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
