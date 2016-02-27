import numpy as np
import math
import scipy.stats as stats
from numba import jit
import matplotlib.pyplot as plt


T_array = [1, 1, 1, 3, 0.5, 1]
density_array = [0.88, 0.80, 0.70, 0.3, 1.2, 0.6873]

for z in range(len(T_array)):
    plt.close("all")
    # L determines the number of FCC cells in each spatial direction.
    # Each FCC cell contains 4 atoms.
    L = 6
    T = T_array[z]
    density = density_array[z]

    # The time step and the number of time steps are defined here. Relaxation_time is the time amount of timesteps
    # the system gets to reach a steady state (within this time the thermostat is used).
    dt = 0.004
    relaxation_time = 2500
    Nt = 100000 + relaxation_time

    # Initialize constants and variables needed for the statistics. Samples is the number of intervals
    # the measurement will be divided up in. The mean of each quantity will be calculated over this number of
    # samples and the error will be related to the variance between these samples
    samples = 20
    sample_index = 0
    sample_length = math.floor((Nt - relaxation_time) / samples)
    cv = np.zeros(samples)
    pressure_array = np.zeros(samples)
    distance = np.zeros(samples)
    diff_c = np.zeros(samples)
    e_pot_t_avg = np.zeros(samples)

    # Initialize all the variables needed for the calculations
    N = 4 * L ** 3
    box_size = (N / density) ** (1 / 3)
    eps_kb = 125
    e_kin = np.zeros(Nt)
    virial = np.zeros(Nt)
    mom_x = np.zeros(Nt)
    mom_y = np.zeros(Nt)
    mom_z = np.zeros(Nt)
    e_pot = np.zeros(Nt)
    pres = np.zeros(Nt)
    diff = np.zeros(Nt)
    distance = np.zeros((N, 3))
    bins = 300
    pc_r = np.linspace(0.001, box_size, bins)
    pc_dr = box_size / bins
    pc_n = np.zeros([bins])
    pc_n_tot = np.zeros([bins])
    pcf = np.zeros((samples, bins))
    pc_n_array = np.zeros((samples, bins))
    rc2 = 9.0
    e_pot_correction = 8 * math.pi * density * (N - 1) / (3 * math.sqrt(rc2)) * (1 / (3 * rc2 ** 4) - 1 / rc2)
    pres_correction = 48 * math.pi * density / (T * 9 * math.sqrt(rc2)) * (2 / (3 * rc2 ** 4) - 1 / rc2)

    # Here we open a new file to write our data in:
    name = "N" + str(N) + "_T" + repr(T) + "_roh" + repr(density) + ".txt"
    print(name)
    target = open(name, 'w')


    @jit
    def calc_forces(locations):
        f = np.zeros((N, 3))
        virial = 0
        potential = 0
        pc_n = np.zeros(shape=(bins,))
        # These for loops are configured to only take into account the upper triangle in the particle's matrix. This means that the force will be added to particle i and subtracted from particle j
        for i in range(N):
            for j in range(i + 1, N):
                # Determine the distances in each spatial direction:
                dx = locations[i, 0] - locations[j, 0]
                dy = locations[i, 1] - locations[j, 1]
                dz = locations[i, 2] - locations[j, 2]
                # Periodic boundary conditions:
                dx -= np.rint(dx / box_size) * box_size
                dy -= np.rint(dy / box_size) * box_size
                dz -= np.rint(dz / box_size) * box_size
                r2 = dx * dx + dy * dy + dz * dz
                # Implement the cut-off radius
                if r2 < rc2:
                    ir2 = 1 / r2
                    ir6 = ir2 * ir2 * ir2
                    ir12 = ir6 * ir6
                    # Implement Lennard-Jones
                    common_force_factor = 24 * ir2 * (2 * ir12 - ir6)
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
                    virial -= 24 * (2 * ir12 - ir6)
                # Place the particle in the appropriate bin for the correlation function:
                r = np.sqrt(r2)
                pc_n[int(r / pc_dr)] += 1
        return f, potential, virial, pc_n

    # This function initiates all the starting locations, forces and velocities.
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
        velocities *= math.sqrt((N - 1) * 3 * T / (2 * energy_kinetic))
        forces, potential, virial, pc_n = calc_forces(locations)
        return locations, velocities, forces

    # This function is used to make a time step, returning the new locations, velocities, forces and data extracted from the force calculation.
    def make_time_step(locations, velocities, old_f):
        velocities += 0.5 * old_f * dt
        locations += velocities * dt
        locations = np.mod(locations, box_size)
        new_f, potential, virial, pc_n = calc_forces(locations)
        velocities += 0.5 * new_f * dt
        return locations, velocities, new_f, potential, virial, pc_n

    # MAIN PART OF THE SCRIPT
    locs, velos, forces = initiate()
    for t in range(0, Nt):
        locs, velos, forces, e_pot[t], virial[t], pc_n = make_time_step(locs, velos, forces)
        e_kin[t] = 0.5 * np.sum(velos * velos)
        # Rescale the velocities in order to get temperature to the desired value, only for the first steps until equilibrium:
        if t < relaxation_time:
            velos *= math.sqrt((N - 1) * 3 * T / (2 * e_kin[t]))
            e_kin[t] = 0.5 * np.sum(velos * velos)
        # Processing of this sample:
        elif (t + 1 - relaxation_time) % sample_length == 0 and t > (
                        relaxation_time + sample_length - samples):
            print((t + 1 - relaxation_time))
            print("On sample", sample_index, "/", samples)
            # Calculate the diffusion coefficient for this interval:
            d2 = np.sum(distance ** 2)
            diff_c[sample_index] = d2 / (6 * N * sample_length * dt)
            # Calculate NPc for this interval:
            pc_n_array[sample_index] = pc_n_tot / sample_length
            # Reset the variables needed for the calculations:
            distance = np.zeros((N, 3))
            pc_n_tot = np.zeros((bins,))
            sample_index += 1
        # If the sample has not ended and the system is already in equilibrium, then make a normal timestep:
        else:
            distance += velos * dt
            pc_n_tot += pc_n

        mom_x[t] = sum(velos[:, 0])
        mom_y[t] = sum(velos[:, 1])
        mom_z[t] = sum(velos[:, 2])

    # Calculating the temperature, pressure, and potential energy
    temp = e_kin * 2 / (3 * N)
    pressure = 1 - 1 / (3 * N * temp[relaxation_time:]) * virial[relaxation_time:] + pres_correction
    e_pot_corrected = e_pot + e_pot_correction

    # Here we take a number of samples to determine the Cv over. Then a mean is calculated from these samples
    # And the error is determined according to the standard deviation.
    for i in range(samples):
        # Determine where the sample's interval starts and stops, the quantities will be calculated over this region
        interval_start = relaxation_time + i * sample_length
        interval_stop = interval_start + sample_length - 1
        # Calculation for the specific heat:
        k = np.mean(e_kin[interval_start:interval_stop])
        dk2 = np.mean(np.power(e_kin[interval_start:interval_stop] - k, 2))
        k2 = k ** 2
        cv[i] = 3 * k2 / (2 * k2 - 3 * N * dk2)
        # Calculate the pair correlation function on sample's interval. This is evaluated for every bin:
        pcf[i] = 2 * pc_n_array[i] / (4 * math.pi * pc_r ** 2 * pc_dr * density * (N - 1))

    # Pair correlation function mean and error calculation:
    pcf_mean = np.mean(pcf, axis=0)
    pcf_error = np.std(pcf, axis=0) / math.sqrt(samples)


    print("Emperical C_v = ", np.mean(cv), ", with error: ", np.std(cv) / math.sqrt(samples))
    print()
    print("Mean temperature: ", np.mean(temp[relaxation_time:]), " with error: ",
          np.std(temp[relaxation_time:]))
    print("The intended temperature was: ", T)
    print()
    print("Emprical self-diffusion coefficient: ", np.mean(diff_c), " with error: ", np.std(diff_c) / math.sqrt(samples))
    print()
    print("Empirical pressure: ", np.mean(pressure), " with error: ", np.std(pressure))
    print()
    print("Average potential energy: ", np.mean(e_pot_corrected[relaxation_time:]), "with error: ",
          np.std(e_pot_corrected[relaxation_time:]))

    # Write data to a text file:

    target.write("Emperical C_v = " + repr(np.mean(cv)) + ", with error: " + repr(np.std(cv) / math.sqrt(samples)))
    target.write("\n")
    target.write("Mean temperature: " + repr(np.mean(temp[relaxation_time:])) + " with error: " +
                 repr(np.std(temp[relaxation_time:])))
    target.write("\n")
    target.write("The intended temperature was: " + repr(T))
    target.write("\n")
    target.write("Emprical self-diffusion coefficient: " + repr(np.mean(diff_c)) + " with error: " + repr(
        np.std(diff_c) / math.sqrt(samples)))
    target.write("\n")
    target.write("Empirical pressure: " + repr(np.mean(pressure)) + " with error: " + repr(
        np.std(pressure)))
    target.write("\n")
    target.write("Average potential energy: " + repr(np.mean(e_pot_corrected[relaxation_time:])) + "with error: " + repr(
        np.std(e_pot_corrected[relaxation_time:])))
    target.write("\n")
    target.write("FOR LATEX TABLE:")
    target.write("\n")
    target.write(repr(density) + ' & ' + repr(T) + ' & ' + repr(np.mean(temp[relaxation_time:])) + ' & ' + repr(
        np.std(temp[relaxation_time:])) + ' & ' + repr(np.mean(pressure)) + "&" + repr(
        np.std(pressure)) + ' & ' + repr(np.mean(cv)) + " & " + repr(
        np.std(cv) / math.sqrt(samples)) + ' & ' + repr(np.mean(diff_c)) + " & " + repr(
        np.std(diff_c) / math.sqrt(samples)) + ' & ' + repr(np.mean(e_pot_corrected[relaxation_time:])) + " & " + repr(
        np.std(e_pot_corrected[relaxation_time:])))
    target.close()

    # PLOTS
    y_adjustment = 0.12
    legend_offset = -0.12

    name = "N" + str(N) + "_T" + repr(T) + "_roh" + repr(density) + "_t" + repr(Nt)
    name = name.replace(".", "")

    fig1 = plt.figure()
    ax1 = plt.subplot(111)
    plt.plot(pc_r, pcf_mean, label='$g(r/\sigma)$')
    plt.plot(pc_r, np.ones(bins), '--', label='$g(r/\sigma)=1$')
    plt.fill_between(pc_r, pcf_mean - pcf_error, pcf_mean + pcf_error, label='error')
    plt.xlabel('$r/\sigma$')
    plt.ylabel('$g(r/\sigma)$')
    plt.xlim([0, 0.5 * box_size])
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * y_adjustment,
                      box.width, box.height * 0.9])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, legend_offset),
               fancybox=True, shadow=True, ncol=3)
    plt.show()
    fig1.savefig(name + "_pcf.eps", format='eps', dpi=1000)

    fig2 = plt.figure()
    ax = plt.subplot(111)
    linee_pot, = plt.plot(range(Nt), e_pot, label="$E_{pot}$")
    linee_kin, = plt.plot(range(Nt), e_kin, label="$E_{kin}$")
    line_E, = plt.plot(range(Nt), e_kin + e_pot, label="$E$")
    box = ax.get_position()
    plt.xlim([0, Nt])
    plt.ylabel('$E(r/\sigma)/\epsilon$')
    plt.xlabel('$t\sqrt{\epsilon/m}/\sigma$')
    ax.set_position([box.x0, box.y0 + box.height * y_adjustment,
                     box.width, box.height * 0.9])
    plt.legend(handles=[linee_pot, line_E, linee_kin], loc='upper center', bbox_to_anchor=(0.5, legend_offset),
               fancybox=True, shadow=True, ncol=3)
    plt.show()
    fig2.savefig(name + "_energies.eps", format='eps', dpi=1000)

    fig3 = plt.figure()
    ax = plt.subplot(111)
    line_mom_x, = plt.plot(range(Nt), mom_x, label="$p_x$")
    line_mom_y, = plt.plot(range(Nt), mom_y, label="$p_y$")
    line_mom_z, = plt.plot(range(Nt), mom_z, label="$p_z$")
    box = ax.get_position()
    plt.ylim([-1, 1])
    plt.xlim([0, Nt])
    plt.ylabel('$p/\sqrt{m\epsilon}$')
    plt.xlabel('$t\sqrt{\epsilon/m}/\sigma$')
    ax.set_position([box.x0, box.y0 + box.height * y_adjustment,
                     box.width, box.height * 0.9])
    plt.legend(handles=[line_mom_x, line_mom_y, line_mom_z], loc='upper center', bbox_to_anchor=(0.5, legend_offset),
               fancybox=True, shadow=True, ncol=3)
    plt.show()
    fig3.savefig(name + "_momenta.eps", format='eps', dpi=1000)
