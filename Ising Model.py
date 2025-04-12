import matplotlib.pyplot as plt
from scipy.ndimage import convolve, generate_binary_structure
import numpy as np
import random


def initialize_lattice(N):
    init_random = np.random.random((N, N))
    lattice = np.zeros((N, N))
    x = random.uniform(0, 1)
    lattice[init_random <= x] = 1
    lattice[init_random > x] = -1
    return lattice


def plot_lattice(lattice, title):
    plt.imshow(lattice, cmap='viridis')
    plt.title(title)
    plt.show()


def calc_energy(lattice_a):
    kern = generate_binary_structure(2, 1)
    kern[1][1] = False
    arr = -lattice_a * convolve(lattice_a, kern, mode='constant', cval=0)
    return arr.sum()


def metropolis(spin_arr, times, BJ, energy):
    spin_arr = spin_arr.copy()
    net_spins = np.zeros(times - 1)
    net_energy = np.zeros(times - 1)
    for t in range(0, times - 1):
        x = np.random.randint(0, N)
        y = np.random.randint(0, N)
        spin_i = spin_arr[x, y]
        spin_f = spin_i * -1

        E_i = 0
        E_f = 0
        if x > 0:
            E_i += -spin_i * spin_arr[x - 1, y]
            E_f += -spin_f * spin_arr[x - 1, y]
        if x < N - 1:
            E_i += -spin_i * spin_arr[x + 1, y]
            E_f += -spin_f * spin_arr[x + 1, y]
        if y > 0:
            E_i += -spin_i * spin_arr[x, y - 1]
            E_f += -spin_f * spin_arr[x, y - 1]
        if y < N - 1:
            E_i += -spin_i * spin_arr[x, y + 1]
            E_f += -spin_f * spin_arr[x, y + 1]

        dE = E_f - E_i
        if (dE > 0) and (np.random.random() < np.exp(-BJ * dE)):
            spin_arr[x, y] = spin_f
            energy += dE
        elif dE <= 0:
            spin_arr[x, y] = spin_f
            energy += dE

        net_spins[t] = spin_arr.sum()
        net_energy[t] = energy

    return net_spins, net_energy


def get_spin_energy(lattice_a, BJs):
    ms = np.zeros(len(BJs))
    E_means = np.zeros(len(BJs))
    E_stds = np.zeros(len(BJs))
    E_var = np.zeros(len(BJs))
    initial_energy = calc_energy(lattice_a)
    for i, bj in enumerate(BJs):
        spins, energies = metropolis(lattice_a, 100000, bj, initial_energy)
        ms[i] = spins[-10000:].mean() / N ** 2
        E_means[i] = energies[-10000:].mean()
        E_stds[i] = np.std(energies[-10000:])
        E_var[i] = np.var(spins[-10000:])
    return ms, E_means, E_stds, E_var


# Main part of the code
N = 100
lattice = initialize_lattice(N)

# Display the initial lattice on the same page
plot_lattice(lattice, 'Initial Lattice')

# Run Metropolis algorithm
spins, energies = metropolis(lattice, 100000, 0.7, calc_energy(lattice))

# Analyze spin and energy at different temperatures
Ts = np.linspace(0.2, 4.0, 100)
BJs = 1 / Ts
ms_n, E_means_n, E_stds_n, E_var_n = get_spin_energy(lattice, BJs)


plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(lattice, cmap='viridis')
plt.title('Initial Lattice')

plt.subplot(2, 2, 2)
plt.plot(spins / N ** 2, label=r'$\bar{m}$', color='blue')
plt.xlabel('Algorithm Time Steps')
plt.ylabel(r'Average Spin $\bar{m}$')
plt.grid()
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(energies, label=r'$E/J$', color='red')
plt.xlabel('Algorithm Time Steps')
plt.ylabel('Energy $E/J$')
plt.grid()
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(Ts, ms_n, 'o-', label=r'$\bar{m}$', color='blue')
plt.xlabel(r'$\left(\frac{k}{J}\right)T$')
plt.ylabel(r'$\bar{m}$')
plt.title('Average Spin vs Temperature')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
