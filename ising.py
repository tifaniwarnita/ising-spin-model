from scipy.optimize import fsolve
import math
import matplotlib.pyplot as plt
import numpy as np
import random

# Constants
J = 1
h = 0.5
n_spin = 5
spin_value = [-1, 1]

# Spins
def create_spins(n_spin):
	S = [None] * n_spin
	for i in range (0, n_spin):
		S[i] = random.choice(spin_value)
	return S

def tanh_taylor(x):
	return x - (1 / 3 * x**3 )+ (2 / 15 * x**5) # + ...

# (a) NAIVE MEAN FIELD APPROXIMATION
# Basic idea of mean-field theory is to set the fluctuations into zero.
def spin_avg_i_mfa(i, spins, T):
	z = 0
	neighbours = 0
	if i > 0:
		z += 1
	if i < len(spins) - 1:
		z += 1

	b = 1/T
	func = lambda m : np.tanh((b * z * J * m) + (b * h)) - m
	m_initial_guess = 1
	res = fsolve(func, m_initial_guess)
	return res

# (b) CAVITY METHOD
def spin_avg_i_cavity(i, spins, T):
	z = 0
	neighbours = 0
	if i > 0:
		z += 1
	if i < len(spins) - 1:
		z += 1
	# replace z into 1.00000000001 if z = 1 to avoid division by zero
	if z == 1:
		z = 1.00000000001

	b = 1/T

	func = lambda tetha : (z-1) / b * np.arctanh(np.tanh(b * J) * np.tanh(b * tetha))
	tetha_initial_guess = 1
	tetha = fsolve(func, tetha_initial_guess)

	func = lambda m : np.tanh(z / (z-1) * b * tetha) - m
	m_initial_guess = 1
	res = fsolve(func, m_initial_guess)

	return res

# (c) EXACT SOLUTION
# Hamiltonian
def hamiltonian_i(i, value, spins):
	a = 0
	if i > 0:
		a += spins[i-1]
	if i < len(spins) - 1:
		a += spins[i+1]

	H = (-J * value * a) - (h * value)
	return H

def partition_function_i(i, spins, T):
	b = 1/T
	Z = math.exp(b * hamiltonian_i(i, spins[i], spins,)) + math.exp(b * hamiltonian_i(i, -spins[i], spins))
	return Z

def boltzmann_distribution(i, value, spins, T):
	b = 1/T
	d = math.exp(b * hamiltonian_i(i, value, spins)) / partition_function_i(i, spins, T)
	return d

def spin_avg_i(i, spins, T):
	m = spins[i] * boltzmann_distribution(i, spins[i], spins, T) + spins[i] * boltzmann_distribution(i, -spins[i], spins, T)
	return m

def plot_naive_mfa():
	spins = create_spins(n_spin)

	for i, s in enumerate(spins):
		print("m -", i)
		T = np.linspace(0, 15, 100)
		avg = []
		for t in T:
			avg.append(spin_avg_i_mfa(i, spins, t))
		plt.plot(T, avg)
		plt.title('Naive MFA - m' + str(i+1))
		plt.xlabel("Temperature")
		plt.ylabel("Spin Average")
		plt.grid()
		plt.show()

def plot_cavity_field():
	spins = create_spins(n_spin)

	for i, s in enumerate(spins):
		print("m -", i)
		T = np.linspace(0, 0.5, 100)
		avg = []
		for t in T:
			avg.append(spin_avg_i_cavity(i, spins, t))
		plt.plot(T, avg)
		plt.title('Cavity Method - m' + str(i+1))
		plt.xlabel("Temperature")
		plt.ylabel("Spin Average")
		plt.grid()
		plt.show()

def plot_exact_computation():
	spins = create_spins(n_spin)

	for i, s in enumerate(spins):
		print("m -", i)
		T = np.linspace(0, 0.5, 100)
		avg = []
		for t in T:
			avg.append(spin_avg_i(i, spins, t))
		plt.plot(T, avg)
		plt.title('Cavity Method - m' + str(i+1))
		plt.xlabel("Temperature")
		plt.ylabel("Spin Average")
		plt.grid()
		plt.show()

def plot_comparison():
	spins = create_spins(n_spin)

	for i, s in enumerate(spins):
		print("m -", i)
		T = np.linspace(0, 15, 100)
		avg_mfa = []
		avg_cavity = []
		avg_exact = []
		for t in T:
			avg_mfa.append(spin_avg_i_mfa(i, spins, t))
			avg_cavity.append(spin_avg_i_cavity(i, spins, t))
			avg_exact.append(spin_avg_i(i, spins, t))
		plt.plot(T, avg_mfa)
		plt.plot(T, avg_cavity)
		plt.plot(T, avg_exact)
		plt.title('Cavity Method - m' + str(i+1))
		plt.xlabel("Temperature")
		plt.ylabel("Spin Average")
		plt.grid()
		plt.show()

def main():
	plot_naive_mfa()
	plot_cavity_field()
	plot_exact_computation()
	plot_comparison()



if __name__ == "__main__":
	main()