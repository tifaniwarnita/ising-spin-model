# meanfield.py
# Name       : Tifani Warnita
# Student ID : 17M38271
# November 2017

from mpmath import taylor
from sympy import atanh, Eq, log, solve, symbols, tanh
import math
import matplotlib.pyplot as plt
import numpy as np
import random


# Constants
J = 1
h = 0.5
beta = 1
n_spin = 7
spin_value = [-1, 1]

# Spins
def create_spins(n_spin):
	S = [None] * n_spin
	for i in range (0, n_spin):
		S[i] = random.choice(spin_value)
	return S

# (a) NAIVE MEAN FIELD APPROXIMATION
# Basic idea of mean-field theory is to set the fluctuations into zero.
def mfa_tanh_spin_average(spins):
	z = 2
	m = np.tanh((beta * z * J * np.mean(spins)) + (beta * h))
	return m

def mfa_hamiltonian(spins):
	a = np.mean(spins) * sum(spins)
	b = sum(spins)

	z = len(spins) - 1
	H = (-J * z * a) - (h * b)
	return H

# Partition function
def mfa_partition_function(spins):
	z = 0
	for i in range(0, len(spins)):
		z += math.exp(-beta * mfa_hamiltonian(spins))
	return z

# Spin average
def mfa_spin_average(s0, spins):
	m = 0
	for i in range(0, len(spins)):
		denom = 0
		m += (s0 * math.exp(-beta * mfa_hamiltonian(spins)))/mfa_partition_function(spins)
	return m

def one_mfa_spin_average(i, spins):
	z = 0
	neighbours = 0
	if i > 0:
		neighbours += spins[i-1]
		z += 1
	if i < len(spins) - 1:
		neighbours += spins[i+1]
		z += 1

	m = symbols('m', real=True)
	x = (beta * z * J * m) + (beta * h)
	res = solve(Eq(tanh_taylor(x), m),m)

	return res[0]

def tanh_taylor(x):
	return x - 1 / 3 * x**3 + 2 / 15 * x**5 # + ...

def arc_tanh_taylor(x):
	return x + 1/3 * x**3 + 1/5 * x**5 # + 1/7 * x**7 + 1/9 * x**9 + 1/11 * x**11

# (b) CAVITY METHOD
def one_bethe_spin_average(i, spins):
	z = 0
	neighbours = 0
	if i > 0:
		neighbours += spins[i-1]
		z += 1
	if i < len(spins) - 1:
		neighbours += spins[i+1]
		z += 1

	print(z)
	tetha, m = symbols('tetha, m', real=True)
	x = tanh_taylor(beta * J) * tanh_taylor(beta * tetha)
	print("x:", x)
	print("y:", (z-1) / beta * atanh(x))
	res_tetha = solve(Eq((z-1) / beta * arc_tanh_taylor(x), tetha), tetha)
	print("tetha:", res_tetha)
	if z-1 == 0:
		res_m = [0]
	else:
		res_m = solve(Eq(tanh(z / (z-1) * beta * res_tetha[0]), m),m)

	return res_m[0]


# (c) EXACT SOLUTION
# Hamiltonian
def hamiltonian(spins):
	a = 0
	for i in range(0, len(spins)-1):
		a += spins[i] * spins[i+1]

	b = sum(spins)

	H = (-J * a) - (h * b)
	return H

def one_hamiltonian(i, value, spins):
	a = 0
	if i > 0:
		a += spins[i-1]
	if i < len(spins) - 1:
		a += spins[i+1]

	H = (-J * value * a) - (h * value)
	return H

def one_partition_function(i, spins):
	Z = math.exp(beta * one_hamiltonian(i, spins[i], spins)) + math.exp(beta * one_hamiltonian(i, -spins[i], spins))
	return Z

def boltzmann_distribution(i, value, spins):
	d = math.exp(beta * one_hamiltonian(i, value, spins)) / one_partition_function(i, spins)
	return d

def one_spin_average(i, spins):
	m = spins[i] * boltzmann_distribution(i, spins[i], spins) # + spins[i] * boltzmann_distribution(i, -spins[i], spins)
	return m

# Partition function
def partition_function(spins):
	Z = 0
	for i in range(0, len(spins)):
		Z += math.exp(-beta * hamiltonian(spins)) + math.exp(beta * hamiltonian(spins))
	# Z = 2 * math.pow(2 * np.cosh(beta * J), (len(spins) - 1))	
	return Z

# Spin average
def spin_average(s0, spins):
	m = 0
	for i in range(0, len(spins)):
		m += (s0 * math.exp(-beta * hamiltonian(spins)))/partition_function(spins)
	return m

def specific_heat(spin_avg):
	z = n_spin - 1
	c = math.pow(beta, 2) * math.pow(z, 2) * math.pow(J, 2) * (1 -  math.pow(spin_avg, 2)) * math.pow(spin_avg, 2) / (1 - z * J * beta * (1 - math.pow(spin_avg, 2)))
	return c

def main():
	global beta
	spins = create_spins(n_spin)
	print(spins)
	normal = 0
	mfa = 0
	bethe = 0

	for i in range(0, n_spin):
		print("m", i)
	# 	neighbours = spins[:i] + spins[i+1 :]
		si_norm = one_spin_average(i, spins)
		normal += si_norm
		print("Normal:", si_norm)

		si_mfa = one_mfa_spin_average(i, spins)
		mfa += si_mfa
		print("MFA:", si_mfa)

		si_bethe = one_bethe_spin_average(i, spins)
		bethe += si_bethe
		print("Bethe:", si_bethe)

	print("")
	print("Tanh:", mfa_tanh_spin_average(spins))
	print("Norm:", normal/n_spin)
	print("Avg tanh:", mfa/n_spin)
	print("Avg bethe:", bethe/n_spin)

	print("")
	arr_temp = np.arange(3, 30, 0.5)
	arr_tanh_mfa = []
	arr_norm = []
	arr_mfa = []
	arr_bethe = []

	for idx, t in enumerate(arr_temp):
		beta = 1/t
		normal = 0
		mfa = 0
		bethe = 0
		for i in range(0, n_spin):
			si_norm = one_spin_average(i, spins)
			normal += si_norm
			
			si_mfa = one_mfa_spin_average(i, spins)
			mfa += si_mfa

			si_bethe = one_bethe_spin_average(i, spins)
			bethe += si_bethe
		arr_norm.append(normal/n_spin)
		arr_mfa.append(mfa/n_spin)
		arr_tanh_mfa.append(mfa_tanh_spin_average(spins))
		arr_bethe.append(bethe/n_spin)

		# arr_norm.append(specific_heat(normal/n_spin))
		# arr_mfa.append(specific_heat(mfa/n_spin))
		# arr_tanh_mfa.append(specific_heat(mfa_tanh_spin_average(spins)))
		# arr_bethe.append(specific_heat(bethe/n_spin))

	

	print("TANH MFA")
	plt.plot(arr_temp, arr_tanh_mfa)
	plt.xlabel('Temperature')
	plt.ylabel('Specific Heat')
	plt.show()

	print("NORMAL")
	plt.plot(arr_temp, arr_norm)
	plt.xlabel('Temperature')
	plt.ylabel('Specific Heat')
	plt.show()

	# print("Normal")
	# plt.plot(arr_temp, arr_norm, arr_mfa, arr_tanh_mfa)
	# plt.xlabel('Temperature')
	# plt.ylabel('Specific Heat')
	# plt.show()

	print("MFA")
	plt.plot(arr_temp, arr_mfa)
	plt.xlabel('Temperature')
	plt.ylabel('Specific Heat')
	plt.show()

	print("Bethe")
	plt.plot(arr_temp, arr_bethe)
	plt.xlabel('Temperature')
	plt.ylabel('Specific Heat')
	plt.show()


if __name__ == "__main__":
	main()