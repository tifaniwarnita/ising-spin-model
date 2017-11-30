from cavity import *
from mfa import *
import math

# (c) EXACT SOLUTION
# Hamiltonian
def hamiltonian_i(i, value, spins, b, J, h):
	a = 0
	if i > 0:
		a += spins[i-1]
	if i < len(spins) - 1:
		a += spins[i+1]

	H = (-J * value * a) - (h * value)
	return H

def partition_function_i(i, spins, b, J, h):
	Z = math.exp(b * hamiltonian_i(i, spins[i], spins, b, J, h)) + math.exp(b * hamiltonian_i(i, -spins[i], spins, b, J, h))
	return Z

def boltzmann_distribution(i, value, spins, b, J, h):
	d = math.exp(b * hamiltonian_i(i, value, spins, b, J, h)) / partition_function_i(i, spins, b, J, h)
	return d

def spin_avg_i(i, spins, b, J, h):
	m = spins[i] * boltzmann_distribution(i, spins[i], spins, b, J, h) + spins[i] * boltzmann_distribution(i, -spins[i], spins, b, J, h)
	return abs(m)
		
def main():
	spins = create_spins(VAL_n_spin)
	filename = "Comparison_" + str(spins) + '.png'
	x_label = ['s1', 's2', 's3', 's4', 's5']

	exact_spin_avg = []
	for i, s in enumerate(spins):
		exact_spin_avg.append(spin_avg_i(i, spins[:], VAL_b, VAL_J, VAL_h))

	mfa_spin_avg = spins[:]
	for i in np.arange(VAL_iteration):
		mfa_spin_avg = spin_avg_mfa(mfa_spin_avg, VAL_b, VAL_J, VAL_h)
	
	cavity_spin_avg = spin_avg_cavity(spins, VAL_b, VAL_J, VAL_h)

	print("Exact  :", exact_spin_avg)
	print("MFA    :", mfa_spin_avg)
	print("cavity :", cavity_spin_avg)

	line_s1 = plt.plot(x_label, exact_spin_avg, label='Exact')
	line_s2 = plt.plot(x_label, mfa_spin_avg, label='MFA')
	line_s3 = plt.plot(x_label, cavity_spin_avg, label='Cavity')

	plt.legend(loc='lower right', shadow=True)

	plt.title('Comparison')
	plt.xlabel("Spin")
	plt.ylabel("Spin Average")
	plt.grid()

	directory = "fig"
	if not os.path.exists(directory):
		os.makedirs(directory)
	plt.savefig(directory + "/" + filename)
	plt.show()

if __name__ == "__main__":
	main()