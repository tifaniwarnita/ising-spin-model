import matplotlib.pyplot as plt
import numpy as np
import os
import random
from scipy.optimize import fsolve

# Constants
VAL_J = 1
VAL_h = 0.5
VAL_b = 1
VAL_n_spin = 5
VAL_spin_value = [-1, 1]
VAL_iteration = 15

# Spins
def create_spins(n_spin):
	S = [None] * n_spin
	for i in range (0, n_spin):
		S[i] = random.choice(VAL_spin_value)
	return S

# (a) NAIVE MEAN FIELD APPROXIMATION
# Basic idea of mean-field theory is to set the fluctuations into zero.
def spin_avg_mfa(spins, b, J, h):
	spin_avg = spins
	spin_avg[0] = np.tanh(b * (J * spin_avg[1] + h))
	spin_avg[1] = np.tanh(b * (J * spin_avg[0] + J * spin_avg[2] + h))
	spin_avg[2] = np.tanh(b * (J * spin_avg[1] + J * spin_avg[3] + h))
	spin_avg[3] = np.tanh(b * (J * spin_avg[2] + J * spin_avg[4] + h))
	spin_avg[4] = np.tanh(b * (J * spin_avg[3] + h))
	return spin_avg

def main():
	spins = create_spins(VAL_n_spin)
	filename = "MFA_" + str(spins) + '.png'
	description = "[s1, s2, s3, s4, s5] = " + str(spins)

	s1 = []
	s2 = []
	s3 = []
	s4 = []
	s5 = []
	
	for i in np.arange(VAL_iteration):
		spin_avg = spin_avg_mfa(spins, VAL_b, VAL_J, VAL_h)
		s1.append(spin_avg[0])
		s2.append(spin_avg[1])
		s3.append(spin_avg[2])
		s4.append(spin_avg[3])
		s5.append(spin_avg[4])

	line_s1 = plt.plot(np.arange(VAL_iteration), s1, label='<s1>')
	line_s2 = plt.plot(np.arange(VAL_iteration), s2, label='<s2>')
	line_s3 = plt.plot(np.arange(VAL_iteration), s3, label='<s3>')
	line_s4 = plt.plot(np.arange(VAL_iteration), s4, label='<s4>')
	line_s5 = plt.plot(np.arange(VAL_iteration), s5, label='<s5>')

	plt.legend(loc='lower right', shadow=True)
	plt.gca().set_position((.13, .3, .8, .6))

	description += "\n\nResult when the spin averages are convergent:"
	description += "\n<s1> = " + str(spin_avg[0])
	description += ", <s2> = " + str(spin_avg[1])
	description += ", <s3> = " + str(spin_avg[2])
	description += "\n<s4> = " + str(spin_avg[3])
	description += ", <s5> = " + str(spin_avg[4])
	plt.figtext(.02, .02, description)

	plt.title('Naive Mean Field Approximation')
	plt.xlabel("Iteration")
	plt.ylabel("Spin Average")
	plt.grid()

	directory = "fig"
	if not os.path.exists(directory):
		os.makedirs(directory)
	plt.savefig(directory + "/" + filename)
	plt.show()

if __name__ == "__main__":
	main()