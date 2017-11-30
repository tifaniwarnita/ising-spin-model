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

# (b) CAVITY METHOD
# Basic idea of the cavity method is introducing a cavity in the model
# The nearest interactions are handled exactly
def spin_avg_cavity(spins, b, J, h):
	spin_avg = spins
	
	# forward
	tetha1to12 = b * h
	tetha12to2 = np.arctanh(np.tanh(tetha1to12) * np.tanh(b * J))
	tetha2to23 = b * h + tetha12to2
	tetha23to3 = np.arctanh(np.tanh(tetha2to23) * np.tanh(b * J))
	tetha3to34 = b * h + tetha23to3
	tetha34to4 = np.arctanh(np.tanh(tetha3to34) * np.tanh(b * J))
	tetha4to45 = b * h + tetha34to4
	tetha45to5 = np.arctanh(np.tanh(tetha4to45) * np.tanh(b * J))

	# backward
	tetha5to45 = b * h
	tetha45to4 = np.arctanh(np.tanh(tetha5to45) * np.tanh(b * J))
	tetha4to34 = b * h + tetha45to4
	tetha34to3 = np.arctanh(np.tanh(tetha4to34) * np.tanh(b * J))
	tetha3to23 = b * h + tetha34to3
	tetha23to2 = np.arctanh(np.tanh(tetha3to23) * np.tanh(b * J))
	tetha2to12 = b * h + tetha23to2
	tetha12to1 = np.arctanh(np.tanh(tetha2to12) * np.tanh(b * J))

	spin_avg[0] = np.tanh(b * h + tetha12to1)
	spin_avg[1] = np.tanh(b * h + tetha12to2 + tetha23to2)
	spin_avg[2] = np.tanh(b * h + tetha23to3 + tetha34to3)
	spin_avg[3] = np.tanh(b * h + tetha34to4 + tetha45to4)
	spin_avg[4] = np.tanh(b * h + tetha45to5)
	return spin_avg

def main():
	spins = create_spins(VAL_n_spin)
	filename = "MFA_" + str(spins) + '.png'
	x_label = ['s1', 's2', 's3', 's4', 's5']
	description = "[s1, s2, s3, s4, s5] = " + str(spins)

	s1 = []
	s2 = []
	s3 = []
	s4 = []
	s5 = []
	
	spin_avg = spin_avg_cavity(spins, VAL_b, VAL_J, VAL_h)
	line_s1 = plt.plot(x_label, spin_avg)
	
	plt.legend(loc='lower right', shadow=True)
	plt.gca().set_position((.13, .3, .8, .6))

	description += "\n\nResult:"
	description += "\n<s1> = " + str(spin_avg[0])
	description += ", <s2> = " + str(spin_avg[1])
	description += ", <s3> = " + str(spin_avg[2])
	description += "\n<s4> = " + str(spin_avg[3])
	description += ", <s5> = " + str(spin_avg[4])
	plt.figtext(.02, .02, description)

	plt.title('Cavity Method')
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