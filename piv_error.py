"""
PIV ERROR ANALYSIS FUNCTION


Cameron Dallas
University of Toronto
Department of Mechanical and Industrial Engineering
Turbulence Research Laboratory
"""


import numpy as np


def correlation_stats(c, i_peak, j_peak, u_row, u_col, im1, im2):

	"""
	Uses the Correlation Statistics method to approximate the error in the velocity
	arising from the image pair correlation algorithm. 
	See paper 'Generic a-posteriori uncertainty quantification for PIV vector fields 
	by correlation statistics. 2014'

	Inputs:
		c = correlation map matrix
		i_peak = row where c is max
		j_peak = column where c is max
		u_row = row image shift (from correlation map)
		u_col = coloum image shift (from correlation map)
		im1 = first frame IW
		im2 = second frame IW

	Outputs:
		Ux = x uncertainty in pixels
		Uy = y uncertainty in pixels

	"""

	#get integer values for correlation max
	i_peak = int(round(i_peak))
	j_peak = int(round(j_peak))

	#dewarp images onto each other



	#correlation map maximum
	c0 = c[i_peak, j_peak]

	#x uncertiainty
	c1_x = (c[i_peak, j_peak + 1] + c[i_peak, j_peak - 1])/2. - sig_x/2.
	c2_x = (c[i_peak, j_peak + 1] + c[i_peak, j_peak - 1])/2. + sig_x/2.
	Ux = 0.5*(np.log(c2_x) - np.log(c1_x))/(2*np.log(c0) - np.log(c2_x) - np.log(c1_x))

	#y uncertiainty
	c1_y = (c[i_peak + 1, j_peak] + c[i_peak - 1, j_peak])/2. - sig_y/2.
	c2_y = (c[i_peak + 1, j_peak] + c[i_peak - 1, j_peak])/2. + sig_y/2.
	Uy = 0.5*(np.log(c2_y) - np.log(c1_y))/(2*np.log(c0) - np.log(c2_x) - np.log(c1_x))

	return Ux, Uy

