"""
PIV ERROR ANALYSIS FUNCTION


Cameron Dallas
University of Toronto
Department of Mechanical and Industrial Engineering
Turbulence Research Laboratory
"""


import numpy as np

def recursiveGaussSmooth(data, L):

	"""
	Fast imlimentation of a recursive gaussian smoother. 
	For reference, see paper 'Tips & Tricks: Fast Image Filtering Algorithms (2007)'

	Inputs:
		data: 2d array
			array that you want smoothed
		L: int
			filter length (effective interrogation window size)

	Outputs:
		filter: 2d array
			smoothed array multiplied by square of the filter length
	"""

	


def correlation_stats(frame_a, frame_b, x, y, u, v, dt, scaling_factor = None):

	"""
	Uses the Correlation Statistics method to approximate the error in the velocity
	arising from the image pair correlation algorithm. 
	See paper 'Generic a-posteriori uncertainty quantification for PIV vector fields 
	by correlation statistics. 2014'

	Inputs:
		frame_a: 2d array
			first full frame
		frame_b: 2d array
			second full frame 
		x: 2d array 
			x location of velocity vectors
		y: 2d array
			location of velocity vectors
		u: 2d array 
			x velocity
		v: 2d array 
			y velocity
		dt: float  
			time step between windows
		scaling factor: int 
			image scaling factor in pixels per meter. By default this assumes that all inputs for 
			velocity and location are in units of pixels and pixels per second
		  
	Outputs:
		Ux: 2d array
			x velocity uncertainty in m/s
		Uy: 2darray 
			y velocity uncertainty in m/s

	""" 


	#ensure frames are the same shape
	if frame_a.shape != frame_b.shape:
		raise ValueError('Image pair must be the same shape') 

	#get field shape
	n_rows, n_cols = frame_a.shape

	#Calculate Displacement field. 
	#ie local shift of the interrogation window in pixels at each vector location
	u_row = v*dt 	#row shift field
	u_col = u*d 	#column shift field

	#scale the displacement field if needed
	if scaling_factor:
		u_row = u_row*scaling_factor
		v_row = v_row*scaling_factor

	#to help with multiplying windows near boundary
	rowShift_max = np.max(u_row) #probably a better way to do this
	colShift_max = np.max(u_col)

	"""
	-----------------  x uncertainty -----------------------
	"""

	#calculate C(u), C+, C-, and S_xy field for x
	c_x = np.zeros(frame_a.shape)
	cPlus_x = np.zeros(frame_a.shape)
	cMinus_x = np.zeros(frame_a.shape)
	S_x = np.zeros(frame_a.shape)

	for i in range(n_rows - rowShift_max):
		for j in range(n_cols - colShift_max):
			c_x[i,j] = frame_a[i,j]*frame_b[i-u_row[i,j], j-u_col[i,j]]
			cPlus_x[i,j] = frame_a[i,j]*frame_b[i-u_row[i,j], j-u_col[i,j]+1]
			cMinus_x[i,j] = frame_a[i,j]*frame_b[i-u_row[i,j], j-u_col[i,j]-1]

			#calculate S
			for k in range(5):
				for l in range(5):
					C_xy = frame_a[i, j]*frame_b[i-u_row[i,j], j-u_col[i,j]+1] - frame_a[i, j+1]*frame_b[i-u_row[i,j], j-u_col[i,j]]  
					C_dxdy = frame_a[i+k, j+l]*frame_b[i+k-u_row[i,j], j+l-u_col[i,j]+1] - frame_a[i+k, j+l+1]*frame_b[i+k-u_row[i,j], j+l-u_col[i,j]]  
					S_x[i,j] += C_xy*C_dxdy



	"""
	-----------------  y uncertainty -----------------------
	"""


	return Ux, Uy

