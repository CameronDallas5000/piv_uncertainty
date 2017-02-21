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
			effective interrogation window size

	Outputs:
		filter: 2d array
			smoothed array multiplied by square of the filter length

	General idea: The filtering acts as a fast way to do a weighted sum of the values
	in each interrogation window. The output should produce values of the Cs and S at
	each vector point in the image.
	"""

	
	sig = L  #filter radius
	n_rows, n_cols = data.shape
	output = np.zeros(data.shape)  #array for output data

	#get q value
	if sig >= 0.5 and sig <2.5:
		q = 3.97156 - 4.14554*np.sqrt(1 - 0.26891*sig)
	if sig >= 2.5:
		q = 0.98711*sig - 0.96330
	else:
		raise ValueError('Filter radius too small')	


	#get filter coefficients
	b0 = 1.57825 + 2.44413*q + 1.4281*(q**2) + 0.422205*(q**3)
	b1 = 2.44413*q + 2.85619*(q**2) + 1.26661*(q**3)
	b2 = -1.4281*(q**2) - 1.26661*(q**3)
	b3 = 0.422205*(q**3)
	B = 1 - (b1 + b2 + b3)/b0

	#define recursive gauss filter
	def rg_filter(x):

		#x is the signal to be filtered
		y = np.array(x.shape)
		y[0] = B*x[0]
		y[1] = B*x[1] + (b1*y[0]) / b0
		y[2] = B*x[2] + (b1*y[1] + b2*y[0]) / b0
		y[3] = B*x[3] + (b1*y[2] + b2*y[1] + b3*y[0]) / b0

		for i in range(4, len(x)):
			y[i] = B*x[i] + (b1*y[i-1] + b2*y[i-2] + b3*y[i-3]) / b0

		return y


	#aplly filter backwards and forwards to each row of data

	#apply filter in x direction
	for i in range(n_rows):

		y_temp = rg_filter(data[i,:])
		output[i,:] = rg_filter(y_temp[::-1])[::-1]

	#apply filter in y direction
	for i in range(n_cols):

		o_temp = rg_filter(output[:,i])
		output[:,i] = rg_filter(o_temp[::-1])[::-1]

return output


def correlation_stats(frame_a, frame_b, x, y, u, v, dt, scaling_factor = None, dx = 1, dy = 1):

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
		dx: int
			pixels displacement for the delta x value (typically should be 1)
		dy: int
			pixel displacement for the delta y value (typically should be 1)

		  
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

		#locations in pixels
		row_pix = int(np.round(y*scaling_factor))
		col_pix = int(np.round(x*scaling_factor))

	#to help with multiplying windows near boundary
	rowShift_max = np.max(u_row) #probably a better way to do this
	colShift_max = np.max(u_col)

	"""
	-----------------  x uncertainty -----------------------
	"""

	#calculate C(u), C+, C-, and S_xy field for x
	c_u = np.zeros(frame_a.shape)  #this value is the same for both x and y uncertainty
	cPlus_x = np.zeros(frame_a.shape)
	cMinus_x = np.zeros(frame_a.shape)
	S_x = np.zeros(frame_a.shape)

	for i in range(n_rows - rowShift_max):
		for j in range(n_cols - colShift_max):
			c_u[i,j] = frame_a[i,j]*frame_b[i-u_row[i,j], j-u_col[i,j]]
			cPlus_x[i,j] = frame_a[i,j]*frame_b[i-u_row[i,j], j-u_col[i,j] + dx]
			cMinus_x[i,j] = frame_a[i,j]*frame_b[i-u_row[i,j], j-u_col[i,j] - dx]

			#calculate S
			for k in range(5):
				for l in range(5):
					C_xy = frame_a[i, j]*frame_b[i-u_row[i,j], j-u_col[i,j]+dx] - frame_a[i, j+dx]*frame_b[i-u_row[i,j], j-u_col[i,j]]  
					C_dxdy = frame_a[i+k, j+l]*frame_b[i+k-u_row[i,j], j+l-u_col[i,j]+dx] - frame_a[i+k, j+l+dx]*frame_b[i+k-u_row[i,j], j+l-u_col[i,j]]  
					S_x[i,j] += C_xy*C_dxdy


	#smooth and sum the fields
	c_filt = (L**2) * recursiveGaussSmooth(c_u, L)[row_pix, col_pix]
	cPlus_filt_x = (L**2) * recursiveGaussSmooth(cPlus_x, L)[row_pix, col_pix]
	cMinus_filt_x = (L**2) * recursiveGaussSmooth(cMinus_x, L)[row_pix, col_pix]
	S_filt_x = (L**2) * recursiveGaussSmooth(S_x, L)[row_pix, col_pix]

	#calculate standard deviation  of correlation difference
	sig_x = np.sqrt(S_filt_x)
	Ux = np.zeros(x.shape) 
	cpm_x = (cPlus_filt_x + cMinus_filt_x) / 2. 

	for i in range(x.shape[0]):
		for j in range(x.shape[1])
			Ux[i,j] = ((np.log(cpm_x[i,j]+sig_x[i,j]/2.) - np.log(cpm_x[i,j] - sig_x[i,j]/2.)) / 
		(4*np.log(c_filt[i,j]) - 2*np.log(cpm_x[i,j] + sig_x[i,j]/2.) - 2*np.log(cpm_x[i,j] - sig_x[i,j]/2.)))


	"""
	-----------------  y uncertainty -----------------------
	"""

	#calculate C(u), C+, C-, and S_xy field for y
	cPlus_y = np.zeros(frame_a.shape)
	cMinus_y = np.zeros(frame_a.shape)
	S_y = np.zeros(frame_a.shape)

	for i in range(n_rows - rowShift_max):
		for j in range(n_cols - colShift_max):
			cPlus_y[i,j] = frame_a[i,j]*frame_b[i - u_row[i,j] + dy, j-u_col[i,j]]
			cMinus_x[i,j] = frame_a[i,j]*frame_b[i - u_row[i,j] - dy, j-u_col[i,j]]

			#calculate S
			for k in range(5):
				for l in range(5):
					C_xy = frame_a[i, j]*frame_b[i-u_row[i,j]+dy, j-u_col[i,j]] - frame_a[i+dy, j]*frame_b[i-u_row[i,j], j-u_col[i,j]]  
					C_dxdy = frame_a[i+k, j+l]*frame_b[i+k-u_row[i,j]+dy, j+l-u_col[i,j]] - frame_a[i+k +dy, j+l]*frame_b[i+k-u_row[i,j], j+l-u_col[i,j]]  
					S_y[i,j] += C_xy*C_dxdy


	#smooth the fields and sum the fields
	cPlus_filt_y = (L**2) * recursiveGaussSmooth(cPlus_y, L)[row_pix, col_pix]
	cMinus_filt_y = (L**2) * recursiveGaussSmooth(cMinus_y, L)[row_pix, col_pix]
	S_filt_y = (L**2) * recursiveGaussSmooth(S_y, L)[row_pix, col_pix]

	#calculate standard deviation  of correlation difference
	sig_y = np.sqrt(S_filt_y)
	Uy = np.zeros(x.shape) 
	cpm_y = (cPlus_filt_y + cMinus_filt_y) / 2. 

	for i in range(x.shape[0]):
		for j in range(x.shape[1])
			Uy[i,j] = ((np.log(cpm_y[i,j]+sig_y[i,j]/2.) - np.log(cpm_y[i,j] - sig_y[i,j]/2.)) / 
		(4*np.log(c_filt[i,j]) - 2*np.log(cpm_y[i,j] + sig_y[i,j]/2.) - 2*np.log(cpm_y[i,j] - sig_y[i,j]/2.)))
	

	return Ux, Uy


