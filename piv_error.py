"""
PIV ERROR ANALYSIS FUNCTION


Cameron Dallas
University of Toronto
Department of Mechanical and Industrial Engineering
Turbulence Research Laboratory
"""


import numpy as np
import scipy.ndimage.filters as filt




def correlation_stats(frame_a, frame_b, x, y, u, v, dt, L, overlap, scaling_factor = None, dx = 1, dy = 1):

	"""
	Uses the Correlation Statistics method to approximate the error in the velocity
	arising from the image pair correlation algorithm. 
	See paper 'PIV uncertainty quantification from correlation statistics'. 2015

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
		L: int
			Interrogation window size
		overlap: 
			number of pixels that adjacent windows overlap
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


	#Make sure inputs make sense
	if frame_a.shape != frame_b.shape:
		raise ValueError('Image pair must be the same shape') 
	if not(x.shape == y.shape == u.shape == v.shape):
		raise ValueError('vector shapes and locations must be the same size')

	#normalize images/ convert to float
	frame_a = frame_a.astype(float)
	frame_b = frame_b.astype(float)

	#get field shape
	n_rows, n_cols = frame_a.shape

	#Calculate displacement field and vector locations.
	#Must all be integer values because they represent pixel locations
	if scaling_factor:
		u_row_temp = np.round(-v*dt*scaling_factor).astype('int32')
		u_col_temp = np.round(u*dt*scaling_factor).astype('int32')

		row_pix = np.round(y*scaling_factor).astype('uint32')
		col_pix = np.round(x*scaling_factor).astype('uint32')
	else:
		u_row_temp = np.round(-v*dt).astype('int32') 	
		u_col_temp = np.round(u*dt).astype('int32') 	

		row_pix = np.round(y).astype('uint32')
		col_pix = np.round(x).astype('uint32')


	#expand displacement field to full image
	u_row = np.zeros(frame_a.shape, dtype = np.int32)   #row shift at each pixel location
	u_col = np.zeros(frame_a.shape, dtype = np.int32)	#column shift at each pixel location

	for i in range(v.shape[0]):
		for j in range(v.shape[1]):
			u_row[overlap/2 + i*overlap: overlap/2 + (i+1)*overlap, overlap/2 + j*overlap: overlap/2 + (j+1)*overlap] = u_row_temp[i,j]
			u_col[overlap/2 + i*overlap: overlap/2 + (i+1)*overlap, overlap/2 + j*overlap: overlap/2 + (j+1)*overlap] = u_col_temp[i,j]


	#shift frame_b to match frame a
	frame_b_shift = np.zeros(frame_b.shape)
	for i in range(overlap/2, (n_rows - overlap/2)):
		for j in range(overlap/2, n_cols - overlap/2):
			frame_b_shift[i,j] = frame_b[i- u_row[i,j], j - u_col[i,j]]


	"""
	-----------------  x uncertainty -----------------------
	"""

	#calculate C(u), C+, C-, and S_xy field for x
	c_u = np.multiply(frame_a, frame_b_shift)  #this value is the same for both x and y uncertainty
	S_x = np.zeros(frame_a.shape)
	dC_x = np.zeros(frame_a.shape)  #delta C_i in x direction
	dC_x[0:-dx, 0:-dx] = frame_a[0:-dx,0:-dx]*frame_b_shift[0:-dx, dx:] - frame_a[0:-dx, dx:]*frame_b_shift[0:-dx, 0:-dx]
	dC_x = dC_x - np.mean(dC_x) #remove the mean


	for i in range(overlap/2, n_rows - overlap/2 -1):
		for j in range(overlap/2, n_cols - overlap/2 -1):

			S0 = dC_x[i,j]**2
			for k in range(1,4):
				if dC_x[i, j]*dC_x[i+k,j+k]/S0 < 0.05:
					S_x[i,j] = np.sum(dC_x[i, j]*dC_x[i:i+k, j:j+k])
					break


	#get shifted correlation function fields
	cPlus_x = np.zeros(frame_a.shape)
	cPlus_x[:, 0:-1] = c_u[:,1:]
	cMinus_x = np.zeros(frame_a.shape)
	cMinus_x[:, 1:] = c_u[:,0:-1]

	#smooth and sum the fields
	c_filt = (L**2) * filt.gaussian_filter(c_u, L)[row_pix, col_pix]
	cPlus_filt_x = (L**2) * filt.gaussian_filter(cPlus_x, L)[row_pix, col_pix]
	cMinus_filt_x = (L**2) * filt.gaussian_filter(cMinus_x, L)[row_pix, col_pix]
	S_filt_x = (L**2) * filt.gaussian_filter(S_x, L)[row_pix, col_pix]

	#solve for final uncertainty
	sig_x = np.sqrt(np.abs(S_filt_x))  #put in abs to avoid runtime warnings
	Ux = np.zeros(x.shape) 
	cpm_x = (cPlus_filt_x + cMinus_filt_x) / 2. 

	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			Ux[i,j] = ((np.log(np.abs(cpm_x[i,j]+sig_x[i,j]/2.)) - np.log(np.abs(cpm_x[i,j] - sig_x[i,j]/2.))) / 
		(4*np.log(c_filt[i,j]) - 2*np.log(np.abs(cpm_x[i,j] + sig_x[i,j]/2.)) - 2*np.log(np.abs(cpm_x[i,j] - sig_x[i,j]/2.))))


	"""
	-----------------  y uncertainty -----------------------
	"""

	#calculate C(u), C+, C-, and S_xy field for y
	S_y = np.zeros(frame_a.shape)
	dC_y = np.zeros(frame_a.shape)  #delta C_i in x direction
	dC_y[0:-dy, 0:-dy] = frame_a[0:-dy, 0:-dy]*frame_b_shift[dy:, 0:-dy] - frame_a[dy:, 0:-dy]*frame_b_shift[0:-dy, 0:-dy]
	dC_y = dC_y - np.mean(dC_y) #remove the mean

	for i in range(overlap/2, n_rows - overlap/2 -1):
		for j in range(overlap/2, n_cols - overlap/2 -1):

			S0 = dC_y[i,j]**2
			for k in range(1,5):
				if dC_y[i, j]*dC_y[i+k,j+k]/S0 < 0.05:
					S_y[i,j] =  np.sum(dC_y[i, j]*dC_y[i:i+k, j:j+k])


	#get shifted correlation function fields
	cPlus_y = np.zeros(frame_a.shape)
	cPlus_y[0:-1,:] = c_u[1: , :]
 	cMinus_y = np.zeros(frame_a.shape)
 	cMinus_y[1:, :] = c_u[0:-1, :] 


	#smooth the fields and sum the fields
	cPlus_filt_y = (L**2) * filt.gaussian_filter(cPlus_y, L)[row_pix, col_pix]
	cMinus_filt_y = (L**2) * filt.gaussian_filter(cMinus_y, L)[row_pix, col_pix]
	S_filt_y = (L**2) * filt.gaussian_filter(S_y, L)[row_pix, col_pix]

	#calculate standard deviation  of correlation difference
	sig_y = np.sqrt(np.abs(S_filt_y))
	Uy = np.zeros(x.shape) 
	cpm_y = (cPlus_filt_y + cMinus_filt_y) / 2. 

	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			Uy[i,j] = ((np.log(cpm_y[i,j]+sig_y[i,j]/2.) - np.log(np.abs(cpm_y[i,j] - sig_y[i,j]/2.))) / 
		(4*np.log(c_filt[i,j]) - 2*np.log(np.abs(cpm_y[i,j] + sig_y[i,j]/2.)) - 2*np.log(np.abs(cpm_y[i,j] - sig_y[i,j]/2.))))
	

	#propogae uncertainty to the velocity field
	#Ux = Ux/dt.scaling_factor
	#Uy = Uy/dt.scaling_factor

	return Ux, Uy




def image_dewarp(frame_b, u,v,dt, scaling_factor):
	"""
	Dewarp the second image back onto the first using the
	displacement field and a high order sub-pixel interpolation
	scheme.

	Inputs:
		frame_b: 2d array
			second image of the PIV image pair
		u,v: 2d array 
			u and v velocity calculated by PIV algorithm

	"""

	frame_b_shift = np.zeros(frame_b.shape)


	return(frame_b_shift)




#
#----------------------------------------------------------------------------------------------------
#
# 
# This is supposed to be a fast implimentation of a recursive gaussian filter,
# but it is way slower than the scipy gaussian filter. Will keep this because it
# may be usefull for GPU acceleration. 



# def recursiveGaussSmooth(data, L):

# 	"""
# 	Fast implimentation of a recursive gaussian smoother. 
# 	For reference, see paper 'Tips & Tricks: Fast Image Filtering Algorithms (2007)'

# 	Inputs:
# 		data: 2d array
# 			array that you want smoothed
# 		L: int
# 			effective interrogation window size

# 	Outputs:
# 		filter: 2d array
# 			smoothed array multiplied by square of the filter length

# 	General idea: The filtering acts as a fast way to do a weighted sum of the values
# 	in each interrogation window. The output should produce values of the Cs and S at
# 	each vector point in the image.
# 	"""

	
# 	sig = L  #filter radius
# 	n_rows, n_cols = data.shape
# 	output = np.zeros(data.shape)  #array for output data

# 	#get q value
# 	if sig >= 0.5 and sig <2.5:
# 		q = 3.97156 - 4.14554*np.sqrt(1 - 0.26891*sig)
# 	if sig >= 2.5:
# 		q = 0.98711*sig - 0.96330
# 	else:
# 		raise ValueError('Filter radius too small')	


# 	#get filter coefficients
# 	b0 = 1.57825 + 2.44413*q + 1.4281*(q**2) + 0.422205*(q**3)
# 	b1 = 2.44413*q + 2.85619*(q**2) + 1.26661*(q**3)
# 	b2 = -1.4281*(q**2) - 1.26661*(q**3)
# 	b3 = 0.422205*(q**3)
# 	B = 1 - (b1 + b2 + b3)/b0

# 	#define recursive gauss filter
# 	def rg_filter(x):

# 		#x is the signal to be filtered
# 		y = np.zeros(len(x))
# 		y[0] = B*x[0]
# 		y[1] = B*x[1] + (b1*y[0]) / b0
# 		y[2] = B*x[2] + (b1*y[1] + b2*y[0]) / b0
# 		y[3] = B*x[3] + (b1*y[2] + b2*y[1] + b3*y[0]) / b0

# 		for i in range(4, len(x)):
# 			y[i] = B*x[i] + (b1*y[i-1] + b2*y[i-2] + b3*y[i-3]) / b0

# 		return y


# 	#aplly filter backwards and forwards to each row of data

# 	#apply filter in x direction
# 	for i in range(n_rows):

# 		y_temp = rg_filter(data[i,:])
# 		output[i,:] = rg_filter(y_temp[::-1])[::-1]

# 	#apply filter in y direction
# 	for i in range(n_cols):

# 		o_temp = rg_filter(output[:,i])
# 		output[:,i] = rg_filter(o_temp[::-1])[::-1]

# 	return output

# '''