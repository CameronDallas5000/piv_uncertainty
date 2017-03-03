"""
PIV ERROR ANALYSIS FUNCTION


Cameron Dallas
University of Toronto
Department of Mechanical and Industrial Engineering
Turbulence Research Laboratory
"""


import numpy as np
import scipy.ndimage.filters as filt
import scipy.interpolate as interp




def correlation_stats(frame_a, frame_b, x, y, u, v, dt, L, overlap, scaling_factor = 1., dx = 1, dy = 1):

	"""
	Uses the Correlation Statistics method to approximate the error in the velocity
	arising from the image pair correlation algorithm. 
	See paper 'PIV uncertainty quantification from correlation statistics'. 2015

	Inputs:
		frame_a: 2d array
			first full frame
		frame_b: 2d array
			second full frame 
		x: 2d array , float
			x location of velocity vectors
		y: 2d array, float
			location of velocity vectors
		u: 2d array , float
			x velocity
		v: 2d array , float
			y velocity
		dt: float  
			time step between windows
		L: int
			Interrogation window size
		overlap: int
			number of pixels that adjacent windows overlap
		scaling factor: float
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

	#scale piv data
	x = x*scaling_factor
	y = y*scaling_factor
	u = u*scaling_factor
	v = v*scaling_factor

	row_pix = y[::-1].astype('int32')
	col_pix = x.astype('int32')

	#Dewarp frame_b
	frame_b_shift = image_dewarp(frame_b, x,y,u,v)

	"""
	-----------------  x uncertainty -----------------------

	Get C and Sxy values for each pixel then filter and smooth them, which is
	equivalent to doing the sms over the interrogation window.
	"""


	#calculate C(u)
	C_u = np.multiply(frame_a, frame_b_shift)  #this value is the same for both x and y uncertainty
	
	#get delta C_xy for x direction
	dC_x = np.zeros(frame_a.shape)  #delta C_i in x direction
	cPlus_x = np.zeros(frame_a.shape)
	cPlus_x[:, 0:-dx] = frame_a[:, 0:-dx]*frame_b_shift[:, dx:]
	cMinus_x = np.zeros(frame_a.shape)
	cMinus_x[:, 0:-dx] = frame_a[:, dx:]*frame_b_shift[:, 0:-dx]
	dC_x = cPlus_x - cMinus_x
	dC_x = dC_x - np.mean(dC_x) #remove the mean

	#smooth dC_x in the x direction
	for i in range(dC_x.shape[0]):
		for j in range(1, dC_x.shape[1]-1):
			dC_x[i, j] = (dC_x[i, j-1] + 2*dC_x[i, j] + dC_x[i, j+1])/4.

	#calculate covariance matrix (S_x)
	S_x = np.zeros(frame_a.shape)
	for i in range(frame_a.shape[0]):
		for j in range(frame_a.shape[1]):

			S0 = dC_x[i,j]**2
			for k in range(1,4):
				try:
					if dC_x[i, j]*dC_x[i+k,j+k]/S0 < 0.05:
						S_x[i,j] = np.sum(dC_x[i, j]*dC_x[i:i+k, j:j+k])
						break
					if k == 3 :
						S_x[i,j] = np.sum(dC_x[i, j]*dC_x[i:i+k, j:j+k])
				except IndexError:
					S_x[i,j] = 0.


	#smooth and sum the fields
	C_filt = (L**2) * filt.gaussian_filter(C_u, L)[row_pix, col_pix]
	cPlus_filt_x = (L**2) * filt.gaussian_filter(cPlus_x, L)[row_pix, col_pix]
	cMinus_filt_x = (L**2) * filt.gaussian_filter(cMinus_x, L)[row_pix, col_pix]
	S_x_filt = (L**2) * filt.gaussian_filter(S_x, L)[row_pix, col_pix]

	sig_x = np.sqrt(S_x_filt)  #put in abs to avoid runtime warnings
	Ux = np.zeros(x.shape) 
	cpm_x = (cPlus_filt_x + cMinus_filt_x) / 2. 

	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			Ux[i,j] = ((np.log(np.abs(cpm_x[i,j]+sig_x[i,j]/2.)) - np.log(np.abs(cpm_x[i,j] - sig_x[i,j]/2.))) / 
		(4*np.log(C_filt[i,j]) - 2*np.log(np.abs(cpm_x[i,j] + sig_x[i,j]/2.)) - 2*np.log(np.abs(cpm_x[i,j] - sig_x[i,j]/2.))))


	"""
	-----------------  y uncertainty -----------------------
	"""

	#get delta C_xy for x direction
	dC_y = np.zeros(frame_a.shape)  #delta C_i in x direction
	cPlus_y = np.zeros(frame_a.shape)
	cPlus_y[dy:, :] = frame_a[dy:,:]*frame_b_shift[0:-dy, :]
	cMinus_y = np.zeros(frame_a.shape)
	cMinus_y[dy:, :] = frame_a[0:-dy, :]*frame_b_shift[dy:, :]
	dC_y = cPlus_y - cMinus_y
	dC_y = dC_y - np.mean(dC_y) #remove the mean

	#smooth dC_y in the y direction
	for i in range(1, dC_y.shape[0]-1):
		for j in range(dC_y.shape[1]):
			dC_y[i, j] = (dC_y[i-1, j] + 2*dC_y[i, j] + dC_y[i+1, j])/4.


	#calculate covariance matrix
	S_y = np.zeros(frame_a.shape)
	for i in range(frame_a.shape[0]):
		for j in range(frame_a.shape[1]):

			S0 = dC_y[i,j]**2
			for k in range(1,4):
				try:
					if dC_y[i, j]*dC_y[i+k,j+k]/S0 < 0.05:
						S_y[i,j] = np.sum(dC_y[i, j]*dC_y[i:i+k, j:j+k])
						break
					if (k == 3):
						S_y[i,j] = np.sum(dC_y[i, j]*dC_y[i:i+k, j:j+k])
						break
				except IndexError:
					S_x[i,j] = 0.


	#smooth the fields and sum the fields
	cPlus_filt_y = (L**2) * filt.gaussian_filter(cPlus_y, L)[row_pix, col_pix]
	cMinus_filt_y = (L**2) * filt.gaussian_filter(cMinus_y, L)[row_pix, col_pix]
	S_filt_y = (L**2) * filt.gaussian_filter(S_y, L)[row_pix, col_pix]

	#calculate standard deviation  of correlation difference
	sig_y = np.sqrt(S_filt_y)
	Uy = np.zeros(x.shape) 
	cpm_y = (cPlus_filt_y + cMinus_filt_y) / 2. 

	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			Uy[i,j] = ((np.log(cpm_y[i,j]+sig_y[i,j]/2.) - np.log(cpm_y[i,j] - sig_y[i,j]/2.)) / 
		(4*np.log(C_filt[i,j]) - 2*np.log(cpm_y[i,j] + sig_y[i,j]/2.) - 2*np.log(cpm_y[i,j] - sig_y[i,j]/2.)))
	

	#Write uncertainty in m/s
	#Ux = Ux/dt/scaling_factor
	#Uy = Uy/dt/scaling_factor

	return Ux, Uy



def image_dewarp(frame_b,x,y, u,v):
	"""
	Dewarp the second image back onto the first using the
	displacement field and a Gaussian sub-pixel interpolation
	scheme.
	Reference: refer to paper 'Analysis of interpolation schemes for image deformation methods in PIV ' 2005

	Inputs:
		frame_b: 2d array
			second image of the PIV image pair
		u,v: 2d array 
			u and v velocity calculated by PIV algorithm

	Outputs:
		frame_b_shift: 2d array
			2nd frame dewarped back onto the first frame

	"""

	#Interpolate the dispalcement field onto each pixel 
	#using a bilinear interploation scheme


	#interpolate u and v 
	F1 = interp.RectBivariateSpline(y[::-1,0],x[0,:] , u)
	u_interp = F1(range(frame_b.shape[0]), range(frame_b.shape[1]))
	F2 = interp.RectBivariateSpline(y[::-1,0],x[0,:] ,v)
	v_interp = F2(range(frame_b.shape[0]), range(frame_b.shape[1]))

	#define shifted frame
	frame_shift = np.zeros(frame_b.shape)
	ul = u_interp.astype('int32') #lower int bound of u displacement
	vl = v_interp.astype('int32') #lower int bound of v displacement
	ur = abs(u_interp - ul)  #remainder of u displacement
	vr = abs(v_interp - vl)	#remainder of v displacement
	uc = (np.sign(u_interp)*np.ceil(np.abs(u_interp))).astype('int32') #upper int bound of u dispalcement
	vc = (np.sign(v_interp)*np.ceil(np.abs(v_interp))).astype('int32') #upper int bound of v displacement

	#shift second frame
	for i in range(frame_b.shape[0]):
		for j in range(frame_b.shape[1]):
			try:
				#get surrounding pixel intensities (fxy)
				f00 = frame_b[i-vl[i,j], j+ul[i,j]]
				f01 = frame_b[i-vc[i,j], j+ul[i,j]]	
				f10 = frame_b[i-vl[i,j], j+uc[i,j]]
				f11 = frame_b[i-vc[i,j], j+uc[i,j]]
				#do bilinear interpolation
				frame_shift[i,j] = ((1-ur[i,j])*(1-vr[i,j])*f00 + ur[i,j]*(1-vr[i,j])*f10 
									 + (1-ur[i,j])*vr[i,j] + ur[i,j]*vr[i,j]*f11)
			except IndexError:
				#index is out of bounds
				#Use unshifted value
				frame_shift[i,j] = 0.

	return(frame_shift)




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



# """
# -------------------------------------------------------------------------------------------------------
# ALTERNATE  COVARIANCE MATRIX CALCULATION

# This is another way to calculated the covariance matrix Sxy. Each dx and dy term is
# calcualted as separate fields and then summed together after filtering. 
# -------------------------------------------------------------------------------------------------------
# """

# S00_x = dC_x*dC_x
# S01_x = np.zeros(S00_x.shape)
# S01_x[1:,:] = dC_x[1:,:]*dC_x[0:-1,:]
# S02_x = np.zeros(S00_x.shape)
# S02_x[2:,:] = dC_x[2:,:]*dC_x[0:-2,:]
# S03_x = np.zeros(S00_x.shape)
# S03_x[3:,:] = dC_x[3:,:]*dC_x[0:-3,:]
# S10_x = np.zeros(S00_x.shape)
# S10_x[:,:-1] = dC_x[:,:-1]*dC_x[:,1:]
# S11_x = np.zeros(S00_x.shape)
# S11_x[1:,:-1] = dC_x[1:,:-1]*dC_x[0:-1,1:]
# S12_x = np.zeros(S00_x.shape)
# S12_x[2:,:-1] = dC_x[2:,:-1]*dC_x[0:-2,1:]
# S13_x = np.zeros(S00_x.shape)
# S13_x[3:,:-1] = dC_x[3:,:-1]*dC_x[0:-3,1:]
# S20_x = np.zeros(S00_x.shape)
# S20_x[:,:-2] = dC_x[:,:-2]*dC_x[:,2:]
# S21_x = np.zeros(S00_x.shape)
# S21_x[1:,:-2] = dC_x[1:,:-2]*dC_x[0:-1,2:]
# S22_x = np.zeros(S00_x.shape)
# S22_x[2:,:-2] = dC_x[2:,:-2]*dC_x[0:-2,2:]
# S23_x = np.zeros(S00_x.shape)
# S23_x[3:,:-2] = dC_x[3:,:-2]*dC_x[0:-3,2:]
# S30_x = np.zeros(S00_x.shape)
# S30_x[:,:-3] = dC_x[:,:-3]*dC_x[:,3:]
# S31_x = np.zeros(S00_x.shape)
# S31_x[1:,:-3] = dC_x[1:,:-3]*dC_x[0:-1,3:]
# S32_x = np.zeros(S00_x.shape)
# S32_x[2:,:-3] = dC_x[2:,:-3]*dC_x[0:-2,3:]
# S33_x = np.zeros(S00_x.shape)
# S33_x[3:,:-3] = dC_x[3:,:-3]*dC_x[0:-3,3:]


# S00_x_filt = filt.gaussian_filter(S00_x, L)[row_pix, col_pix]
# S01_x_filt = filt.gaussian_filter(S01_x, L)[row_pix, col_pix]
# S02_x_filt = filt.gaussian_filter(S02_x, L)[row_pix, col_pix]
# S03_x_filt = filt.gaussian_filter(S03_x, L)[row_pix, col_pix]
# S10_x_filt = filt.gaussian_filter(S10_x, L)[row_pix, col_pix]
# S11_x_filt = filt.gaussian_filter(S11_x, L)[row_pix, col_pix]
# S12_x_filt = filt.gaussian_filter(S12_x, L)[row_pix, col_pix]
# S13_x_filt = filt.gaussian_filter(S13_x, L)[row_pix, col_pix]
# S20_x_filt = filt.gaussian_filter(S20_x, L)[row_pix, col_pix]
# S21_x_filt = filt.gaussian_filter(S21_x, L)[row_pix, col_pix]
# S22_x_filt = filt.gaussian_filter(S22_x, L)[row_pix, col_pix]
# S23_x_filt = filt.gaussian_filter(S23_x, L)[row_pix, col_pix]
# S30_x_filt = filt.gaussian_filter(S30_x, L)[row_pix, col_pix]
# S31_x_filt = filt.gaussian_filter(S31_x, L)[row_pix, col_pix]
# S32_x_filt = filt.gaussian_filter(S32_x, L)[row_pix, col_pix]
# S33_x_filt = filt.gaussian_filter(S33_x, L)[row_pix, col_pix]

# #solve for final uncertainty
# S = (L**2)*( S00_x_filt + S01_x_filt + S02_x_filt + S03_x_filt  
# 		   + S10_x_filt + S11_x_filt + S12_x_filt + S13_x_filt
# 		   + S20_x_filt + S21_x_filt + S22_x_filt + S23_x_filt   
# 		   + S30_x_filt + S31_x_filt + S32_x_filt + S33_x_filt)