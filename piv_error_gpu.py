"""
PIV ERROR ANALYSIS FUNCTION
Cameron Dallas
University of Toronto
Department of Mechanical and Industrial Engineering
Turbulence Research Laboratory
"""

import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.cumath

import numpy as np
import scipy.ndimage.filters as filt
import scipy.interpolate as interp




def correlation_stats(frame_a, frame_b, x, y, u, v, dt, L, scaling_factor = 1., dx = 1, dy = 1):

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
	u = u*scaling_factor*dt
	v = v*scaling_factor*dt

	row_pix = y[::-1].astype('int32')
	col_pix = x.astype('int32')

	#Dewarp frame_b
	frame_b_shift = image_dewarp(frame_b, x,y,u,v)
	
	#complie gpu covariance calculation
	mod_cov = SourceModule("""
	__global__ void covariance(float *S, float *dC, float *tmp, int w, int h)
	{
		int blockId = blockIdx.x + blockIdx.y * gridDim.x;
		int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
		int i = threadId % w;
		int j = threadId / w;
		int k;
		
		for(k = 1; k < 5; k++)
		{
			if(i+k > w-1 || j+k > h-1)
			{
				S[threadId] = 0.0;
				break;
			}
			else if( dC[threadId + k + w*k] / dC[threadId] < 0.05)
			{
				int l;
				
				//sum over two dimensions
				for(l = 0; l <= k; l++)
				{
					int m;
					for(m = 0; m <= k; m++)
					{
						tmp[threadId] += dC[threadId + l*w + m];
					}
				}
				
				S[threadId] = dC[threadId]*tmp[threadId];
				break;
			}
			else if(k == 4)
			{
				int l;
				
				//sum over two dimensions
				for(l = 0; l <= k; l++)
				{
					int m;
					for(m = 0; m <= k; m++)
					{
						tmp[threadId] += dC[threadId + l*w + m];
					}
				}
				
				S[threadId] = dC[threadId]*tmp[threadId];
				break;
			}
		}
	}
	""")
	


	"""
	-----------------  x uncertainty -----------------------
	Get C and Sxy values for each pixel then filter and smooth them, which is
	equivalent to doing the sums over the interrogation window.
	"""
	
	#calculate C(u)
	C_u = np.multiply(frame_a, frame_b_shift)  #this value is the same for both x and y uncertainty
	
	#get delta C_xy for x direction
	dC_x = np.zeros(frame_a.shape)  #delta C_i in x direction
	cPlus_x = np.zeros(frame_a.shape)
	cMinus_x = np.zeros(frame_a.shape)
	cPlus_x[:, 0:-dx] = frame_a[:, 0:-dx]*frame_b_shift[:, dx:]
	cMinus_x[:, 0:-dx] = frame_a[:, dx:]*frame_b_shift[:, 0:-dx]
	dC_x = cPlus_x - cMinus_x
	dC_x = dC_x - np.mean(dC_x) #remove the mean

	#smooth dC in the x direction
	dC_x = smooth_filt(dC_x)

	#Define GPU variables
	dC_x = dC_x.astype(np.float32)
	S_x = np.empty_like(dC_x)
	tmp_x = np.zeros_like(dC_x)
	
	#GPU calculation parameters
	block_size = 16
	h,w = np.int32(dC_x.shape)
	x_blocks = w / block_size
	y_blocks = h / block_size
	
	#calculate covariance matrix
	covariance = mod_cov.get_function('covariance')
	covariance(drv.Out(S_x), drv.In(dC_x), drv.In(tmp_x), w, h, block = (block_size, block_size,1), grid = (x_blocks, y_blocks))

	#filter standard deviation
	sigma = L/2.  
	
	#smooth and sum the fields
	C_filt = (L**2) * filt.gaussian_filter(C_u, sigma, truncate = 2.0)[row_pix, col_pix]
	cPlus_filt_x = (L**2) * filt.gaussian_filter(cPlus_x, sigma, truncate = 2.0)[row_pix, col_pix]
	cMinus_filt_x = (L**2) * filt.gaussian_filter(cMinus_x, sigma, truncate = 2.0)[row_pix, col_pix]
	S_x_filt = (L**2) * filt.gaussian_filter(S_x, sigma, truncate = 2.0)[row_pix, col_pix]

	#define values for uncertainty formula
	sig_x = np.sqrt(S_x_filt)  
	cpm_x = (cPlus_filt_x + cMinus_filt_x) / 2. 

	#final x uncertainty
	Ux = ((np.log(cpm_x+sig_x/2.) - np.log(cpm_x - sig_x/2.)) / 
		(4*np.log(C_filt) - 2*np.log(cpm_x + sig_x/2.) - 2*np.log(cpm_x - sig_x/2.)))


	"""
	-----------------  y uncertainty -----------------------
	"""

	#get delta C_xy for y direction
	dC_y = np.zeros(frame_a.shape)  #delta C_i in x direction
	cPlus_y = np.zeros(frame_a.shape)
	cMinus_y = np.zeros(frame_a.shape)
	cPlus_y[dy:, :] = frame_a[dy:,:]*frame_b_shift[0:-dy, :]
	cMinus_y[dy:, :] = frame_a[0:-dy, :]*frame_b_shift[dy:, :]
	dC_y = cPlus_y - cMinus_y
	dC_y = dC_y - np.mean(dC_y) #remove the mean

	dC_y = np.rot90(smooth_filt(np.rot90(dC_y, k=1)), k=-1)

	#covariance matrix
	dC_y = dC_y.astype(np.float32)
	tmp_y = np.zeros_like(dC_y)
	S_y = np.empty_like(dC_y)
	covariance(drv.Out(S_y), drv.In(dC_y), drv.In(tmp_y), w, h, block = (block_size, block_size,1), grid = (x_blocks, y_blocks))

	#smooth and sum the fields
	cPlus_filt_y = (L**2) * filt.gaussian_filter(cPlus_y, sigma, truncate = 2.0)[row_pix, col_pix]
	cMinus_filt_y = (L**2) * filt.gaussian_filter(cMinus_y, sigma, truncate = 2.0)[row_pix, col_pix]
	S_filt_y = (L**2) * filt.gaussian_filter(S_y, sigma, truncate = 2.0)[row_pix, col_pix]

	#calculate standard deviation of correlation difference
	sig_y = np.sqrt(S_filt_y) 
	cpm_y = (cPlus_filt_y + cMinus_filt_y) / 2. 

	#final y uncertainty
	Uy = ((np.log(cpm_y+sig_y/2.) - np.log(cpm_y - sig_y/2.)) / 
		(4*np.log(C_filt) - 2*np.log(cpm_y + sig_y/2.) - 2*np.log(cpm_y - sig_y/2.)))
	

	#Write uncertainty in m/s
	Ux = Ux/dt/scaling_factor
	Uy = Uy/dt/scaling_factor

	return Ux, Uy



def image_dewarp(frame_b,x,y, u,v, method = 'bilinear'):

	"""
	Dewarp the second image back onto the first using the
	displacement field and a bilinear sub-pixel interpolation
	scheme.
	Reference: refer to paper 'Analysis of interpolation schemes for image deformation methods in PIV ' 2005
	Inputs:
		frame_b: 2d array
			second image of the PIV image pair
		u,v: 2d array 
			u and v velocity calculated by PIV algorithm
		method: string
			type of subpixel image dewarping function. 
	Outputs:
		frame_b_shift: 2d array
			2nd frame dewarped back onto the first frame
	"""
	
	#define data types
	DTYPEi = np.int32
	DTYPEf = np.float32
	
	#define output array
	frame_shift = np.empty(frame_b.shape, dtype = DTYPEf)

	#Interpolate the dispalcement field onto each pixel 
	#using a bilinear interploation scheme
	F1 = interp.RectBivariateSpline(y[::-1,0],x[0,:] , u, kx=1, ky=1)
	u_interp = F1(range(frame_b.shape[0]), range(frame_b.shape[1]))
	F2 = interp.RectBivariateSpline(y[::-1,0],x[0,:] ,v, kx=1, ky=1)
	v_interp = F2(range(frame_b.shape[0]), range(frame_b.shape[1]))


	#get displacement values
	ul = u_interp.astype('int32') #lower int bound of u displacement
	vl = v_interp.astype('int32') #lower int bound of v displacement
	ur = u_interp - ul  #remainder of u displacement
	vr = v_interp - vl	#remainder of v displacement

	#dewarp the image	
	if method == 'bilinear':
	
		#define dewarp function
		mod_dewarp = SourceModule("""
		__global__ void dewarp(float *frame_shift, float *frame_b, int *uc, int *vc, int *ul, int *vl, float *ur, float *vr, int w, int h)
		{
			int blockId = blockIdx.x + blockIdx.y * gridDim.x;
			int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
			int i = threadId % w;
			int j = threadId / w;
		
			//set out of bounds values to zero
			if(i+uc[threadId] < 0 || i+uc[threadId] > w-1 || j+vc[threadId] < 0 || j+vc[threadId] > h-1)
			{
				frame_shift[threadId] = 0.0;
			}
			else
			{ 		
				float f00 = frame_b[threadId + ul[threadId] - w * vl[threadId]];
				float f01 = frame_b[threadId + ul[threadId] - w * vc[threadId]];
				float f10 = frame_b[threadId + uc[threadId] - w * vl[threadId]];
				float f11 = frame_b[threadId + uc[threadId] - w * vc[threadId]];
		
				frame_shift[threadId] = (1-ur[threadId])*(1-vr[threadId])*f00 + ur[threadId]*(1-vr[threadId])*f10 + vr[threadId]*(1-ur[threadId])*f01 + ur[threadId]*vr[threadId]*f11;
			}		
		}
		""")
		
		#define shape of computation
		block_size = 32
		h, w = np.array(frame_b.shape).astype(np.int32)
		x_blocks = w / block_size
		y_blocks = h / block_size
		
		uc = (np.sign(u_interp)*np.ceil(np.abs(u_interp))).astype(np.int32)
		vc = (np.sign(v_interp)*np.ceil(np.abs(v_interp))).astype(np.int32)
		ur = abs(ur).astype(np.float32)  #remainder of u displacement
		vr = abs(vr).astype(np.float32)	#remainder of v displacement
		
		#get handle for dewarp function
		dewarp = mod_dewarp.get_function('dewarp')
		
		#compute dewarp
		dewarp(drv.Out(frame_shift), drv.In(frame_b), drv.InOut(uc), drv.InOut(vc), drv.InOut(ul), drv.InOut(vl), drv.InOut(ur), drv.InOut(vr),  w, h, block = (block_size, block_size, 1), grid = (x_blocks, y_blocks))

		return(frame_shift)

	else:
		#TODO add a gaussian method here
		raise ValueError('Image dewarping method not supported. Use Bilinear')




def smooth_filt(z):
	"""
	Simple filter to smooth an array in one direction
	
	Inputs: 
		z: 2d numpy array
			array for smoothing
	Outputs:
		z_sm: 2d array
			array smoothed along the rows
	"""
	
	z_sm = np.zeros(z.shape)
	z_sm[:, 1:-1] = (z[:, 0:-2] + 2*z[:,1:-1] + z[:,2:])/4.
	z_sm[:,0] = (2*z[:,0] + z[:,1])/3.
	z_sm[:,-1] = (2*z[:,-1] + z[:,-2])/3.
	
	return(z_sm)	

 


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
