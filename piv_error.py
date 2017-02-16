"""
PIV ERROR ANALYSIS FUNCTION


Cameron Dallas
University of Toronto
Department of Mechanical and Industrial Engineering
Turbulence Research Laboratory
"""


import numpy as np


def correlation_stats(im1, im2, ux, uy):

	"""
	Uses the Correlation Statistics method to approximate the error in the velocity
	arising from the image pair correlation algorithm. 
	See paper 'Generic a-posteriori uncertainty quantification for PIV vector fields 
	by correlation statistics. 2014'
	Inputs:
		im1 = first image
		im2 = second image
		ux = x displacement between the two images
		uy = y displacement between the two images

	Outputs:
		Ux = x uncertainty
		Uy = y uncertainty

	"""


