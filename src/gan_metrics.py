# calculate inception score in numpy
import numpy as np
from numpy import asarray, expand_dims, log, mean, exp, cov, trace, iscomplexobj
from PIL import Image
from numpy.random import random
from scipy.linalg import sqrtm
 

img_gan = np.array(Image.open('./src/comp/tower_gan.png'))
img_og = np.array(Image.open('./src/comp/tower.png'), dtype=np.float64)

# calculate frechet inception distance
def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid
 
# define two collections of activations

# fid between act1 and act1
fid = calculate_fid(img_gan.reshape((2048,512)), img_og.reshape((2048,512)))
print('FID: %.3f' % fid)
