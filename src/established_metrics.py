import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2

img = cv2.imread('./original_image.png')[100:400]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#rows, cols = img.shape

img_noise = cv2.imread('./image_29751.png')[100:400]
img_noise = cv2.cvtColor(img_noise, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4),
                         sharex=True, sharey=True)
ax = axes.ravel()

mse_none = mse(img, img)
ssim_none = ssim(img, img, data_range=img.max() - img.min(), multichannel=True)
psnr_none = psnr(img, img, data_range=img.max() - img.min())

mse_noise = mse(img, img_noise)
ssim_noise = ssim(img, img_noise,
                  data_range=img_noise.max() - img_noise.min(), multichannel=True)
psnr_noise = psnr(img, img_noise, data_range=img_noise.max() - img_noise.min())

ax[0].imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[0].set_xlabel(f'MSE: {mse_none:.2f}, SSIM: {ssim_none:.2f}, PSNR: {psnr_none:.2f}')
ax[0].set_title('Original image')

ax[1].imshow(img_noise, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[1].set_xlabel(f'MSE: {mse_noise:.2f}, SSIM: {ssim_noise:.2f}, PSNR: {psnr_noise:.2f}')
ax[1].set_title('Image with noise')

plt.tight_layout()
plt.show()