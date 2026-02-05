import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image

img1 = np.array(Image.open("/mnt/c/users/Issea Occupia/Pictures/34F99057CBFE2BCD34B33C8AEBCB7585.jpg").convert('L')) / 255.0
img2 = np.array(Image.open("/mnt/c/users/Issea Occupia/Pictures/Screenshots/2026-01-21 100948.png").convert('L')) / 255.0
rng = np.random.default_rng()
kernel_v = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])

kernel_h = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]])
out1_v = convolve2d(img1, kernel_v, mode='same', boundary='symm')
out1_h = convolve2d(img1, kernel_h, mode='same', boundary='symm')
out2_v = convolve2d(img2, kernel_v, mode='same', boundary='symm')
out2_h = convolve2d(img2, kernel_h, mode='same', boundary='symm')

fig, axs = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)

axs[0, 0].imshow(img1, cmap='gray')
axs[0, 0].set_title('Original')
axs[0, 0].axis('off')

axs[1, 0].imshow(img2, cmap='gray')
axs[1, 0].set_title('Original')
axs[1, 0].axis('off')

norm1_v = out1_v / (np.max(np.abs(out1_v)) + 1e-8)
norm1_h = out1_h / (np.max(np.abs(out1_h)) + 1e-8)
norm2_v = out2_v / (np.max(np.abs(out2_v)) + 1e-8)
norm2_h = out2_h / (np.max(np.abs(out2_h)) + 1e-8)

axs[0, 1].imshow(norm1_v, cmap='gray', vmin=-1, vmax=1)
axs[0, 1].set_title('Vertical Edge')
axs[0, 1].axis('off')

axs[0, 2].imshow(norm1_h, cmap='gray', vmin=-1, vmax=1)
axs[0, 2].set_title('Horizontal Edge')
axs[0, 2].axis('off')

axs[1, 1].imshow(norm2_v, cmap='gray', vmin=-1, vmax=1)
axs[1, 1].set_title('Vertical Edge')
axs[1, 1].axis('off')

axs[1, 2].imshow(norm2_h, cmap='gray', vmin=-1, vmax=1)
axs[1, 2].set_title('Horizontal Edge')
axs[1, 2].axis('off')

plt.suptitle('Multiple Convolution Examples Photo', fontsize=20)
plt.show()