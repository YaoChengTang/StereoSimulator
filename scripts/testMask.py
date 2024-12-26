import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

x = np.load('/data4/lzd/iccv25/data/mask_sets1/billboardshortvideoyoutubeshortsshinewithshortstrendingforyouchina3dviralvideo/0.npy')


b = np.zeros_like(x)
b[x] = 255
# plt.imsave('../../vis/haha.png', b)
img = Image.fromarray(b)
img.save('../../vis/haha.png')