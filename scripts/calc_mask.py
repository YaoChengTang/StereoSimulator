from PIL import Image
import torch
import numpy as np




mask_old = np.array(Image.open('/data4/lzd/iccv25/vis/mask/1_1/0.png'))
mask_new = np.array(Image.open('/data4/lzd/iccv25/vis/test_mask/binary_mask_re.png'))


test = np.abs(mask_old - mask_new)

image = Image.fromarray(test)
image.save('../../vis/test_mask/out.png')