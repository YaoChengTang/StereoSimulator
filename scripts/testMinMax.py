import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

def morphological_erosion(input_mask, kernel_size=3):
    input_mask_neg = -input_mask.float() 
    eroded_mask = -F.max_pool2d(input_mask_neg.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    return (eroded_mask > 0).squeeze().int() 

def morphological_dilation(input_mask, kernel_size=3):
    dilated_mask = F.max_pool2d(input_mask.unsqueeze(0).unsqueeze(0).float(), kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    return (dilated_mask > 0).squeeze().int()

def sum_pooling(input_mask, kernel_size=3):
    summed_mask = F.avg_pool2d(input_mask.unsqueeze(0).unsqueeze(0).float(), kernel_size=kernel_size, stride=1, padding=kernel_size//2) * kernel_size * kernel_size
    return (summed_mask > (kernel_size * kernel_size // 2)).squeeze().int() 

# 示例：去噪声的操作

mask =  Image.open('../../vis/mask/1_1/0.png').convert('L')
transform = transforms.ToTensor()
mask = transform(mask)
print(mask.sum())
# mask = torch.tensor([[ [0, 0, 1], [0, 1, 1], [0, 1, 0] ]], dtype=torch.uint8)  # 示例黑白mask
mask = mask.squeeze()  # 除去多余的维度
# 腐蚀和膨胀操作
dilated_mask = morphological_dilation(mask)
eroded_mask = morphological_erosion(dilated_mask)

# 使用sum pooling来精细化
refined_mask = sum_pooling(eroded_mask)

print("原始mask:")
print(mask)
print("腐蚀后的mask:")
print(eroded_mask)
print("膨胀后的mask:")
print(dilated_mask)
print("精细化后的mask（sum pooling）:")
print(refined_mask)
image_numpy = refined_mask.squeeze().cpu().numpy()  # 去除通道维度并转换为 NumPy 数组
image_pil = Image.fromarray(image_numpy.astype(np.uint8) * 255)  # 0 -> 0, 1 -> 255

# 4. 保存为二值图像文件（如 PNG 或 JPEG）
image_pil.save("/data4/lzd/iccv25/vis/test_mask/binary_mask_re.png")