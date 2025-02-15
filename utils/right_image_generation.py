import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from simple_lama_inpainting import SimpleLama

class ImageProcess():
    def __init__(self):
        self.simple_lama = SimpleLama()

    def morphological_min(self, input_mask, kernel_size=3):
        input_mask_neg = -input_mask.float() 
        eroded_mask = -F.max_pool2d(input_mask_neg.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        return (eroded_mask > 0).squeeze().int() 

    def morphological_max(self, input_mask, kernel_size=3):
        dilated_mask = F.max_pool2d(input_mask.unsqueeze(0).unsqueeze(0).float(), kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        return (dilated_mask > 0).squeeze().int()

    def sum_pooling(self, input_mask, kernel_size=3):
        summed_mask = F.avg_pool2d(input_mask.unsqueeze(0).unsqueeze(0).float(), kernel_size=kernel_size, stride=1, padding=kernel_size//2) * kernel_size * kernel_size
        return (summed_mask > (kernel_size * kernel_size // 2)).squeeze().int() 
    def get_occlusion_mask(self, shifted):

        mask_up = shifted > 0
        mask_down = shifted > 0

        shifted_up = np.ceil(shifted)
        shifted_down = np.floor(shifted)

        for col in range(shifted.shape[1] - 2):
            loc = shifted[:, col:col + 1]  # keepdims
            loc_up = np.ceil(loc)
            loc_down = np.floor(loc)

            _mask_down = ((shifted_down[:, col + 2:] != loc_down) * (
            (shifted_up[:, col + 2:] != loc_down))).min(-1)
            _mask_up = ((shifted_down[:, col + 2:] != loc_up) * (
            (shifted_up[:, col + 2:] != loc_up))).min(-1)

            mask_up[:, col] = mask_up[:, col] * _mask_up
            mask_down[:, col] = mask_down[:, col] * _mask_down

        mask = mask_up + mask_down
        return mask
    def warp(self, left_image, disparity_map):
        H, W, C = left_image.shape
        # print(left_image)
        right_image_np = np.zeros_like(left_image)
        # mask = np.ones((H, W), dtype=np.uint8) * 255
        mask = np.zeros((H, W), dtype=np.uint8)
        xs, ys = np.meshgrid(np.arange(W), np.arange(H))
        shifted = xs - disparity_map
        mask_ = self.get_occlusion_mask(shifted)
        mask[mask_] = 255
        # for i in range(H):
        #     for j in range(W):
        #         if j - int(disparity_map[i,j]) < 0 :
        #             continue
        #         right_image_np[i,j - int(disparity_map[i,j])] = left_image[i,j]
        #         mask[i,j - int(disparity_map[i,j])] = 0
        right_image_np = cv2.cvtColor(right_image_np, cv2.COLOR_BGR2RGB)
        right_image = Image.fromarray(right_image_np)
        # right_image.save('imgR/imgR.png')
        print(mask)
        cv2.imwrite('mask/mask1.png', mask)
        exit(0)
        # mask = Image.fromarray(mask).convert('L')
        result = self.simple_lama(right_image, mask)
        # result.save('imgR/imgR_.png')
        return right_image
    
    def project_image(self, image, disp_map):
        feed_height, process_width, C = image.shape
        image = np.array(image)
        # background_image = np.array(background_image)

        # set up for projection
        warped_image = np.zeros_like(image).astype(float)
        warped_image = np.stack([warped_image] * 2, 0)
        xs, ys = np.meshgrid(np.arange(process_width), np.arange(feed_height))
        pix_locations = xs - disp_map

        # find where occlusions are, and remove from disparity map
        mask = self.get_occlusion_mask(pix_locations)
        masked_pix_locations = pix_locations * mask - process_width * (1 - mask)

        # do projection - linear interpolate up to 1 pixel away
        weights = np.ones((2, feed_height, process_width)) * 10000

        for col in range(process_width - 1, -1, -1):
            loc = masked_pix_locations[:, col]
            loc_up = np.ceil(loc).astype(int)
            loc_down = np.floor(loc).astype(int)
            weight_up = loc_up - loc
            weight_down = 1 - weight_up

            mask = loc_up >= 0
            mask[mask] = \
                weights[0, np.arange(feed_height)[mask], loc_up[mask]] > weight_up[mask]
            weights[0, np.arange(feed_height)[mask], loc_up[mask]] = \
                weight_up[mask]
            warped_image[0, np.arange(feed_height)[mask], loc_up[mask]] = \
                image[:, col][mask] / 255.

            mask = loc_down >= 0
            mask[mask] = \
                weights[1, np.arange(feed_height)[mask], loc_down[mask]] > weight_down[mask]
            weights[1, np.arange(feed_height)[mask], loc_down[mask]] = weight_down[mask]
            warped_image[1, np.arange(feed_height)[mask], loc_down[mask]] = \
                image[:, col][mask] / 255.

        weights /= weights.sum(0, keepdims=True) + 1e-7  # normalise
        weights = np.expand_dims(weights, -1)
        warped_image = warped_image[0] * weights[1] + warped_image[1] * weights[0]
        warped_image *= 255.

        # # now fill occluded regions with random background
        # if not disable_background:
        #     warped_image[warped_image.max(-1) == 0] = background_image[warped_image.max(-1) == 0]
        
        warped_image = warped_image.astype(np.uint8)
        mask_ =  np.zeros((feed_height, process_width), dtype=np.uint8)
        mask_[warped_image.max(-1) == 0] = 1
        mask_t = torch.from_numpy(mask_).squeeze()
        
        mask_max = self.morphological_max(mask_t)
        mask_min = self.morphological_min(mask_max)
        mask_re = self.sum_pooling(mask_min)
        mask_re = mask_re.numpy() * 255
        warped_image[mask_re > 0] = 0
        right_image_np = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)
        right_image_np[mask_re > 0] = 0
        right_image = Image.fromarray(right_image_np)
        mask = Image.fromarray(mask_re).convert('L')
        result = self.simple_lama(right_image, mask)
            
        return right_image, result, mask




