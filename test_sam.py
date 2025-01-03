import torch
from sam2.sam2.build_sam import build_sam2
from sam2.sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "./sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))


img = Image.open('/data4/lzd/iccv25/vis/imgL/1_1/6.png')
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(img)
    input_points = np.array([[667,977],[1138,96],[837,579]])
    input_labels = np.array([1,1,1])
    masks, _, _ = predictor.predict(np.array([1,1,1]))
    print(masks)