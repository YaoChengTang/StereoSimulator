from simple_lama_inpainting import SimpleLama
from PIL import Image

simple_lama = SimpleLama()

img_path = "/data4/lzd/iccv25/imgR/imgR.png"
mask_path = "/data4/lzd/iccv25/mask/mask.png"

image = Image.open(img_path)
mask = Image.open(mask_path).convert('L')

result = simple_lama(image, mask)
result.save("inpainted.png")