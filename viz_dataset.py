import h5py
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image

def random_paste(background_image, ood_image, min_scale=0.25, max_scale=0.65):
    """Randomly scales and pastes the turtle image onto the background image"""
    
    w, h = ood_image.size
    # first, we will randomly downscale the turtle image
    new_w = int(random.uniform(min_scale, max_scale) * w)
    new_h = int(random.uniform(min_scale, max_scale) * h)
    resized_turtle_image = ood_image.resize((new_w, new_h))

    # second, will randomly choose the locations where to paste the new image
    start_w = random.randint(0, w - new_w)
    start_h = random.randint(0, h - new_h)

    # third, will create the blank canvas of the same size as the original image
    canvas_image = Image.new('RGBA', (w, h))

    # and paste the resized turtle onto it, preserving the mask
    canvas_image.paste(resized_turtle_image, (start_w, start_h), resized_turtle_image)
    
    # Turtle image is of mode RGBA, while background image is of mode RGB;
    # `.paste` requires both of them to be of the same type.
    background_image = background_image.copy().convert('RGBA')
    # finally, will paste the resized turtle onto the background image
    background_image.paste(resized_turtle_image, (start_w, start_h), resized_turtle_image)
    return background_image, canvas_image

with h5py.File('neurips2020/data/depth_train.h5', 'r') as hf:
    print(hf.keys())
    depth = hf["depth"]
    image = hf["image"]
    print(f"Depth: {type(depth)}")
    print(depth)

    print(f"Image: {type(image)}")
    print(image)
    depth = np.array(depth)
    image = np.array(image)
    # import pdb;pdb.set_trace()
    sample_image = image[0]
    sample_depth = depth[0].squeeze()

    import pdb;pdb.set_trace()
    sample_image_pil = Image.fromarray(sample_image)

    ood_image = Image.open("neurips2020/pineapple2.png")
    
    background_image, canvas_image = random_paste(sample_image_pil, ood_image, min_scale=0.25, max_scale=0.65)

    

    # plt.subplots(1, 2)
    # plt.subplot(1)

    # plt.imshow(sample_image)
    # plt.show()
