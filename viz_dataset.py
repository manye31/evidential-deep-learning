import h5py
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image

SHOW = False

def random_paste(background_image, ood_image, min_scale=0.25, max_scale=0.65):
    """Randomly scales and pastes the ood image onto the background image"""
    w_back, h_back = background_image.size
    w_ood, h_ood = ood_image.size

    # Downsample to fit inside background image.Randomly downscale the ood image
    asp_ratio = min(w_back/w_ood, h_back/h_ood)
    w_fit = int(w_ood*asp_ratio*random.uniform(min_scale, max_scale))
    h_fit = int(h_ood*asp_ratio*random.uniform(min_scale, max_scale))
    ood_image = ood_image.resize((w_fit, h_fit))

    # second, will randomly choose the locations where to paste the new image
    start_w = random.randint(0, w_back - w_fit)
    start_h = random.randint(0, h_back - h_fit)

    # third, will create the blank canvas of the same size as the original image
    canvas_image = Image.new('RGBA', (w_back, h_back))

    # and paste the resized ood onto it, preserving the mask
    canvas_image.paste(ood_image, (start_w, start_h), ood_image)
    
    # ood image is of mode RGBA, while background image is of mode RGB;
    # `.paste` requires both of them to be of the same type.
    background_image = background_image.copy().convert('RGBA')

    # finally, will paste the resized ood onto the background image
    background_image.paste(ood_image, (start_w, start_h), ood_image)
    return background_image, canvas_image

with h5py.File('neurips2020/data/depth_train.h5', 'r') as hf:
    print(hf.keys())
    depth = hf["depth"]
    image = hf["image"]
    print(f"Depth: {type(depth)}")
    print(depth)
    print(f"Image: {type(image)}")
    print(image)
    depths = np.array(depth)
    images = np.array(image)

    for i in range(0,20):
        element = np.random.randint(images.shape[0])
        print(element)
        sample_image = images[element].squeeze()
        sample_depth = depth[element].squeeze()

        sample_depth = Image.fromarray(sample_depth)
        background_image = Image.fromarray(sample_image) # Image.open("neurips2020/trees.jpg") # 
        ood_item= Image.open("neurips2020/data/depth/image/items/betty.png")

        ood_image, canvas_image = random_paste(background_image, ood_item, min_scale=.75, max_scale=.95)
        if SHOW:
            plt.subplot(1,3,1)
            plt.imshow(ood_image)
            plt.subplot(1,3,2)
            plt.imshow(background_image)
            plt.subplot(1,3,3)
            plt.imshow(sample_depth)
            plt.show()

        ood_image.save(f'neurips2020/data/depth/image/ood/ood_betty{i}.png')
        background_image.save(f'neurips2020/data/depth/image/background/background_{i}.png')
        sample_depth.save(f'neurips2020/data/depth/depth/depth_{i}.png')