import numpy as np

import skimage
import skimage.util
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation
import matplotlib.pyplot as plt

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions
    sigma = 0.1
    noisy = skimage.util.random_noise(image, var=sigma**2)
    image = skimage.restoration.denoise_tv_chambolle(noisy, weight=0.1, multichannel=True)
    #image = skimage.restoration.denoise_bilateral(image)
    # apply threshold
    gray = skimage.color.rgb2gray(image)
    thresh = skimage.filters.threshold_otsu(gray)
    bw = skimage.morphology.closing(gray < thresh,skimage.morphology.square(6))
    #bw = skimage.morphology.erosion(bw)
    bw = skimage.morphology.dilation(bw)
    # remove artifacts connected to image border
    cleared = skimage.segmentation.clear_border(bw)
    #bw = cleared
    # label image regions
    label_image = skimage.measure.label(cleared)
    bw = np.invert(bw).astype(int)
    #bboxes = skimage.measure.regionprops(label_image)
    fig, ax = plt.subplots(figsize=(10, 6))
    for region in skimage.measure.regionprops(label_image):
    # take regions with large enough areas
        if region.area >= 300:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            small_box = [minr, minc, maxr, maxc ]
            bboxes.append(small_box)
    bboxes = np.matrix(bboxes)
    return bboxes, bw