import cv2
import numpy as np


def compute_vegetation_indices(img):
    b=img.copy()[:,:,2] / 255
    g=img.copy()[:,:,1] / 255
    r=img.copy()[:,:,0] / 255
    r_g_b=(r + g + b).copy() + 1e-7

    r=r / (r_g_b).copy()
    g=g / (r_g_b).copy()
    b=b / (r_g_b).copy()

    ExG_=2 * g - r - b
    ExR_=1.4 * r - g
    NDI_=(g - r) / (g + r + 1e-7) + 1


    im_lab=cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
    a_lab=im_lab[:,:,1].astype(np.float)
    # Normalisations
    ExG_-=np.min(ExG_)
    ExG_/=np.max(ExG_)
    ExG=np.uint8(255. * ExG_)

    NDI_-=np.min(NDI_)
    NDI_/=np.max(NDI_)
    NDI=np.uint8(255. * NDI_)

    a_lab-=np.min(a_lab)
    a_lab/=np.max(a_lab)
    a_lab=np.uint8(255. * a_lab)

    # Mean
    mean=np.mean(np.asarray([ExG,NDI,255 - a_lab]),axis=0).astype(np.uint8)

    return ExG, a_lab, NDI,mean


def drawLines(image,lines,color=(0,0,0),thickness=5):
    """
    Draws lines. This function was used to debug Hough Lines generation.
    """
    # print("Going to print: ", len(lines))
    for l in lines:
        l.draw(image,color,thickness)



