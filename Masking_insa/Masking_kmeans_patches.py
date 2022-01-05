import numpy as np
import cv2
import time
import vegetation_index as vi

def split_into_patches(index,r,c):
    n1,n2=index.shape
    patches=[]
    for i in range(r):
        for j in range(c):
            patches.append(index[i * n1 // r:(i + 1) * n1 // r,j * n2 // c:(j + 1) * n2 // c])
    return patches


def merge_patches(mask_patches,r,c,n1,n2):
    mask=np.zeros((n1,n2))
    l=0
    for i in range(r):
        for j in range(c):
            mask[i * n1 // r:(i + 1) * n1 // r,j * n2 // c:(j + 1) * n2 // c]=mask_patches[l]
            l+=1
    return mask


def kmeans_parameters(img,veg):
    """Compute the adequate parameters for different type of vegetation"""
    ExG, a_lab, NDI,mean =vi.compute_vegetation_indices(img)

    if veg == 'ble':
        index_vg=mean
        Ncluster=2
        Npatches=[2,2]
        morph_kernel_size=1
        dilat=0
    elif veg == 'levee':
        index_vg=ExG
        Ncluster=2
        Npatches=[2,2]
        morph_kernel_size=1
        dilat=2
    elif veg == 'mais':
        index_vg = 255-a_lab
        Ncluster=3
        Npatches = [2,1]
        morph_kernel_size = 2
        dilat = 1
    else:
        index_vg = 255-a_lab
        Ncluster = 3
        Npatches = [2,2]
        morph_kernel_size=1
        dilat=4
    return index_vg,Ncluster,Npatches, morph_kernel_size,dilat


def mask_kmeans(img, index_vg, Ncluster, morph_kernel_size,dilat,r,c):
    """The estimated mask is done throw the following steps:
    Apply the algorithm Kmeans on a vegetation index
    Binarization of the estimated labels (Kmeans result) using another Kmeans
    Noise reduction with a morphological operator on a vegetation index

        Input
            img: RGB image
            index_vg: index of vegetation (one or multiple indexes)
            Ncluster: number of clusters for the algorithm Kmeans
            morph_kernel_size: the size of the kernel for the morphological operator (closing and opening)
            rxc: number of patches

        Output
            Masked_image: the estimated mask (mask the ground)
            tau: computation time
     """

    t1 = time.time()
    # reshape the image img
    if len(index_vg.shape) ==3:
        l=index_vg.shape[2]
    else:
        l=1
    n1,n2 = index_vg.shape
    #split the image into rxc patches
    patches = split_into_patches(index_vg,r,c)
    # define stopping criteria
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,100,0.1)
    criteria1=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,0.1)
    labels_patches=[]
    centers_Patch=[]
    for i in range(len(patches)):
        index_vg = patches[i]
        pixel_values=np.float32(index_vg.reshape((-1,l)))
        # First Kmeans
        _,labels,(centers)=cv2.kmeans(pixel_values,Ncluster,None,criteria, 3,cv2.KMEANS_PP_CENTERS)


        # convert back to 8 bit values
        centers=np.uint8(centers)
        # flatten the labels array
        labels=labels.flatten()
        c1=np.float32((np.array(centers)).reshape((-1,l)))
        # Apply a threshold (80) to reduce the distance between classes that compose the ground
        if min(c1)<80:
            c1[np.argmin(c1)]=np.mean(c1)-10
        centers_Patch.append(c1)
        labels=(labels+i*Ncluster).reshape(index_vg.shape[0:2])
        labels_patches.append(labels)

    all_centers=np.array(centers_Patch).reshape(1,-1)
    #Merge all the patches labels
    labels=merge_patches(labels_patches,r,c,n1,n2)
    #Second Kmeans for binarization of the matrix labels
    _,labels1,(centers1)=cv2.kmeans(all_centers,2,None,criteria1,10,cv2.KMEANS_PP_CENTERS)
    #The ground class correspond to the class with the lowest center
    ground_center=np.argmin(np.sum(centers1,axis=1))
    #mask definition
    mask = np.ones((n1,n2))
    for j in range(len(labels1)):
        if labels1[j] == ground_center:
             mask[labels == j]=0

    mask=mask.reshape(n1,n2)

    # noise removal(erosion)
    kernel=np.ones((morph_kernel_size,morph_kernel_size),np.uint8)
    open_mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel,iterations=2)
    # sure background area
    kernel=np.ones((morph_kernel_size+dilat,morph_kernel_size+dilat),np.uint8)
    mask=cv2.dilate(open_mask,kernel,iterations=3)

    masked_image=img.copy()
    masked_image[mask == 0]=[0,0,0]

    t2=time.time()
    tau=t2 - t1


    return masked_image,tau,mask



























