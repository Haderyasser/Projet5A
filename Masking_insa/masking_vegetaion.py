import cv2
import os
import Masking_kmeans_patches as mki
import time

import sys

if __name__ == '__main__':
    folder = sys.argv[1]
    path_output = sys.argv[2]
    if not os.path.exists(path_output):
        os.mkdir(path_output)
    if not os.path.exists(os.path.join(path_output, 'masked_image')):
        os.mkdir(os.path.join(path_output, 'masked_image'))
    if not os.path.exists(os.path.join(path_output, 'mask')):
        os.mkdir(os.path.join(path_output, 'mask'))
    i=0
    for imname in sorted(os.listdir(folder)):
        # %% Read image
        t1 = time.time()
        image = cv2.imread(os.path.join(folder, imname))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        n1, n2, n3 = image.shape

        # %% ################################## Masking  ##################################
        index_vg, Ncluster, Npatches, morph_kernel_size, dilat = mki.kmeans_parameters(image,
                                                                                       'autres')  # masking parameters

        masked_image, tau, mask = mki.mask_kmeans(image, index_vg, Ncluster, morph_kernel_size, dilat, 1,
                                                  1)  # image masking

        #masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        t2 = time.time()
        cv2.imwrite(os.path.join(os.path.join(path_output, 'mask'), imname), mask * 255)
        cv2.imwrite(os.path.join(os.path.join(path_output, 'masked_image'), imname[:-4] + 'm.JPG'), masked_image)
        print("{} is done in {}s (total time): the {}th images".format(imname,str(t2-t1),i))
        i += 1

