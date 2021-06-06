from cv2 import HOGDescriptor
from skimage.feature import hog
from skimage import exposure
from skimage import color
import matplotlib.pyplot as plt
import skimage.io
import cv2

def hog_feature():
    winSize = (30,30)
    blockSize = (10, 10)
    blockStride = (5,5)
    cellSize = (2,2)
    nbins = 16
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    useSignedGradients = True
    hog = HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels, useSignedGradients)


    return hog


# print(img.shape)
def composed_hog(img, multichannel=False):
    #img = color.rgb2gray(img)
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(3,3),
                        cells_per_block=(1,1), block_norm="L2-Hys", visualize=True, multichannel=multichannel)

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    #
    # ax1.axis('off')
    # ax1.imshow(img, cmap=plt.cm.gray)
    # ax1.set_title('Input image')
    #
    # # Rescale histogram for better display
    # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    #
    # ax2.axis('off')
    # ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    # ax2.set_title('Histogram of Oriented Gradients')
    # plt.show()
    return hog_image

# testPath = "img_518.jpg"
# img = skimage.io.imread("class2.jpg")
#
# composed_hog(img)