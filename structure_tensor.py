#     # window size is WxW
# C_Thr = 0.43    # threshold for coherency
# LowThr = 35     # threshold1 for orientation, it ranges from 0 to 180
# HighThr = 57    # threshold2 for orientation, it ranges from 0 to 180

# from

import numpy as np
import cv2 as cv

def calcGST(inputIMG, w):
    """
    from https://docs.opencv.org/4.2.0/d4/d70/tutorial_anisotropic_image_segmentation_by_a_gst.html
    :param inputIMG:
    :param w:
    :return:
    """
    img = inputIMG.astype(np.float32)
    # GST components calculation (start)
    # J =  (J11 J12; J12 J22) - GST
    img_diff_x = cv.Sobel(img, cv.CV_32F, 1, 0, 3) #Calculates the first image derivatives using an extended Sobel operator
    img_diff_y = cv.Sobel(img, cv.CV_32F, 0, 1, 3)
    img_diff_xy = cv.multiply(img_diff_x, img_diff_y)

    img_diff_xx = cv.multiply(img_diff_x, img_diff_x)
    img_diff_yy = cv.multiply(img_diff_y, img_diff_y)
    J11 = cv.boxFilter(img_diff_xx, cv.CV_32F, (w, w))
    J22 = cv.boxFilter(img_diff_yy, cv.CV_32F, (w, w))
    J12 = cv.boxFilter(img_diff_xy, cv.CV_32F, (w, w))
    # GST components calculations (stop)
    # eigenvalue calculation (start)
    # lambda1 = J11 + J22 + sqrt((J11-J22)^2 + 4*J12^2)
    # lambda2 = J11 + J22 - sqrt((J11-J22)^2 + 4*J12^2)
    tmp1 = J11 + J22
    tmp2 = J11 - J22
    tmp2 = cv.multiply(tmp2, tmp2)
    tmp3 = cv.multiply(J12, J12)
    tmp4 = np.sqrt(tmp2 + 4.0 * tmp3)
    lambda1 = tmp1 + tmp4  # biggest eigenvalue
    lambda2 = tmp1 - tmp4  # smallest eigenvalue
    # eigenvalue calculation (stop)
    # Coherency calculation (start)
    # Coherency = (lambda1 - lambda2)/(lambda1 + lambda2)) - measure of anisotropism
    # Coherency is anisotropy degree (consistency of local orientation)
    coherency = cv.divide(lambda1 - lambda2, lambda1 + lambda2)
    # Coherency calculation (stop)
    # orientation angle calculation (start)
    # tan(2*Alpha) = 2*J12/(J22 - J11)
    # Alpha = 0.5 atan2(2*J12/(J22 - J11))
    orientation = cv.phase(J22 - J11, 2.0 * J12, angleInDegrees=True)
    orientation = 0.5 * orientation
    # orientation angle calculation (stop)
    return coherency, orientation


def ApplyDenoisingAndStructureTensor(img_pathname, filterstrenght=20, xin=2600, yin=1700, xstep=2400, ystep=1700, W = 10):  #es: "sharad_data/s_00429402_thm.jpg" or "sharad_data/s_00387302_thm.jpg"
    imgIn_pre = cv.imread(img_pathname, cv.IMREAD_GRAYSCALE)

    imgIn = cv.fastNlMeansDenoising(imgIn_pre,None,filterstrenght,7,21)
    #imgInSquare = imgIn[1600:3600,2900:4900]
    #imgInTiny = imgIn[1700:3400,2600:5000]
    imgInTiny = imgIn[yin:yin+ystep,xin:xin+xstep]

    imgCoherency, ori = calcGST(imgInTiny, W)
    #imgCoherencySquare, oriSquare = calcGST(imgInSquare, W)

    ori[ori>=90] -=180

    _, imgCoherencyBin = cv.threshold(imgCoherency,0.1, 255, cv.THRESH_BINARY) #2nd value: C_Thr
    #_, imgCoherencyBinSquare = cv.threshold(imgCoherencySquare,0.1, 255, cv.THRESH_BINARY) 

    ori_bin = cv.inRange(ori, -50, 50)
    #ori_binSquare = cv.inRange(oriSquare, -50, 50)

    imgBin = cv.bitwise_and(imgCoherencyBin, ori_bin.astype(np.float32))
    #imgBinSquare = cv.bitwise_and(imgCoherencyBinSquare, ori_binSquare.astype(np.float32))
    dilation_size = 5
    element = cv.getStructuringElement( cv.MORPH_ELLIPSE,
                                       ( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       ( dilation_size, dilation_size ) );
    eroded = cv.erode(imgBin, element)
    imgBinClear = cv.dilate(eroded, element)
    
    return imgInTiny, imgCoherency, ori, imgCoherencyBin, ori_bin, imgBin, imgBinClear