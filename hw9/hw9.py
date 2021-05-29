import sys
import os
import math
import cv2

import numpy as np
import matplotlib.pyplot as plt

M, N = 16, 16


def DCT(input):

    # declare temporary output array
    tmp_output = np.zeros((16, 16, 3))

    # declare C(u), C(v)
    Cu, Cv = 0.0, 0.0
    for i in range(3):
        arr = np.array([])
        for u in range(M):
            for v in range(N):

                # according from the formula of DCT
                if u == 0:
                    Cu = 1 / np.sqrt(M)
                else:
                    Cu = np.sqrt(2) / np.sqrt(M)
                if v == 0:
                    Cv = 1 / np.sqrt(N)
                else:
                    Cv = np.sqrt(2) / np.sqrt(N)

                # calculate DCT
                tmp_sum = 0
                for x in range(M):
                    for y in range(N):
                        dct = input[x][y][i] * math.cos((2 * x + 1) * u * math.pi / (
                            2 * M)) * math.cos((2 * y + 1) * v * math.pi / (2 * N))
                        tmp_sum += dct

                tmp_output[u][v][i] = Cu * Cv * tmp_sum
                arr = np.append(arr, Cu * Cv * tmp_sum)
        arr.sort()
        maxval = arr[arr.size-16]
        for u in range(M):
            for v in range(N):
                if tmp_output[u][v][i] < maxval:
                    tmp_output[u][v][i] = 0

    return tmp_output

def IDCT(input):

    # declare temporary output array
    tmp_output = np.zeros((16, 16, 3))

    # declare C(u), C(v)
    Cu, Cv = 0.0, 0.0
    for i in range(3):
        for x in range(M):
            for y in range(N):

                # calculate IDCT
                tmp_sum = 0
                for u in range(M):
                    for v in range(N):

                        # according from the formula of IDCT
                        if u == 0:
                            Cu = 1 / np.sqrt(M)
                        else:
                            Cu = np.sqrt(2) / np.sqrt(M)
                        if v == 0:
                            Cv = 1 / np.sqrt(N)
                        else:
                            Cv = np.sqrt(2) / np.sqrt(N)

                        idct = input[u][v][i] * math.cos((2 * x + 1) * u * math.pi / (
                            2 * M)) * math.cos((2 * y + 1) * v * math.pi / (2 * N))
                        tmp_sum += Cu * Cv * idct
                if tmp_sum < 0: tmp_sum = 0
                elif tmp_sum > 255: tmp_sum = 255
                
                tmp_output[x][y][i] = tmp_sum

    return tmp_output

def main():

    # read image
    lena_ = cv2.imread('InputData/photo1.jpg')
    lena = cv2.resize(lena_, dsize=(256, 256), interpolation=cv2.INTER_AREA)

    # declare arrarys with full padding zeros
    dct_transform_lena = np.zeros((256, 256, 3))
    quantized_lena = np.zeros((256, 256, 3))
    inverse_quantized_lena = np.zeros((256, 256, 3))
    idct_transform_lena = np.zeros((256, 256, 3))

    # cut lena into 8x8 and send into DCT() to calculate
    for x in range(0, 256, 16):
        for y in range(0, 256, 16):
            cut_lena = lena[x: x + 16, y: y + 16]
            dct_lena = DCT(cut_lena)
            dct_transform_lena[x: x + 16, y: y + 16] = np.copy(dct_lena)
    print("DCT done")

    # cut inverse_quantized_lena into 8x8 and send into IDCT() to calculate
    for x in range(0, 256, 16):
        for y in range(0, 256, 16):
            cut_lena = dct_transform_lena[x: x + 16, y: y + 16]
            idct_lena = IDCT(cut_lena)
            idct_transform_lena[x: x + 16, y: y + 16] = np.copy(idct_lena)
    print("IDCT done")

    cv2.imwrite('OutputData/original1.png', lena)
    # write images as png files
    cv2.imwrite('OutputData/dct_lena1.png', dct_transform_lena)

    cv2.imwrite('OutputData/idct_lena1.png', idct_transform_lena)

    cv2.imshow('Original Image', lena)
    cv2.waitKey()
    cv2.imshow('DCT Transform', dct_transform_lena)
    cv2.waitKey()
    cv2.imshow('After IDCT', idct_transform_lena)
    cv2.waitKey()

    # save PSNR result as txt


if __name__ == "__main__":
    main()