import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io


class Prob3():
    def __init__(self):
        """Load input color image inputPS1Q3.jpg here and assign it as a class variable"""
        self.img = io.imread("inputPS1Q3.jpg")
        ###### START CODE HERE ######
        ###### END CODE HERE ######
        pass

    def rgb2gray(self, rgb):
        """
        Do RGB to Gray image conversion here. Input is the RGB image and you must return the grayscale image as gray

        Returns:
            gray: grayscale image (1 channeled image with integer values lying between 0 - 255)
        """

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        return gray

    def prob_3_1(self):
        """
        Swap red and green color channels here, and return swapImg

        Returns:
            swapImg: RGB image with R and G channels swapped (3 channeled image with integer values lying between 0 - 255)
        """

        swapImg = self.img.copy()
        tmp = swapImg[:, :, 0].copy()
        swapImg[:, :, 0] = swapImg[:, :, 1]
        swapImg[:, :, 1] = tmp

        return swapImg

    def prob_3_2(self):
        """
        This function would simply call your rgb2gray function and return the grayscale image.

        Returns:
            grayImg: grayscale image (1 channeled image with integer values lying between 0 - 255)
        """

        grayImg = self.rgb2gray(self.img)

        return grayImg

    def prob_3_3(self):
        """
        Convert grayscale image to its negative.

        Returns:
            negativeImg: negative image (1 channeled image with integer values lying between 0 - 255)
        """

        negativeImg = 255 - self.prob_3_2()
        pass

        return negativeImg

    def prob_3_4(self):
        """
        Create mirror image of gray scale image here.

        Returns:
            mirrorImg: mirror image (1 channeled image with integer values lying between 0 - 255)
        """

        mirrorImg = cv2.flip(self.prob_3_2(), 1)

        return mirrorImg

    def prob_3_5(self):
        """
        Average grayscale image with mirror image here.

        Returns:
            avgImg: average of grayscale and mirror image (1 channeled image with integer values lying between 0 - 255)
        """

        avgImg = (self.prob_3_2() + self.prob_3_4()) // 2

        return avgImg

    def prob_3_6(self):
        """
        Create noise matrix N and save as noise.npy. Add N to grayscale image, clip to ensure that max value is 255.

        Returns:
            addNoiseImg: grayscale image after adding noise (1 channeled image with integer values lying between 0 - 255)
        """

        noise = np.random.randint(0, 255, size=self.prob_3_2().shape, dtype=np.int16)
        noisyImg = self.prob_3_2() + noise
        addNoiseImg = np.clip(noisyImg, 0, 255).astype(np.uint8)
        np.save('noise.npy', noise)

        return addNoiseImg


if __name__ == '__main__':
    p3 = Prob3()

    swapImg = p3.prob_3_1()
    grayImg = p3.prob_3_2()
    negativeImg = p3.prob_3_3()
    mirrorImg = p3.prob_3_4()
    avgImg = p3.prob_3_5()
    addNoiseImg = p3.prob_3_6()

    print(swapImg)
    print(grayImg)
    print(negativeImg)
    print(mirrorImg)
    print(avgImg)
    print(addNoiseImg)

    plt.figure(figsize=(12, 12))

    plt.subplot(2, 3, 1)
    plt.imshow(swapImg)
    plt.title('Swapped Channels')

    plt.subplot(2, 3, 2)
    plt.imshow(grayImg, cmap='gray')
    plt.title('Grayscale Image')

    plt.subplot(2, 3, 3)
    plt.imshow(negativeImg, cmap='gray')
    plt.title('Negative Image')

    plt.subplot(2, 3, 4)
    plt.imshow(mirrorImg, cmap='gray')
    plt.title('Mirror Image')

    plt.subplot(2, 3, 5)
    plt.imshow(avgImg, cmap='gray')
    plt.title('Average of Grayscale and Mirror')

    plt.subplot(2, 3, 6)
    plt.imshow(addNoiseImg, cmap='gray')
    plt.title('Image with Noise')

    plt.tight_layout()
    plt.show()