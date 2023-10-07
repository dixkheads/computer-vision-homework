import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class Prob2():
    def __init__(self):
        """Load inputAPS1Q2.npy here as a class variable A."""
        self.A = np.load('inputAPS1Q2.npy')
        pass

    def prob_2_1(self):
        """Do plotting of intensities of A in decreasing value."""
        sorted_intensities = np.sort(self.A.flatten())[::-1]
        colors = cm.Greys_r(sorted_intensities)
        plot_height = 0.01

        plt.figure(figsize=(10, 1))
        plt.bar(range(len(sorted_intensities)), np.ones(len(sorted_intensities)),
                color=colors, width=1.0)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.xlim(0, 10000)
        plt.ylim(0, 1)
        plt.title('2.1 Intensity Plot')
        plt.show()
        pass

    def prob_2_2(self):
        """Display histogram of A's intensities with 20 bins here."""
        plt.hist(self.A.flatten(), bins=20, edgecolor='black')
        plt.xlabel('Intensity Range')
        plt.ylabel('Frequency')
        plt.title('2.2 Intensity Histogram')
        plt.show()
        pass

    def prob_2_3(self):
        """
        Create a new matrix X that consists of the bottom left quadrant of A here.
        Returns:
            X: bottom left quadrant of A which is of size 50 x 50
        """
        X = self.A[self.A.shape[0] // 2:, :self.A.shape[1] // 2]
        # self.A = X
        # self.prob_2_2()
        pass

        return X

    def prob_2_4(self):
        """Create a new matrix Y, which is the same as A, but with A’s mean intensity value subtracted from each pixel.
        Returns:
            Y: A with A's mean intensity subtracted from each pixel. Output Y is of size 100 x 100.
        """
        Y = self.A - np.mean(self.A)

        return Y

    def prob_2_5(self):
        """
        Create your threshholded A i.e Z here.
        Returns:
            Z: A with only red pixels when the original value of the pixel is above the threshhold. Output Z is of size 100 x 100.
        """
        threshold = 0.5
        # Z = np.where(self.A > threshold, self.A, 0)
        Z = np.zeros((self.A.shape[0], self.A.shape[1], 3))

        Z[self.A > threshold, 0] = 1  # Set red channel (R) to 1

        plt.imshow(Z)
        plt.title('2.5 Image')
        plt.show()
        return Z


if __name__ == '__main__':
    p2 = Prob2()

    p2.prob_2_1()
    p2.prob_2_2()

    X = p2.prob_2_3()
    Y = p2.prob_2_4()
    Z = p2.prob_2_5()