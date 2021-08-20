# Built-int imports
import sys

# External imports 
import cv2 as cv
import numpy as np

class ImageAnalysis:
    """
    This is a python class that was implemented to be able to perform 
    a correct analysis to an image, this class has a method to plot the 
    histogram of a specific image and also has another method which 
    allows to change the gamma factor of a specific image. 
    :param image: BGR-image to plot its histogram or to change its gamma factor
    """
    def __init__(self, image):
        self.image = image
    
    def nothing(self,x):
        """
        nothing function that is necessary for the trackbar event
        """
        pass

    def create_trackbar(self,name,title):
        return cv.createTrackbar(
            name,
            title,
            1,
            20,
            self.nothing
        )
    
    def plot_histogram(self):
        """
        This is a method which allows to plot the histogram of a 
        specific rgb image.
        """
        gray_image = cv.cvtColor(self.image, cv.COLOR_RGB2GRAY)
        # Define the dimensions of the plot
        wbins = 256
        hbins = 256

        #cv.calcHist(images, channels, mask, histSize, ranges)
        histr = cv.calcHist([gray_image],[0],None,[hbins],[0,wbins])
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(histr)
        hist_image = np.zeros([hbins, wbins], np.uint8)

        # Plot the lines to make the histogram
        for w in iter(range(wbins)):
            binVal = histr[w]
            intensity = int(binVal*(hbins-1)/max_val)
            cv.line(hist_image, (w,hbins), (w,hbins-intensity),255)
            cv.imshow("IMAGE'S HISTOGRAM",hist_image)
    
    def set_gamma_value(self, gamma):
        """
        This is a method to set and adjust the gamma value of a 
        specific rgb image.
        """
        image = self.image.copy()
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv.LUT(image, table)

    

    def get_gamma_value(self):
        """
        This is a method which allows to iterate using a trackbar
        in order to change the gamma value of an image and thus to find 
        the better value of gamma.
        """
        cv.namedWindow("Image Gamma adjust")
        self.create_trackbar("Gamma-X10", "Image Gamma adjust")

        while(True):
            gamma_value = 0.1*cv.getTrackbarPos("Gamma-X10","Image Gamma adjust")

            if (gamma_value != 0):
                gamma_image = self.set_gamma_value(gamma_value)
            else:
                gamma_image = self.set_gamma_value(1)

            cv.imshow("Image Gamma adjust", gamma_image)

            if (cv.waitKey(1) & 0xFF == ord('q')):
                break
        cv.destroyAllWindows()


def main():
    pass


if __name__=='__main__':
    sys.exit(main())