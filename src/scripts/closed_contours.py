import numpy as np
import cv2

class ContourIdentifier:
    def __init__(self, image, sigma):
        self.T1 = 32
        self.T2 = 128
        self.min_length = 50
        self.max_length = 200
        self.max_contours = 8000
        self.image = image
        self.sigma = sigma
        self.contour_image = np.zeros_like(image, dtype=np.uint8)
        self.sobel_filter()
        self.log_filter()

    def sobel_filter(self):
        sobelx = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=3)
        self.edge_strength = np.sqrt(sobelx**2 + sobely**2)
        self.edge_strength = np.uint8(self.edge_strength / self.edge_strength.max() * 255)

    def log_filter(self):
        r = int(round(3 * self.sigma * np.sqrt(2)))
        log_kernel = cv2.getGaussianKernel(2 * r + 1, self.sigma)
        log_kernel = log_kernel * log_kernel.T
        log_kernel = log_kernel * (2 * r + 1)**2 / np.sum(log_kernel)
        log_kernel = log_kernel - np.mean(log_kernel)
        temp = cv2.filter2D(self.image, cv2.CV_64F, log_kernel)
        zero_crossings = np.where((temp[:-1, :] * temp[1:, :] < 0) | (temp[:, :-1] * temp[:, 1:] < 0))
        self.contour_image[zero_crossings] = self.edge_strength[zero_crossings]

    def get_contours(self):
        contours, _ = cv2.findContours(self.contour_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def draw_contours(self, grayscale):
        contours = self.get_contours()
        contour_img = cv2.cvtColor(self.contour_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_img, contours, -1, (grayscale, grayscale, grayscale), 1)
        return contour_img

# Usage example:
# Load an image using OpenCV
image = cv2.imread('path_to_image', 0)  # Load as grayscale
sigma = 1.0  # Example sigma value for LoG filter

# Create an instance of the ContourIdentifier
contour_identifier = ContourIdentifier(image, sigma)

# Get the contours and draw them on the image
contour_img = contour_identifier.draw_contours(255)

# Display the result
cv2.imshow('Contours', contour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


