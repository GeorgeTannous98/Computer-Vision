import matplotlib.pyplot as plt
import cv2

def Canny_randomEllipses(im):
    blurred_img = cv2.GaussianBlur(im, (5, 5), 0.7)
    upper, thresh_im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lower = 0.2 * upper
    upper = upper * 1.81
    edges = cv2.Canny(blurred_img, lower, upper)
    return edges

def Canny_bigTruck(im):
    blurred_img = cv2.GaussianBlur(im, (5, 5), 1.8)
    upper, thresh_im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lower = 0.2 * upper
    upper = upper*1.81
    edges = cv2.Canny(blurred_img, lower, upper)
    return edges
def Canny_bicycle(im):
    blurred_img = cv2.GaussianBlur(im, (7, 7), 5)

    edges = cv2.Canny(blurred_img, 20, 268)

    return edges
def Canny_images(im):
    blurred_img = cv2.GaussianBlur(im, (11, 11), 3.486)

    edges = cv2.Canny(blurred_img, 20, 305)

    return edges

def Canny_water(im):
    blurred_img = cv2.GaussianBlur(im, (11, 11),3)

    high_thresh, thresh_im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lowThresh = 0.4 * high_thresh

    edges = cv2.Canny(blurred_img, lowThresh, high_thresh)
    return edges
def Canny_Headline(im):
    blurred_img = cv2.GaussianBlur(im, (11,11), 5.5)

    edges = cv2.Canny(blurred_img, 50, 150)

    return edges
if __name__ == '__main__':
    image_rgb = cv2.imread((r'images\WaterT.jpg'))
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    Canny= Canny_water(image_gray)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(Canny, cmap='gray', vmin=0, vmax=255)
    plt.title("Canny edge")
    plt.subplot(1, 2, 2)
    plt.imshow(image_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("original image")

    #---------------
    image_rgb = cv2.imread((r'images\bigTruck.png'))
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    Canny = Canny_bigTruck(image_gray)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(Canny, cmap='gray', vmin=0, vmax=255)
    plt.title("Canny edge")
    plt.subplot(1, 2, 2)
    plt.imshow(image_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("original image")
    #--------------------
    image_rgb = cv2.imread((r'images\randomEllipses.jpg'))
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    Canny = Canny_randomEllipses(image_gray)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(Canny, cmap='gray', vmin=0, vmax=255)
    plt.title("Canny edge")
    plt.subplot(1, 2, 2)
    plt.imshow(image_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("original image")
    #---------------------
    image_rgb = cv2.imread((r'images\images.jpg'))
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    Canny = Canny_images(image_gray)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(Canny, cmap='gray', vmin=0, vmax=255)
    plt.title("Canny edge")
    plt.subplot(1, 2, 2)
    plt.imshow(image_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("original image")
    #------------------
    image_rgb = cv2.imread((r'images\bicycle.png'))
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    Canny = Canny_bicycle(image_gray)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(Canny, cmap='gray', vmin=0, vmax=255)
    plt.title("Canny edge")
    plt.subplot(1, 2, 2)
    plt.imshow(image_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("original image")
    #--------------------
    image_rgb = cv2.imread((r'images\Headline-Pic.jpg'))
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    Canny = Canny_Headline(image_gray)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(Canny, cmap='gray', vmin=0, vmax=255)
    plt.title("Canny edge")
    plt.subplot(1, 2, 2)
    plt.imshow(image_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("original image")

    plt.show()


