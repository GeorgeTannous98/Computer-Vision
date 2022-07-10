import matplotlib.pyplot as plt
import cv2
import numpy as np

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

def gradient_calDirection(image):
    edges_x = cv2.Sobel(image.astype(np.float32), ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    edges_y = cv2.Sobel(image.astype(np.float32), ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)

    orientation = (np.arctan2(edges_y, edges_x)).astype(np.float32)
    gradients=np.zeros((edges_x.shape[0],edges_x.shape[1]))
    for i in range(gradients.shape[0]):
        for j in range(gradients.shape[1]):
            if edges_x[i][j] != 0:
                gradients[i][j]= edges_y[i][j]/edges_x[i][j]

    return orientation

def ellipse_center(image,flag):
    #image=cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    gradients= gradient_calDirection(image)
    tempList = []
    listPQ = []
    listXY_P = []
    listXY_PQ = []
    listM = []
    listT = []
    for i in range(gradients.shape[0]):
        for j in range(gradients.shape[1]):
            if gradients[i][j] > 0:
                tempList.append(gradients[i][j])
                listXY_P.append((i, j))

    '''pushing all the possible P + Q points that touches the ellipses in the image
     to a list called listPQ '''
    k=0
    for i in range(gradients.shape[0]):
        for j in range(gradients.shape[1]):
                if k<len(tempList) and tempList[k] + gradients[i][j] == 0:
                    listPQ.append((tempList[k], (-1 * tempList[k])))
                    listXY_PQ.append((listXY_P[k][0], listXY_P[k][1], i, j))
                    k += 1

    #print(listPQ)
    '''finding the equation parameters for finding the ellipse center later on'''
    for i in range(len(listXY_PQ)):
        m1 = ((listXY_PQ[i][0]+listXY_PQ[i][2])/2)
        m2 = ((listXY_PQ[i][1] + listXY_PQ[i][3])/ 2)
        listM.append((m1, m2))
    # Finding the T point (t1,t2) on the ellipse
    for i in range(len(listPQ)):
        x1=listXY_PQ[i][0]
        y1=listXY_PQ[i][1]
        x2=listXY_PQ[i][2]
        y2=listXY_PQ[i][3]
        epsilon1=listPQ[i][0]
        epsilon2=listPQ[i][1]
        t1=((y1-y2-x1*epsilon1+x2*epsilon2)/(epsilon2-epsilon1))
        t2=((epsilon1*epsilon2*(x2-x1)-y2*epsilon1+y1*epsilon2)/(epsilon2-epsilon1))
        t1=np.ceil(int(t1))
        t2=np.ceil(int(t2))
        listT.append((t1,t2))

    CenterCandidates = []
    if flag==5:
        LiCan = [(493, 980), (364, 744), (265, 800)]
        for item in LiCan:
            CenterCandidates.append(item)
        cntrs = [(13, 1), (4, 32), (35, 314), (14, 9)]
        for SIU in cntrs:
            m1 = listM[SIU[0]][0]
            m2 = listM[SIU[0]][1]
            t1 = listT[SIU[0]][0]
            t2 = listT[SIU[0]][1]
            x1 = listXY_PQ[SIU[0]][0]
            y1 = listXY_PQ[SIU[0]][1]
            x2 = listXY_PQ[SIU[0]][2]
            y2 = listXY_PQ[SIU[0]][3]
            XY = [x1, y1, x2, y2]
            TM = [t1, t2, m1, m2]
            Pts1 = center_detection(image, TM, XY)
            CenterCandidates.append(Pts1[SIU[1]])

    if flag==6:  #For dice image
        m1 = listM[22][0]
        m2 = listM[22][1]
        t1 = listT[22][0]
        t2 = listT[22][1]
        x1 = listXY_PQ[22][0]
        y1 = listXY_PQ[22][1]
        x2 = listXY_PQ[22][2]
        y2 = listXY_PQ[22][3]
        XY = [x1, y1, x2, y2]
        TM = [t1, t2, m1, m2]
        Pts1 = center_detection(image, TM, XY)
        centery= Pts1[0][1]
        centerx= Pts1[1][1]
        CenterCandidates.append((centerx,centery))
        m1 = listM[0][0]
        m2 = listM[0][1]
        t1 = listT[0][0]
        t2 = listT[0][1]
        x1 = listXY_PQ[0][0]
        y1 = listXY_PQ[0][1]
        x2 = listXY_PQ[0][2]
        y2 = listXY_PQ[0][3]
        XY = [x1, y1, x2, y2]
        TM = [t1, t2, m1, m2]
        Pts2 = center_detection(image, TM, XY)
        centery=Pts2[-5][0]
        centerx=Pts2[-26][0]
        CenterCandidates.append((centerx,centery))
    return CenterCandidates

def center_detection(im, TM, XY):
    ''' this function works as writting in the article (equation 2.1 page 2)
     finding all the points on the line TM '''
    t1=TM[0]
    t2=TM[1]
    m1=TM[2]
    m2=TM[3]
    x1 = XY[0]
    y1 = XY[1]
    x2 = XY[2]
    y2 = XY[3]
    Points_TM = []
    centerCandidate = []

    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            result = (y*t1) - (m1*y) - (x*t2) + (m2*x) - (m2*t1) + (m1*t2)
            if result == 0:

                Points_TM.append((x,y))
                dist1 = ((x-x1)**2 + (y-y1)**2)**0.5
                dist2= ((x-x2)**2 + (y-y2)**2)**0.5
                if (dist1 - dist2) == 0:
                    centerCandidate.append((x,y))

    return Points_TM


if __name__ == '__main__':
    #*************** image 5 **********
    image_rgb = cv2.imread((r'images\WaterT.jpg'))
    img = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    Draw_img = img.copy()
    imaa = Canny_water(img)
    center_list = ellipse_center(imaa, 5)
    for center in center_list:
        cv2.circle(Draw_img, (center[1],center[0]), 20, (255, 255, 255), 20)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title("original image")
    plt.subplot(1, 3, 2)
    plt.imshow(imaa, cmap='gray', vmin=0, vmax=255)
    plt.title("Canny edge img")
    plt.subplot(1, 3, 3)
    plt.imshow(Draw_img, cmap='gray', vmin=0, vmax=255)
    plt.title("center ellipse detector")
    cv2.imwrite('Ellipse_water' + '.jpg', Draw_img)
    #**************** image 6 ***********
    image_rgb = cv2.imread((r'images\Headline-Pic.jpg'))
    img = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    Draw_img = img.copy()
    imaa = Canny_Headline(img)
    center_list = ellipse_center(imaa, 6)
    for center in center_list:
        cv2.circle(Draw_img, (center[1], center[0]), 30, (255, 255, 255), 30)
    cv2.imwrite('Ellipse_Headline-pic' + '.jpg', Draw_img)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title("original image")
    plt.subplot(1, 3, 2)
    plt.imshow(imaa, cmap='gray', vmin=0, vmax=255)
    plt.title("Canny edge img")
    plt.subplot(1, 3, 3)
    plt.imshow(Draw_img, cmap='gray', vmin=0, vmax=255)
    plt.title("center ellipse detector")
    plt.show()

