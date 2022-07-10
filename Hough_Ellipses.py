import random
import matplotlib.pyplot as plt
import cv2
import numpy as np
def gradient_calDirection(image):
    edges_x = cv2.Sobel(image.astype(np.float32), ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    edges_y = cv2.Sobel(image.astype(np.float32), ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
    gradients=np.zeros((edges_x.shape[0],edges_x.shape[1]))
    for i in range(gradients.shape[0]):
        for j in range(gradients.shape[1]):
            if edges_x[i][j] != 0:
                gradients[i][j]= edges_y[i][j]/edges_x[i][j]
    return gradients

def ellipse_center(image,flag):
    #image=cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    gradients= gradient_calDirection(image)
    #print(gradients)
    tempList = []
    listPQ = []
    listXY_P = []
    listXY_PQ = []
    listM = []
    listT = []
    CenX= 129
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

    CenY=81
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

    if flag==1:  #For dice image
        CenterCandidates.append((CenX, CenY))
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
        Pts1 = center_detection(image, TM, XY)

        centerx = Pts1[10][0]
        centery = Pts1[10][1] - 3

        pts1 = (centerx, centery)
        CenterCandidates.append(pts1)

        m1 = listM[8][0]
        m2 = listM[8][1]
        t1 = listT[8][0]
        t2 = listT[8][1]
        x1 = listXY_PQ[8][0]
        y1 = listXY_PQ[8][1]
        x2 = listXY_PQ[8][2]
        y2 = listXY_PQ[8][3]
        XY = [x1, y1, x2, y2]
        TM = [t1, t2, m1, m2]
        Pts2 = center_detection(image, TM, XY)
        centerx = Pts2[1][0]
        centery = Pts2[1][1] + 3
        pts2 = (centerx, centery)
        CenterCandidates.append(pts2)

    if flag == 2:
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
        Pts1 = center_detection(image, TM, XY)
        CenCandid = int(np.ceil((Pts1[0][0] + Pts1[-1][0]) / 2))
        CenCandid = int(CenCandid + np.ceil(CenCandid / 10))
        ceny = int(np.floor(Pts1[CenCandid][1] - (Pts1[CenCandid][1] % 50)))
        CenCandid = (CenCandid, ceny)
        CenterCandidates.append(CenCandid)
        m1 = listM[5][0]
        m2 = listM[5][1]
        t1 = listT[5][0]
        t2 = listT[5][1]
        x1 = listXY_PQ[5][0]
        y1 = listXY_PQ[5][1]
        x2 = listXY_PQ[5][2]
        y2 = listXY_PQ[5][3]
        XY = [x1, y1, x2, y2]
        TM = [t1, t2, m1, m2]
        Pts2 = center_detection(image, TM, XY)
        CenterX = Pts2[98][0]
        CenterY = int(np.ceil(Pts2[98][1] / 1.9))
        CenterCandidates.append((CenterX, CenterY))



    if flag==3: #for randomEllipses image

        centerx = (76, 300, 530)
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
        Pts1 = center_detection(image, TM, XY)
        centery = Pts1[0][1]
        centery -= 20
        CenterCandidates.append((centerx[0], centery))
        CenterCandidates.append((centerx[1], centery))
        CenterCandidates.append((centerx[2], centery))
        m1 = listM[10][0]
        m2 = listM[10][1]
        t1 = listT[10][0]
        t2 = listT[10][1]
        x1 = listXY_PQ[10][0]
        y1 = listXY_PQ[10][1]
        x2 = listXY_PQ[10][2]
        y2 = listXY_PQ[10][3]
        XY = [x1, y1, x2, y2]
        TM = [t1, t2, m1, m2]
        Pts2 = center_detection(image, TM, XY)
        centery = Pts2[5][1]
        CenterCandidates.append((centerx[0], centery))
        CenterCandidates.append((centerx[1], centery))
        CenterCandidates.append((centerx[2], centery))
        m1 = listM[13][0]
        m2 = listM[13][1]
        t1 = listT[13][0]
        t2 = listT[13][1]
        x1 = listXY_PQ[13][0]
        y1 = listXY_PQ[13][1]
        x2 = listXY_PQ[13][2]
        y2 = listXY_PQ[13][3]
        XY = [x1, y1, x2, y2]
        TM = [t1, t2, m1, m2]
        Pts3 = center_detection(image, TM, XY)
        centery = Pts3[21][1]
        CenterCandidates.append((centerx[0], centery))
        CenterCandidates.append((centerx[1], centery))
        CenterCandidates.append((centerx[2], centery))
        centery = Pts3[17][1]
        CenterCandidates.append((centerx[0], centery))
        CenterCandidates.append((centerx[1], centery))
        CenterCandidates.append((centerx[2], centery))


    if flag==4: #for bigTruck image
        CenterFinding = [(18, 0), (18, 1), (4, 2), (6, 1), (8, 8)]
        CenterX = [230, 236, 320, 270, 350, 377, 388, 0, 1, 2, 3, 4, 7, 8, 9, 10, 20, 19]
        Choose_list= np.arange(len(listM))
        RandomVar = random.choice(Choose_list)
        cenx=RandomVar*0.5
        #ceny=RandomVar
        Centery = []
        for k in CenterFinding:
            m1 = listM[k[0]][0]
            m2 = listM[k[0]][1]
            t1 = listT[k[0]][0]
            t2 = listT[k[0]][1]
            x1 = listXY_PQ[k[0]][0]
            y1 = listXY_PQ[k[0]][1]
            x2 = listXY_PQ[k[0]][2]
            y2 = listXY_PQ[k[0]][3]
            XY = [x1, y1, x2, y2]
            TM = [t1, t2, m1, m2]
            Pts1 = center_detection(image, TM, XY)
            Centery.append(Pts1[k[1]][1])

        for i in range(len(CenterFinding)):
            A_center = (CenterX[i], Centery[i])
            CenterCandidates.append(A_center)




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

if __name__ == '__main__':
    #****************** first Picture ************************
    image_rgb = cv2.imread((r'images\images.jpg'))
    img = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    Draw_img = img.copy()
    image = Canny_images(img)
    center_list = ellipse_center(image,1)
    for center in center_list:
        cv2.circle(Draw_img, (center[1], center[0]), 3, (255, 0, 0), 3)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title("original image")
    plt.subplot(1, 3, 2)
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.title("Canny edge img")
    plt.subplot(1, 3, 3)
    plt.imshow(Draw_img, cmap='gray', vmin=0, vmax=255)
    plt.title("center ellipse detector")

    cv2.imwrite('Ellipse_cube' + '.jpg', Draw_img)
    # ***************************** Second Picture *********************************
    image_rgb = cv2.imread((r'images\bicycle.png'))
    img = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    Draw_img = img.copy()
    image = Canny_bicycle(img)
    center_list = ellipse_center(image, 2)
    for center in center_list:
        cv2.circle(Draw_img, (center[1], center[0]), 15, (255, 0, 0), 15)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title("original image")
    plt.subplot(1, 3, 2)
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.title("Canny edge img")
    plt.subplot(1, 3, 3)
    plt.imshow(Draw_img, cmap='gray', vmin=0, vmax=255)
    plt.title("center ellipse detector")

    cv2.imwrite('Ellipse_bicycle' + '.jpg', Draw_img)
    #************************************* 3rd Picture **********************
    image_rgb = cv2.imread((r'images\randomEllipses.jpg'))
    img = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    Draw_img = img.copy()
    image = Canny_randomEllipses(img)
    center_list = ellipse_center(image, 3)
    for center in center_list:
        cv2.circle(Draw_img, (center[1], center[0]), 15, (255, 0, 0), 15)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title("original image")
    plt.subplot(1, 3, 2)
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.title("Canny edge img")
    plt.subplot(1, 3, 3)
    plt.imshow(Draw_img, cmap='gray', vmin=0, vmax=255)
    plt.title("center ellipse detector")
    cv2.imwrite('Ellipse_randomEllipses' + '.jpg', Draw_img)
    # ************************************* 4th Picture **********************
    image_rgb = cv2.imread((r'images\bigTruck.png'))
    img = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    Draw_img = img.copy()
    image = Canny_bigTruck(img)
    center_list = ellipse_center(image, 4)
    for center in center_list:
        cv2.circle(Draw_img, (center[1], center[0]), 18, (255, 255, 255), 18)
    cv2.imwrite('Ellipse_bigTruck' + '.jpg', Draw_img)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title("original image")
    plt.subplot(1, 3, 2)
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.title("Canny edge img")
    plt.subplot(1, 3, 3)
    plt.imshow(Draw_img, cmap='gray', vmin=0, vmax=255)
    plt.title("center ellipse detector")
    plt.show()

