import cv2
import numpy as np
from cv2.xfeatures2d import SIFT_create
#xfeatures2d belongs to opencv-contrib-python package, not opencv-python.


def Denoise_Cameleon():
    source_files_List= []
    im = cv2.imread((r'cameleon__N_8__sig_noise_5__sig_motion_103\source_01.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im,gray))
    im = cv2.imread((r'cameleon__N_8__sig_noise_5__sig_motion_103\source_02.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im,gray))
    im = cv2.imread((r'cameleon__N_8__sig_noise_5__sig_motion_103\source_03.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im,gray))
    im = cv2.imread((r'cameleon__N_8__sig_noise_5__sig_motion_103\source_04.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im,gray))
    im = cv2.imread((r'cameleon__N_8__sig_noise_5__sig_motion_103\source_05.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im,gray))
    im = cv2.imread((r'cameleon__N_8__sig_noise_5__sig_motion_103\source_06.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im,gray))
    im = cv2.imread((r'cameleon__N_8__sig_noise_5__sig_motion_103\source_07.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im,gray))
    target_image = cv2.imread((r'cameleon__N_8__sig_noise_5__sig_motion_103\target.jpg'))
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    Cmon_bruh = np.zeros(target_gray.shape)
    sum_image = np.zeros(target_image.shape)
    #aligned = np.zeros(target_image.shape)
    for source in source_files_List:
        sift = SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, descriptor1 = sift.detectAndCompute(source[1], None)
        kp2, descriptor2 = sift.detectAndCompute(target_gray, None)

        IP = dict(algorithm=1, trees=5)
        SP = dict(checks=50)
        #best 4 matches homographic
        MATCH = cv2.FlannBasedMatcher(IP, SP)


        matches = MATCH.knnMatch(descriptor1, descriptor2, k=2)

        myList = list()
        for m, n in matches:
            checkif = 0.8
            if m.distance < checkif * n.distance:
                myList.append(m)

        if len(myList) > 5:
            Src = np.float32([kp1[m.queryIdx].pt for m in myList]).reshape(-1, 1, 2)
            Dst = np.float32([kp2[m.trainIdx].pt for m in myList]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(Src, Dst, cv2.RANSAC, 5.0)
            h, w = target_gray.shape

            aligned = cv2.warpPerspective(source[0], M, (w, h), flags=cv2.INTER_CUBIC)
            sum_image = sum_image + aligned
            ones_matrix = np.ones(source[1].shape)
            Align_img = cv2.warpPerspective(ones_matrix, M, (w, h), flags=cv2.INTER_CUBIC)
            Cmon_bruh = Cmon_bruh + Align_img

    Cmon_bruh[Cmon_bruh == 0] = 1
    B = sum_image[:, :, 0] / Cmon_bruh
    G = sum_image[:, :, 1] / Cmon_bruh
    R = sum_image[:, :, 2] / Cmon_bruh
    cleaned_image = np.dstack((B, G, R)).astype(np.uint8)


    cv2.imwrite('cleaned_cameleon' + '_out.jpg', cv2.resize(cleaned_image, (0, 0), fx=0.4, fy=0.4))

    original= cv2.imread((r'cameleon__N_8__sig_noise_5__sig_motion_103\target.jpg'))

    cv2.imshow('noisy target (cameleon)', cv2.resize(original, (0, 0), fx=0.5, fy=0.5))
    cv2.imshow('cleaned target (cameleon)', cv2.resize(cleaned_image, (0, 0), fx=0.5, fy=0.5))

    cv2.waitKey(6000)
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(original)
    # plt.title(" original target")
    # plt.subplot(1, 2, 2)
    # plt.imshow(cleaned_image)
    # plt.title("cleaned image")
    # plt.show()

    return

def Denoise_Eagle():
    source_files_List = []
    im = cv2.imread((r'eagle__N_16__sig_noise_13__sig_motion_76\source_01.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im, gray))
    im = cv2.imread((r'eagle__N_16__sig_noise_13__sig_motion_76\source_02.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im, gray))
    im = cv2.imread((r'eagle__N_16__sig_noise_13__sig_motion_76\source_03.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im, gray))
    im = cv2.imread((r'eagle__N_16__sig_noise_13__sig_motion_76\source_04.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im, gray))
    im = cv2.imread((r'eagle__N_16__sig_noise_13__sig_motion_76\source_05.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im, gray))
    im = cv2.imread((r'eagle__N_16__sig_noise_13__sig_motion_76\source_06.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im, gray))
    im = cv2.imread((r'eagle__N_16__sig_noise_13__sig_motion_76\source_07.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im, gray))
    im = cv2.imread((r'eagle__N_16__sig_noise_13__sig_motion_76\source_08.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im, gray))
    im = cv2.imread((r'eagle__N_16__sig_noise_13__sig_motion_76\source_09.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im, gray))
    im = cv2.imread((r'eagle__N_16__sig_noise_13__sig_motion_76\source_10.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im, gray))
    im = cv2.imread((r'eagle__N_16__sig_noise_13__sig_motion_76\source_11.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im, gray))
    im = cv2.imread((r'eagle__N_16__sig_noise_13__sig_motion_76\source_12.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im, gray))
    im = cv2.imread((r'eagle__N_16__sig_noise_13__sig_motion_76\source_13.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im, gray))
    im = cv2.imread((r'eagle__N_16__sig_noise_13__sig_motion_76\source_14.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im, gray))
    im = cv2.imread((r'eagle__N_16__sig_noise_13__sig_motion_76\source_15.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im, gray))
    target_image = cv2.imread((r'eagle__N_16__sig_noise_13__sig_motion_76\target.jpg'))
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    Cmon_bruh = np.zeros(target_gray.shape)
    sum_image = np.zeros(target_image.shape)
    #aligned = np.zeros(target_image.shape)
    for source in source_files_List:
        sift = SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, descriptor1 = sift.detectAndCompute(source[1], None)
        kp2, descriptor2 = sift.detectAndCompute(target_gray, None)

        IP = dict(algorithm=1, trees=5)
        SP = dict(checks=50)
        # best 4 matches homographic
        MATCH = cv2.FlannBasedMatcher(IP, SP)

        matches = MATCH.knnMatch(descriptor1, descriptor2, k=2)

        myList = list()
        for m, n in matches:
            checkif = 0.8
            if m.distance < checkif * n.distance:
                myList.append(m)

        if len(myList) > 5:
            Src = np.float32([kp1[m.queryIdx].pt for m in myList]).reshape(-1, 1, 2)
            Dst = np.float32([kp2[m.trainIdx].pt for m in myList]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(Src, Dst, cv2.RANSAC, 5.0)
            h, w = target_gray.shape

            aligned = cv2.warpPerspective(source[0], M, (w, h), flags=cv2.INTER_CUBIC)
            sum_image = sum_image + aligned
            ones_matrix = np.ones(source[1].shape)
            Align_img = cv2.warpPerspective(ones_matrix, M, (w, h), flags=cv2.INTER_CUBIC)
            Cmon_bruh = Cmon_bruh + Align_img


    Cmon_bruh[Cmon_bruh == 0] = 1
    B = sum_image[:, :, 0] / Cmon_bruh
    G = sum_image[:, :, 1] / Cmon_bruh
    R = sum_image[:, :, 2] / Cmon_bruh
    cleaned_image = np.dstack((B, G, R)).astype(np.uint8)



    cv2.imwrite('cleaned_eagle' + '_out.jpg', cv2.resize(cleaned_image, (0, 0), fx=0.4, fy=0.4))

    original = cv2.imread((r'eagle__N_16__sig_noise_13__sig_motion_76\target.jpg'))
    cv2.imshow('noisy target (eagle)', cv2.resize(original, (0, 0), fx=0.5, fy=0.5))
    cv2.imshow('cleaned target (eagle)', cv2.resize(cleaned_image, (0, 0), fx=0.5, fy=0.5))
    cv2.waitKey(6000)
    return


def Denoise_Einstein():
    source_files_List = []
    im = cv2.imread((r'einstein__N_5__sig_noise_5__sig_motion_274\source_01.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im, gray))
    im = cv2.imread((r'einstein__N_5__sig_noise_5__sig_motion_274\source_02.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im, gray))
    im = cv2.imread((r'einstein__N_5__sig_noise_5__sig_motion_274\source_03.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im, gray))
    im = cv2.imread((r'einstein__N_5__sig_noise_5__sig_motion_274\source_04.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im, gray))

    target_image = cv2.imread((r'einstein__N_5__sig_noise_5__sig_motion_274\target.jpg'))
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    Cmon_bruh = np.zeros(target_gray.shape)
    sum_image = np.zeros(target_image.shape)
    # aligned = np.zeros(target_image.shape)
    for source in source_files_List:
        sift = SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, descriptor1 = sift.detectAndCompute(source[1], None)
        kp2, descriptor2 = sift.detectAndCompute(target_gray, None)

        IP = dict(algorithm=1, trees=5)
        SP = dict(checks=50)
        # best 4 matches homographic
        MATCH = cv2.FlannBasedMatcher(IP, SP)

        matches = MATCH.knnMatch(descriptor1, descriptor2, k=2)

        myList = list()
        for m, n in matches:
            checkif = 0.8
            if m.distance < checkif * n.distance:
                myList.append(m)



        if len(myList) > 5:
            Src = np.float32([kp1[m.queryIdx].pt for m in myList]).reshape(-1, 1, 2)
            Dst = np.float32([kp2[m.trainIdx].pt for m in myList]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(Src, Dst, cv2.RANSAC, 5.0)
            h, w = target_gray.shape

            aligned = cv2.warpPerspective(source[0], M, (w, h), flags=cv2.INTER_CUBIC)
            sum_image = sum_image + aligned
            ones_matrix = np.ones(source[1].shape)
            Align_img = cv2.warpPerspective(ones_matrix, M, (w, h), flags=cv2.INTER_CUBIC)
            Cmon_bruh = Cmon_bruh + Align_img


    Cmon_bruh[Cmon_bruh == 0] = 1
    B = sum_image[:, :, 0] / Cmon_bruh
    G = sum_image[:, :, 1] / Cmon_bruh
    R = sum_image[:, :, 2] / Cmon_bruh
    cleaned_image = np.dstack((B, G, R)).astype(np.uint8)

    cv2.imwrite('cleaned_einstein' + '_out.jpg', cv2.resize(cleaned_image, (0, 0), fx=0.4, fy=0.4))

    original = cv2.imread((r'einstein__N_5__sig_noise_5__sig_motion_274\target.jpg'))

    cv2.imshow('noisy target (einstein)', cv2.resize(original, (0, 0), fx=0.5, fy=0.5))
    cv2.imshow('cleaned target (einstein)', cv2.resize(cleaned_image, (0, 0), fx=0.5, fy=0.5))

    cv2.waitKey(6000)

    return


def Denoise_palm():
    source_files_List = []
    im = cv2.imread((r'palm__N_4__sig_noise_5__sig_motion_ROT\source_01.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im, gray))
    im = cv2.imread((r'palm__N_4__sig_noise_5__sig_motion_ROT\source_02.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im, gray))
    im = cv2.imread((r'palm__N_4__sig_noise_5__sig_motion_ROT\source_03.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    source_files_List.append((im, gray))

    target_image = cv2.imread((r'palm__N_4__sig_noise_5__sig_motion_ROT\target.jpg'))
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    Cmon_bruh = np.zeros(target_gray.shape)
    sum_image = np.zeros(target_image.shape)
    # aligned = np.zeros(target_image.shape)
    for source in source_files_List:
        sift = SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, descriptor1 = sift.detectAndCompute(source[1], None)
        kp2, descriptor2 = sift.detectAndCompute(target_gray, None)

        IP = dict(algorithm=1, trees=5)
        SP = dict(checks=50)
        # best 4 matches homographic
        MATCH = cv2.FlannBasedMatcher(IP, SP)

        matches = MATCH.knnMatch(descriptor1, descriptor2, k=2)

        myList = list()
        for m, n in matches:
            checkif = 0.8
            if m.distance < checkif * n.distance:
                myList.append(m)

        if len(myList) > 5:
            Src = np.float32([kp1[m.queryIdx].pt for m in myList]).reshape(-1, 1, 2)
            Dst = np.float32([kp2[m.trainIdx].pt for m in myList]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(Src, Dst, cv2.RANSAC, 5.0)
            h, w = target_gray.shape

            aligned = cv2.warpPerspective(source[0], M, (w, h), flags=cv2.INTER_CUBIC)
            sum_image = sum_image + aligned
            ones_matrix = np.ones(source[1].shape)
            Align_img = cv2.warpPerspective(ones_matrix, M, (w, h), flags=cv2.INTER_CUBIC)
            Cmon_bruh = Cmon_bruh + Align_img


    Cmon_bruh[Cmon_bruh == 0] = 1
    B = sum_image[:, :, 0] / Cmon_bruh
    G = sum_image[:, :, 1] / Cmon_bruh
    R = sum_image[:, :, 2] / Cmon_bruh
    cleaned_image = np.dstack((B, G, R)).astype(np.uint8)



    cv2.imwrite('cleaned_palm' + '_out.jpg', cv2.resize(cleaned_image, (0, 0), fx=0.3, fy=0.3))

    original = cv2.imread((r'palm__N_4__sig_noise_5__sig_motion_ROT\target.jpg'))

    cv2.imshow('noisy target (palm)', cv2.resize(original, (0, 0), fx=0.5, fy=0.5))
    cv2.imshow('cleaned target (palm)', cv2.resize(cleaned_image, (0, 0), fx=0.5, fy=0.5))
    cv2.waitKey(3000)
    return

if __name__ == '__main__':
    Denoise_Cameleon()
    Denoise_Eagle()
    Denoise_Einstein()
    Denoise_palm()







