import cv2
import numpy as np
import pytesseract

FRAMEWIDTH = 640
FRAMEHEIGHT = 400
BRIGHTNESS = 100

cap = cv2.VideoCapture('Resources/Getcard4.mp4')
cap.set(3, FRAMEWIDTH)
cap.set(4, FRAMEHEIGHT)
cap.set(10, BRIGHTNESS)

def preprocessing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(gray, (7,7), 1)
    canny = cv2.Canny(gaussian, 50, 200)
    kernel = np.ones((5,5))
    dilate = cv2.dilate(canny, kernel, iterations=2)
    erode = cv2.erode(dilate, kernel, iterations=1)
    return erode

def GetContours(img):
    try:
        biggest = np.array([])
        maxArea = 0
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 85000: #89 for dilate
                # cv2.drawContours(img_contour, cnt, -1, (255,255,0), 3)
                arcLength = cv2.arcLength(cnt, True)
                appox = cv2.approxPolyDP(cnt, 0.02*arcLength, True)
                corners = len(appox)
                if area > maxArea and len(appox) == 4:
                    biggest = appox
                    maxArea = area
                    # print(biggest)
                    cv2.drawContours(img_contour, biggest, -1, (255,255,0), 10)
        return biggest
    except:
        biggest = np.array([[[458,  68]], [[139,  74]], [[139, 349]], [[469, 337]]])
        return biggest

def Reorder(mypoints):
    try:
        mypoints = mypoints.reshape((4, 2))
        Newpoints = np.zeros((4,1,2), np.int32)
        add = np.sum(mypoints, axis=1)
        diff = np.diff(mypoints, axis=1)
        Newpoints[0] = mypoints[np.argmin(add)]
        Newpoints[1] = mypoints[np.argmin(diff)]
        Newpoints[2] = mypoints[np.argmax(diff)]
        Newpoints[3] = mypoints[np.argmax(add)]
        return Newpoints
    except:
        pass



def GetWarp(img, biggest):
    try:
        biggest = Reorder(biggest)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0,0], [FRAMEWIDTH, 0], [0, FRAMEHEIGHT], [FRAMEWIDTH, FRAMEHEIGHT]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(img, matrix, (FRAMEWIDTH, FRAMEHEIGHT))
        result = result[10:-10, 10:-10]
        return result
    except Exception as e:
        pass

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


while True:
    try:
        success, img = cap.read()
        img = cv2.resize(img, (FRAMEWIDTH, FRAMEHEIGHT))
        img1 = img.copy()
        img_contour = img.copy()
        img = preprocessing(img)
        biggest = GetContours(img)
        if biggest.size !=0:
            imgWarped=GetWarp(img_contour,biggest)
            canny = cv2.Canny(imgWarped, 10, 100)
            kernel = np.ones((5,5))
            dilate = cv2.dilate(canny, kernel, iterations=2)
            imageArray = ([img1, canny], [img_contour, imgWarped])
            # gray = cv2.cvtColor(img_contour, cv2.COLOR_BGR2GRAY)
            # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            # text = pytesseract.image_to_string(thresh)
            # print(text)
            cv2.imshow("ImageWarped", imgWarped)
        else:
            zeros = np.zeros_like(img)
            imageArray = ([img_contour, canny], [zeros, zeros])
        stackedImages = stackImages(0.6,imageArray)
        cv2.imshow("WorkFlow", stackedImages)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        break