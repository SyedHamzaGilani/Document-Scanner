import cv2
import numpy as np

FRAMEWIDTH = 640
FRAMEHEIGHT = 400
BRIGHTNESS = 100

cap = cv2.VideoCapture('Resources/Getcard5.mp4')
cap.set(3, FRAMEWIDTH)
cap.set(4, FRAMEHEIGHT)
cap.set(10, BRIGHTNESS)


def Preprocessing(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(gray, (7,7), 1)
        threshold1 = cv2.getTrackbarPos("Threshold1", "TrackBars")
        threshold2 = cv2.getTrackbarPos("Threshold2", "TrackBars")
        # threshold1 = 37
        # threshold2 = 167
        canny = cv2.Canny(gaussian, threshold1=threshold1, threshold2= threshold2)
        kernel = np.ones((5,5))
        dilate = cv2.dilate(canny, kernel, iterations=2)
        erode = cv2.erode(dilate, kernel, iterations=1)
        return erode
    except:
        pass



def GetContour(img):
    max_Area = cv2.getTrackbarPos("Max Area", "TrackBars")
    biggest = np.array([])
    contour, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contour:
        area = cv2.contourArea(cnt)
        if area > max_Area:
            # cv2.drawContours(img_contour, cnt, -1, (255,255,0), 5)
            arcLength = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*arcLength, True)
            if area > max_Area and len(approx) == 4:
                biggest = approx
                area = max_Area
                cv2.drawContours(img_contour, biggest, -1, (255,255,0), 10)
    return biggest


def ReorderPoints(mypoints):
    mypoints = mypoints.reshape((4,2))
    NewPoints = np.zeros((4,1,2), np.int32)
    add = np.sum(mypoints, axis = 1)
    NewPoints[0] = mypoints[np.argmin(add)]
    NewPoints[3] = mypoints[np.argmax(add)]
    diff = np.diff(mypoints, axis=1)
    NewPoints[1] = mypoints[np.argmin(diff)]
    NewPoints[2] = mypoints[np.argmax(diff)]
    return NewPoints


def WarpPerspective(img, biggest, threshold):
    biggest = ReorderPoints(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0], [FRAMEWIDTH, 0], [0, FRAMEHEIGHT], [FRAMEWIDTH, FRAMEHEIGHT]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (FRAMEWIDTH, FRAMEHEIGHT))
    result = result[threshold:-threshold, threshold:-threshold]
    return result

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


def TrackBars():
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", FRAMEWIDTH, 120)
    # 37, 167, 55000 docx
    cv2.createTrackbar("Threshold1", "TrackBars", 250, 250, Preprocessing)
    cv2.createTrackbar("Threshold2", "TrackBars", 250, 250, Preprocessing)
    cv2.createTrackbar("Max Area", "TrackBars", 4500, 100000, Preprocessing)


TrackBars()


while True:
    success, img = cap.read()
    if success != False:
        img = cv2.resize(img, (FRAMEWIDTH, FRAMEHEIGHT))
        img_contour = img.copy()
        img = Preprocessing(img)
        biggest = GetContour(img)
        canny = cv2.Canny(img_contour, 37, 167)
        zeros = np.zeros_like(img_contour)
        if biggest.size != 0:
            warp_img = WarpPerspective(img_contour, biggest, 5)
            stack = stackImages(0.5, ([img, canny], [img_contour, warp_img]))
            cv2.imshow("Warp Image", warp_img)
        else:
            stack = stackImages(0.5, ([img, canny], [zeros, zeros]))
            # pass
        cv2.imshow("WorkFlow", stack)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv2.destroyAllWindows()
        break