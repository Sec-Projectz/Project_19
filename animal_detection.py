import cv2
import numpy as np
Video = cv2.VideoCapture('wild.mkv')


x, f1 = Video.read()
x, f2 = Video.read()

while Video.isOpened():

    differenceOfFrames = cv2.absdiff(f1, f2)

    grayFrame = cv2.cvtColor(differenceOfFrames, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(grayFrame, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    dilated = cv2.dilate(thresh, None, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(f1, contours, -1, (0, 0, 255), 2)
    cv2.imshow("feed", f1)

    f1 = f2
    x, f2 = Video.read()

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
Video.release()