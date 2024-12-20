import cv2
import numpy as np
import HandTrackingModule as htm
import autopy
import time

wCam, hCam = 640, 480
plocX, plocY = 0, 0
clocX, clocY = 0, 0
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
wScr, hSrc = autopy.screen.size()
detector = htm.handDetector(maxHands = 1)
frameR = 200
smoothening = 7
last_left_click_time = 0
last_right_click_time = 0
delay_click = 0.12
index_finger_state = 1


def index_finger_flexed():
    global last_left_click_time, index_finger_state
    current_time = time.time()
    if current_time - last_left_click_time >= delay_click:
        autopy.mouse.toggle(autopy.mouse.Button.LEFT, not bool(index_finger_state))
        last_left_click_time = current_time


def rightClick():
    global last_right_click_time
    current_time = time.time()
    if current_time - last_right_click_time >= delay_click:
        autopy.mouse.click(autopy.mouse.Button.RIGHT)
        last_right_click_time = current_time


def moving():
    global plocX, plocY, clocX, clocY
    x1, y1 = rightHandLm[8][1:]
    x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
    y3 = np.interp(y1, (frameR, (wCam - frameR) * 3 / 4), (0, hSrc))

    clocX = plocX + (x3 - plocX) / smoothening
    clocY = plocY + (y3 - plocY) / smoothening
    autopy.mouse.move(clocX, clocY)
    plocX, plocY = clocX, clocY


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    rightHandLm = detector.findPosition(img)

    if len(rightHandLm) != 0:
        fingers = detector.fingersUp()
        if fingers[0] == 1 and fingers[1] == 1:
            moving()
        if fingers[2] != index_finger_state:
            index_finger_state = fingers[2]
            index_finger_flexed()
        if fingers[3] == 0:
            rightClick()

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
