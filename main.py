import cv2
import matplotlib.pyplot as plt
import numpy as np

eye = cv2.CascadeClassifier("haarcascade_eye.xml")
shades_full = cv2.imread("dealwithit.png", cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(shades_full, cv2.COLOR_BGR2GRAY)
gray = 255*(gray < 128).astype(np.uint8)
points = cv2.findNonZero(gray)
x, y, w, h = cv2.boundingRect(points)
shades = shades_full[y:y+h, x:x+w] 
shades_ratio = shades.shape[0] / shades.shape[1] 

def shades_on(img, scale=None, min_nbs=None):
    shades_scale = 1.4
    result = img.copy()
    rects = eye.detectMultiScale(result, scaleFactor=scale, minNeighbors=min_nbs)
    if len(rects):
        rects_arr = np.transpose(np.array(rects))
        first_eye = np.min(rects_arr, 1)
        second_eye = np.max(rects_arr, 1)

        left = first_eye[0]
        top = first_eye[1]
        right = second_eye[0] + second_eye[2]
        width = right - left
        bottom = round(shades_ratio * width) + top
        
        cv2.rectangle(result, (left, top), (right, bottom), (255, 255, 255))

        height = bottom - top
        width_scaled = round((width) * shades_scale)
        height_scaled = round((height) * shades_scale)

        _shades = cv2.resize(shades, (width_scaled, height_scaled), (width_scaled) / shades.shape[1], (height_scaled) / shades.shape[0])
        alpha = _shades[:, :, 3] / 255
        colors = _shades[:, :, :3]
        alpha_mask = np.dstack((alpha, alpha, alpha))
        rect_center = (top + height // 2, left + width // 2)
        top = rect_center[0] - _shades.shape[0] // 2 + 10
        bottom = top + _shades.shape[0] // 2
        left = rect_center[1] - _shades.shape[1] // 2
        right = left + _shades.shape[1]

        result[top: top + height_scaled , left:left + width_scaled] = result[top: top + height_scaled , left:left + width_scaled]  * (1 - alpha_mask) + colors * alpha_mask

    return result


cam = cv2.VideoCapture(0)
ret, frame = cam.read()

while True:
    ret, frame = cam.read()
    if not ret:
        break
    eyes = shades_on(frame, 2, 5)


    cv2.imshow('frame', eyes)
    k = cv2.waitKey(10)
    if k > 0:
        if chr(k) == 'q':
            break