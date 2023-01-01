import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation



BG_COLOR = (192,192,192) #gray

cap = cv2.VideoCapture(0)

with mp_selfie_segmentation.SelfieSegmentation(model_selection = 1) as selfie_segmentation:
    bg_image = None
    while cap.isOpened():
        success,image = cap.read()
        if not success:
            print("No Camera")
            continue
        start = time.time()

        image = cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = selfie_segmentation.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        condition = np.stack((results.segmentation_mask,)*3,axis = -1) >0.1


        if bg_image is None:
            bg_image = cv2.imread('background img/yoruu.png')
            bg_image = cv2.resize(bg_image,(640,480))
        output_image = np.where(condition,image,bg_image)


        end = time.time()
        totalTime = end-start

        fps = 1/totalTime

        cv2.putText(output_image, f'FPS: {int(fps)}', (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("cam",output_image)
        if cv2.waitKey(1) == ord('q'):
            break
