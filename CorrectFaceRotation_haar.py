import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


def CorrectRotatedFaces(filename):
    img = cv2.imread(filename)
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow('Original Image',img)
    cv2.waitKey(0)

    faces = face_cascade.detectMultiScale(imgGray, 1.1, 3)
    windowName = 'Detected and corrected Face '
    count_faces = 0
    count_eyes = 0
    for (x,y,w,h) in faces:
        count_faces += 1
        roi_gray = imgGray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        width, height = roi_gray.shape
        eyes = eye_cascade.detectMultiScale(roi_gray,1.05,2)
        if len(eyes) == 2:
            count_eyes += 1
            x1 = eyes[0][0]+eyes[0][2]/2
            y1 = eyes[0][1]+eyes[0][3]/2
            x2 = eyes[1][0]+eyes[1][2]/2
            y2 = eyes[1][1]+eyes[1][3]/2
            if x1 > x2: 
                x1_aux = x1
                y1_aux = y1
                x1 = x2
                y1 = y2
                x2 = x1_aux
                y2 = y1_aux
            ang_rad = np.arctan((y2-y1)/(x2-x1))
            ang_deg = 180*ang_rad/np.pi
            roi_color = img[y:y+h, x:x+w]
            rotate_matrix = cv2.getRotationMatrix2D(center=((x1+x2)*0.5,(y1+y2)*0.5), angle=ang_deg, scale=1)
            roi_color_corrected = cv2.warpAffine(src=roi_color, M=rotate_matrix, dsize=(width, height))
            cv2.imshow(windowName+str(count_eyes),roi_color_corrected)
            cv2.waitKey(0)

    print("Se detectaron %d rostros y %d rostros con dos ojos"%(count_faces,count_eyes))
    cv2.destroyAllWindows()

file_name_1 = "./test_image.jpg"
CorrectRotatedFaces(file_name_1)

file_name_2 = "./test_image_2.jpg"
CorrectRotatedFaces(file_name_2)


