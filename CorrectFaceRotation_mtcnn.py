import numpy as np
from mtcnn import MTCNN
import cv2
detector = MTCNN()


def CorrectRotatedFaces(filename):
    img = cv2.imread(filename)
    cv2.imshow('Orig_Image',img)
    cv2.waitKey(0)
    faces = detector.detect_faces(img)
    windowName = 'Detected and corrected Face '
    count_faces = 0
    for face in faces:
        x, y, w, h = face["box"]
        count_faces += 1
        roi_color = img[y:y+h, x:x+w]
        width, height = roi_color[:,:,0].shape
        x1, y1 = face["keypoints"]["left_eye"]
        x2, y2 = face["keypoints"]["right_eye"]
        ang_rad = np.arctan((y2-y1)/(x2-x1))
        ang_deg = 180*ang_rad/np.pi
        roi_color = img[y:y+h, x:x+w]
        rotate_matrix = cv2.getRotationMatrix2D(center=((x1+x2)*0.5,(y1+y2)*0.5), angle=ang_deg, scale=1)
        roi_color_corrected = cv2.warpAffine(src=roi_color, M=rotate_matrix, dsize=(width, height))
        cv2.imshow(windowName+str(count_faces),roi_color_corrected)
        cv2.waitKey(0)

    print("Se detectaron y corrigieron %d rostros "%(count_faces))
    cv2.destroyAllWindows()
    
file_name_1 = "./test_image.jpg"
CorrectRotatedFaces(file_name_1)

file_name_2 = "./test_image_2.jpg"
CorrectRotatedFaces(file_name_2)

