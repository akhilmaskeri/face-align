import cv2
import numpy as np
import imutils
import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

mid_point = (-1,-1)
dim = (-1, -1)

def rotate(image, angle, pivot):
    rot_mat = cv2.getRotationMatrix2D(pivot, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def mark(img):

    global mid_point, dim

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    # find the face
    for (x,y,w,h) in faces:

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        # find eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        e = [(e[0], e[1], e[2], e[3]) for e in eyes]
        e_1 = (e[0][0]+e[0][2]//2, e[0][1]+e[0][3]//2)
        e_2 = (e[1][0]+e[1][2]//2, e[1][1]+e[1][3]//2)

        # mid point and slope of the eye line
        M = ( (e_1[0]+e_2[0])//2,   (e_1[1]+e_2[1])//2 )
        m1 = (e_1[1] - e_2[1])/(e_1[0]-e_2[0])

        # find angle between the eyeline and x=c (horiz)
        angle = np.arctan(m1)*(180/np.pi)

        rotated_grey = rotate(roi_gray, angle, e_1)
        rotated_color = rotate(roi_color, angle, e_1)

        cv2.circle(rotated_color,M,2,(0,0,255),2)

        # find eyes
        eyes = eye_cascade.detectMultiScale(rotated_grey)
        e = [(e[0], e[1], e[2], e[3]) for e in eyes]
        e_1 = (e[0][0]+e[0][2]//2, e[0][1]+e[0][3]//2)
        e_2 = (e[1][0]+e[1][2]//2, e[1][1]+e[1][3]//2)

        M = ( (e_1[0]+e_2[0])//2,   (e_1[1]+e_2[1])//2 )

        if mid_point == (-1,-1):
            print("first time")
            mid_point = M
            dim = img.shape[:2]
        else:

            dx = mid_point[0] - M[0]
            dy = mid_point[1] - M[1]
            w_ = np.float32([[1, 0, dx], [0, 1, dy]])
            
            rotated_color = cv2.warpAffine(rotated_color, w_, rotated_color.shape[:2] ) 

        rotated_color = cv2.resize(rotated_color,dim)

        # find eyes
        eyes = eye_cascade.detectMultiScale(rotated_grey)
        e = [(e[0], e[1], e[2], e[3]) for e in eyes]
        e_1 = (e[0][0]+e[0][2]//2, e[0][1]+e[0][3]//2)
        e_2 = (e[1][0]+e[1][2]//2, e[1][1]+e[1][3]//2)

        M = ( (e_1[0]+e_2[0])//2,   (e_1[1]+e_2[1])//2 )

        print(mid_point, M, dx,dy, (M[0]+dx,M[1]+dy) )
        cv2.imshow("img",rotated_color)
        time.sleep(0.1)

count = 0

while True:

    if count > 17:
        count = 0

    file_name = "faces/img-%d.png"%(count)
    
    img = cv2.imread(file_name)    

    try:
        print(file_name, end=" ")
        mark(img)
        time.sleep(0.1)
        
    except Exception as e:
        print(e)
        print(file_name)

    count += 1
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break