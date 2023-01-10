import cv2
import numpy as np
import imutils
import threading
import math
import time
import DobotDllType as dType
from numpy.linalg import inv

import cv2.aruco as aruco
import os

def segmentaition(frame):

    img_ycrcb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)
    y,cr,cb = cv2.split(img_ycrcb)

    _, cb_th = cv2.threshold(cb, 90, 255, cv2.THRESH_BINARY_INV)
    cb_th = cv2.dilate(cv2.erode(cb_th, None, iterations=2), None, iterations=2)
    #cb_th = cv2.dilate(cb_th, None, iterations=2)

    return cb_th

def get_distance_z(x, y, imagePoints):
    
    objectPoints = np.array([[86,0,0], #33
                            [76,0,0],
                            [76,10,0], #23
                            [86,10,0],],dtype = 'float32')
    
    fx = float(470.5961)
    fy = float(418.18176)
    cx = float(275.7626)
    cy = float(240.41246)
    k1 = float(0.06950)
    k2 = float(-0.07445)
    p1 = float(-0.01089)
    p2 = float(-0.01516)

    #cameraMatrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype = 'float64')
    #distCoeffs = np.array([k1,k2,p1,p2],dtype = 'float64')

    cameraMatrix = np.array([[470.5961,0,275.7626],[0,418.18176,240.41246],[0,0,1]],dtype = 'float32')
    distCoeffs = np.array([0.06950,-0.07445,-0.01089,-0.01516],dtype = 'float32')
    _,rvec,t = cv2.solvePnP(objectPoints,imagePoints,cameraMatrix,distCoeffs)
    R,_ = cv2.Rodrigues(rvec)
    
    u = (x - cx) / fx
    v = (y - cy) / fy
    Qc = np.array([[u],[v],[1]])
    Cc = np.zeros((3,1))
    Rt = np.transpose(R)
    Qw = Rt.dot((Qc-t))
    Cw = Rt.dot((Cc-t))
    V = Qw - Cw
    k = -Cw[-1,0]/V[-1,0]
    Pw = Cw + k*V
    
    py = float(Pw[0])
    pz = float(Pw[1])

    #print("px: %f, py: %f" %(px,py))

    #print("Pw: ", Pw)

    return py,pz

def find_ball_z(frame,cb_th,box_points):

    cnts = cv2.findContours(cb_th, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    py = None
    pz = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 1:
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
                
            py,pz = get_distance_z(center[0], center[1],box_points)
            
            text = " %f " %(pz)
            cv2.putText(frame,text,center,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            #print("py: %f, pz: %f" %(py,pz))
            #print("\n")

    return py,pz

def find_ball(frame,cb_th):

    cnts = cv2.findContours(cb_th, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    px = None
    py = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 1:
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            print(center)

            px = float(268 - center[0])/4
            py = float(center[1])/4

            #text = " %f , %f" %(px,py)
            #cv2.putText(frame,text,center,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    return px,py

def findArucoMarkers(img, markerSize = 6, totalMarkers=250, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(gray, arucoDict, parameters = arucoParam)

    if draw:
        cv2.aruco.drawDetectedMarkers(img,bboxs)
        #print(len(bboxs))

    if len(bboxs) > 0:
        return bboxs[0][0]
    else:
        return [0,0]

def warped_img(frame):

    img_h = 600  #600
    img_w = 268
        
    src = np.float32([[224,79], [417,79], [640,464], [0,457]])
    dst = np.float32([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]])
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

    warped_img = cv2.warpPerspective(frame, M, (img_w, img_h)) # Image warping

    return warped_img

def error_correction(py,pz):

    th1 = math.atan2(59.68, (232.5 - py))

    real_py = py + pz/math.tan(th1) - 11

    return real_py

def main():

    cap1 = cv2.VideoCapture(1)
    cap2 = cv2.VideoCapture(0)
    
    while(1):

        _,frame2 = cap2.read()
        box_points2 = findArucoMarkers(frame2)

        if len(box_points2) > 2: 
             break

    ball_x = []
    ball_y = []

    ball_y2 = []
    ball_z = []
    
    while(1):

        _,frame1 = cap1.read()
        _,frame2 = cap2.read()

        
        #cv2.circle(frame1, (220,35), 3, (0,0,255), -1)
        #cv2.circle(frame1, (420,30), 3, (0,0,255), -1)
        #cv2.circle(frame1, (0,457), 3, (0,0,255), -1)
        #cv2.circle(frame1, (640,464), 3, (0,0,255), -1)

        #cv2.circle(frame1, (222,78), 3, (0,0,255), -1)
        #cv2.circle(frame1, (417,78), 3, (0,0,255), -1)
        
        
        frame1 = warped_img(frame1)
        
        cb_th1 = segmentaition(frame1)
        cb_th2 = segmentaition(frame2)
        
        px,py = find_ball(frame1,cb_th1)
        py2,pz = find_ball_z(frame2,cb_th2,box_points2)

        if px != None and pz != None:        

            py = error_correction(py,pz)

            print("px: %f, py: %f, pz: %f" %(px,py,pz))
        
        cv2.imshow('frame1',frame1)
        cv2.imshow('frame2', frame2)

        if cv2.waitKey(1) == 27:
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

main()
