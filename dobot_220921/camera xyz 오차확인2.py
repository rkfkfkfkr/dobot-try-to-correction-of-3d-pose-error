from numpy import array
from keras.models import Sequential
from keras.layers import Dense,LSTM
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

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

from multiprocessing import Process, Pipe, Queue, Value, Array, Lock

import numpy.random as rnd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def segmentaition(frame):

    img_ycrcb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)
    y,cr,cb = cv2.split(img_ycrcb)

    _, cb_th = cv2.threshold(cb, 90, 255, cv2.THRESH_BINARY_INV)
    cb_th = cv2.dilate(cv2.erode(cb_th, None, iterations=2), None, iterations=2)
    #cb_th = cv2.dilate(cb_th, None, iterations=2)

    return cb_th

def get_distance(x, y, imagePoints):
    '''
    imagePoints = np.array([[222,45],
                           [418,41],
                           [595,465],
                           [47,463]],dtype = 'float32')
    
    objectPoints = np.array([[67,0,0], #33
                            [0,0,0],
                            [0,160,0], #23
                            [67,160,0]],dtype = 'float32')
    '''
    objectPoints = np.array([[28,70.2,0], #33
                            [28,80.2,0],
                            [38.5,80.2,0], #23
                            [38,70.2,0]],dtype = 'float32')
    
    fx = float(816.37633064)
    fy = float(856.61634273)
    cx = float( 642.82955787)
    cy = float(364.43094119)
    k1 = float(0.13055711)
    k2 = float(-0.28815168)
    p1 = float(-0.00576008)
    p2 = float(0.00250949)

    cameraMatrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype = 'float64')
    distCoeffs = np.array([k1,k2,p1,p2],dtype = 'float64')

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
    
    px = float(Pw[0])
    py = float(Pw[1])

    #print("px: %f, py: %f" %(px,py))

    return px,py

def get_distance_z(x, y, imagePoints):
    
    objectPoints = np.array([[85,0,0], #33
                            [75,0,0],
                            [75,10,0], #23
                            [85,10,0],],dtype = 'float32')
    
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

def find_ball(frame,cb_th,box_points):

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
                
            px,py = get_distance(center[0], center[1],box_points)
            
            text = " %f , %f" %(px,py)
            cv2.putText(frame,text,center,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    return px,py

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

def main():

    cap1 = cv2.VideoCapture(1)
    cap2 = cv2.VideoCapture(0)

    while(1):

        _,frame1 = cap1.read()
        _,frame2 = cap2.read()

        box_points1 = findArucoMarkers(frame1)
        box_points2 = findArucoMarkers(frame2)

        print(" bp1: %d, bp2: %d" %(len(box_points1), len(box_points2)))
        cv2.line(frame1, (320,0), (320, 600), (255,255,255), 2  )
        cv2.imshow('1',frame1)
        #cv2.imshow('2',frame2)

        if len(box_points2) > 2 and len(box_points1) > 2:
             break
        
        if cv2.waitKey(1) == 27:
            break

    ball_x = []
    ball_y = []

    ball_y2 = []
    ball_z = []

    while(1):

        data = pd.read_excel('camera xyz 오차확인.xlsx', engine='openpyxl')
        data = np.array(data)
        if len(data) == 1:
            data = np.empty((0,6))
        elif len(data) > 9:
            data = np.delete(data,0,axis=1)

        _,frame2 = cap2.read()
        _,frame1 = cap1.read()

        cb_th1 = segmentaition(frame1)
        cb_th2 = segmentaition(frame2)

        px,py = find_ball(frame1,cb_th1,box_points1)
        py2,pz = find_ball_z(frame2,cb_th2,box_points2)

        if px != None and pz != None and len(ball_x) < 10:

            print("px: %f, py: %f, pz: %f" %(px,py,pz))
            #print("py: %f, pz: %f" %(py2,pz))

            ball_x.append(px)
            ball_y.append(py)
            ball_z.append(pz)

            

        else:
            '''
            if len(ball_x) == 10:

                rx = float(input('x: '))
                ry = float(input('y: '))
                rz = float(13)

                for i in range(10):

                    bx = float(ball_x[i])
                    by = float(ball_y[i])
                    bz = float(ball_z[i])

                    data = np.append(data,np.array([[rx,ry,rz,bx,by,bz]]),axis = 0)

                df = pd.DataFrame(data, columns=['rx', 'ry', 'rz', 'bx', 'by', 'bz'])
                df.to_excel('camera xyz 오차확인.xlsx', sheet_name='new_name')

                print(data)
                print('save')

                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                fontlabel = {"fontsize":"large", "color":"gray", "fontweight":"bold"}
                ax.set_xlabel("X", fontdict=fontlabel, labelpad=12)
                ax.set_ylabel("Y", fontdict=fontlabel, labelpad=12)
                ax.set_title("Z", fontdict=fontlabel)

                ax.scatter(ball_x,ball_y,ball_z, color = 'b', alpha = 0.5)
                
                plt.show()
                '''
            ball_x.clear()
            ball_y.clear()
            ball_z.clear()

        #cv2.line(frame1, (320,0), (320, 600), (255,255,255), 2  )
        #cv2.line(frame2, (0,200), (640, 200), (255,255,255), 2  )

        cv2.imshow('cam2',frame2)
        cv2.imshow('cam1',frame1)
        
        if cv2.waitKey(1) == 27:
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

main()
