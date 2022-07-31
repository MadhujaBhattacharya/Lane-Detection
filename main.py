import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

def roi(image):
    height= image.shape[0]
    width= image.shape[1]
    polygon= np.array([[(int(width/6),height),(int(width/2.5),int(height/1.45)),(int(width/1.9),int(height/1.45)),(int(width/1.3),height)]])
    zeromask= np.zeros_like(image)
    cv2.fillConvexPoly(zeromask,polygon,1)
    roi= cv2.bitwise_and(image,image,mask=zeromask)
    return roi

def getlinecoordinates(frame, lines):
    height= int(frame.shape[0]/1.5)
    slope, yintercept= lines[0], lines[1]
    y1= frame.shape[0]
    y2= int(y1-110)
    x1= int((y1-yintercept)/slope)
    x2= int((y2-yintercept)/slope)
    return np.array([x1,y1,x2,y2])

def getlines(frame,lines):
    copyimage= frame.copy()
    leftline, rightline= [],[]
    roiheight= int(frame.shape[0]/1.5) 
    lineframe= np.zeros_like(frame)
    for line in lines:
        x1,y1,x2,y2= line[0]
        linedata= np.polyfit((x1,y1),(x2,y2),1)
        slope, yintercept= round(linedata[0],8),linedata[1]

        if slope<0:
            leftline.append((slope,yintercept))
        else:
            rightline.append((slope,yintercept))
    
    if leftline:
        leftline_avg= np.average(leftline,axis=0)
        left= getlinecoordinates(frame,leftline_avg)
        try:
            cv2.line(lineframe,(left[0],left[1]),(left[2],left[3]),(255,0,0),2)
        except Exception as e:
            print('Error:',e)
        
    if rightline:
        rightline_avg= np.average(rightline,axis=0)
        right= getlinecoordinates(frame,rightline_avg)
        try:
            cv2.line(lineframe,(right[0],right[1]),(right[2],right[3]),(255,0,0),2)
        except Exception as e:
            print('Error',e)
    return cv2.addWeighted(copyimage,0.8,lineframe,0.8,0.0)

fourcc= cv2.VideoWriter_fourcc('m','p','4','v')
out= cv2.VideoWriter('media/output.mp4', fourcc, 20.0, (640, 380)) 
video= cv2.VideoCapture('media/carDashCam.mp4')
while video.isOpened():
    ret, frame= video.read() 
    cv2.imshow('Original frame',frame)
    grayframe= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    kernel_size=1
    gaussianFrame= cv2.GaussianBlur(grayframe,(kernel_size, kernel_size),kernel_size)
    cv2.imshow('Gaussian Frame', gaussianFrame)
    lower_threshold= 75
    higher_threshold= 160
    edgeFrame= cv2.Canny(gaussianFrame, lower_threshold, higher_threshold)
    cv2.imshow('Edge Frame', edgeFrame)

    roiFrame= roi(edgeFrame)
    lines= cv2.HoughLinesP(roiFrame, rho=1, theta=np.pi/180, threshold=20, lines=np.array([]),minLineLength=10,maxLineGap=180)
    imagewithline= getlines(frame,lines)
    cv2.imshow('Final',imagewithline) 
    out.write(imagewithline)


    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
video.release()
out.release()
cv2.destroyAllWindows()
