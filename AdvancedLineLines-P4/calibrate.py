import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import pickle

num_x = 9 # 9 corners in x-direction
num_y = 6 # 6 corners in y-direction
size_chessboard=(num_x,num_y)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((num_y*num_x,3), np.float32)
objp[:,:2] = np.mgrid[0:num_x, 0:num_y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

img_size = []

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)

    img_size = (img.shape[1], img.shape[0])
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(img, size_chessboard, None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        # save some exampled pics
        img = cv2.drawChessboardCorners(img, size_chessboard, corners, ret)
        folder=fname.split('\\')
        name=folder[1]
        folder=folder[0]
        outname = 'output_images\\' + name
        cv2.imwrite(outname,img)
        
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

with open('calibration.pkl', 'wb') as f:
    pickle.dump([mtx, dist], f)