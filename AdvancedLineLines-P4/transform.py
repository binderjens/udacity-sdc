import numpy as np
import cv2

# undistort image
def undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def scale_sobel(img):
    return np.uint8(255*img/np.max(img))

# sobel
def combined_thresh(img, hls_s_thresh=(180,255), hls_l_thresh=(120,255), luv_l_thresh=(225,255), lab_b_thresh=(155,200)):

    # get hue lightness and saturation of the image
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    luv = cv2.cvtColor(img,cv2.COLOR_BGR2LUV).astype(np.float)
    lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB).astype(np.float)
    
    # yellow & white
    hls_s_channel = hls[:,:,2]
    # detects shadow
    hls_l_channel = hls[:,:,1]
    # white
    luv_l_channel = luv[:,:,0]
    # yellow
    lab_b_channel = lab[:,:,2]
   
    # Threshold color channel
    hls_s_binary = np.zeros_like(hls_s_channel)
    hls_s_binary[(hls_s_channel >= hls_s_thresh[0]) & (hls_s_channel <= hls_s_thresh[1])] = 1
    
    hls_l_binary = np.zeros_like(hls_l_channel)
    hls_l_binary[(hls_l_channel >= hls_l_thresh[0]) & (hls_l_channel <= hls_l_thresh[1])] = 1
    
    luv_l_binary = np.zeros_like(luv_l_channel)
    luv_l_binary[(luv_l_channel >= luv_l_thresh[0]) & (luv_l_channel <= luv_l_thresh[1])] = 1
    
    lab_b_binary = np.zeros_like(lab_b_channel)
    lab_b_binary[(lab_b_channel >= lab_b_thresh[0]) & (lab_b_channel <= lab_b_thresh[1])] = 1
    
    combined_binary = np.zeros_like(hls_s_binary)
    #combined_binary[(s_binary > 0) | (combined > 0) | (l_binary > 0)] = 1
    combined_binary[( (hls_s_binary > 0) | (luv_l_binary > 0) | (lab_b_binary > 0) )] = 1

    return combined_binary

# perspective transform
def transform(img,M):
    img_size = (img.shape[1], img.shape[0])
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    return warped
