import numpy as np
import cv2

# undistort image
def undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def scale_sobel(img):
    return np.uint8(255*img/np.max(img))

# sobel
def combined_thresh(img, sobel_kernel=3, sobel_thresh=(0,255), mag_thresh=(0, 255), dir_thresh=(0, np.pi/2), s_thresh=(0,255)):

    # get hue lightness and saturation of the image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float)
    # l_channel = hls[:,:,1]
    # saturation is the preferred feature
    s_channel = hls[:,:,2]
    
    # for completeness - extract also RGB
    r_channel = img[:,:,0]
    # g_channel = rgb[:,:,1]
    # b_channel = rgb[:,:,2]
    

    sobelx= cv2.Sobel(s_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely= cv2.Sobel(s_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    sobel_abs_x=np.absolute(sobelx)
    sobel_abs_y=np.absolute(sobely)
    
    scaled_sobel_x = scale_sobel(sobel_abs_x)
    scaled_sobel_y = scale_sobel(sobel_abs_y)

    gradx = np.zeros_like(scaled_sobel_x)
    grady = np.zeros_like(scaled_sobel_y)

    gradx[(scaled_sobel_x >= sobel_thresh[0]) & (scaled_sobel_x <= sobel_thresh[1])] = 1
    grady[(scaled_sobel_y >= sobel_thresh[0]) & (scaled_sobel_y <= sobel_thresh[1])] = 1
    
    #mag thresh
    mag_absSobelxy=np.sqrt(np.square(sobelx)+np.square(sobely))
    scaled_mag_sobel=scale_sobel(mag_absSobelxy)
    
    mag_binary = np.zeros_like(scaled_mag_sobel)
    mag_binary[(scaled_mag_sobel >= mag_thresh[0]) & (scaled_mag_sobel <= mag_thresh[1])] = 1
    
    # dir_thresh
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx)) 
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1])] = 1
    
    # combine all four thresholds:
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | 
             ((mag_binary == 1) & (dir_binary == 1) ) ] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    color_binary = np.zeros_like(combined)
    color_binary[(s_binary > 0) | (combined > 0)] = 1

    return color_binary

# perspective transform
def transform(img,src,dst):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    return warped
