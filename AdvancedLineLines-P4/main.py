import pickle
import glob
import cv2
import transform
import lane_detection
import numpy as np
import collections
from moviepy.editor import VideoFileClip

# the calibration has been done before and has been serialized as a pkl file
with open('calibration.pkl', 'rb') as f:
    mtx, dist = pickle.load(f)

test_images=glob.glob('test_images/*.jpg')

output_folder='output_images'

src=np.float32([[180,700],
                [1150,700],
                [715,460],
                [570,460]])
dst=np.float32([[200,720],
                [1080,720],
                [1080,0],
                [200,0]])

M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)

left_fit = []
right_fit = []

left_polylines=collections.deque(maxlen=10)
right_polylines=collections.deque(maxlen=10)

def get_save_name(filename,img_suffix):
    name = filename.split('.')[0]
    out_fname= name + '_' + img_suffix + '.jpg'
    return output_folder+'\\'+out_fname

def save_image(img,filename,img_suffix):
    name = get_save_name(filename,img_suffix)
    cv2.imwrite(name,img)

def create_lane_image(undist_img, warped, left_fit, right_fit):
    img_shape=undist_img.shape
    img_size = (img_shape[1], img_shape[0])
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    out_img = np.dstack((warp_zero, warp_zero, warp_zero))
    
    [left_fitx, right_fitx, ploty] = lane_detection.generate_fitting(warp_zero.shape[0],left_fit,right_fit)

    left_line_window = np.array(np.transpose(np.vstack([left_fitx, ploty])))
    right_line_window = np.array(np.flipud(np.transpose(np.vstack([right_fitx, ploty]))))
    line_points = np.vstack((left_line_window, right_line_window))
    cv2.fillPoly(out_img, np.int_([line_points]), [0,255, 0])
    unwarped = cv2.warpPerspective(out_img, M_inv, img_size , flags=cv2.INTER_LINEAR)
    result = cv2.addWeighted(undist_img, 1, unwarped, 0.3, 0)
    return result

def pipeline_img(img, img_name):
    # undistort image
    img = transform.undistort(img,mtx,dist)
    save_image(img,img_name,'undistort')

    # save example image with mask lines
    img_lines = img.copy()
    cv2.polylines(img_lines,[src.astype(int)],True,(0,0,255),3)
    save_image(img_lines,img_name,'undistort_lines')

    # create combined threshold
    img_transform = transform.combined_thresh( img,
                                    sobel_kernel=3,
                                    sobel_thresh=(40,120),
                                    mag_thresh=(50,120),
                                    dir_thresh=(np.pi/6, np.pi/2),
                                    s_thresh=(140,255))
    # save binary image to output folder                                              
    save_image(img_transform*255,img_name,'binary')
    
    # transform image based on src and dst defined above & save
    img_transform = transform.transform(img_transform,M)
    save_image(img_transform*255,img_name,'transform')

    name = get_save_name(out_fname,'lanes_found')
    [left_fit, right_fit, left_lane_inds, right_lane_inds ] = lane_detection.generate_polyline(img_transform, name)

    # create a lane model
    lane_img = create_lane_image(img,img_transform,left_fit,right_fit)
    save_image(lane_img,img_name,'result')

    return lane_img

def pipeline_vid(img):
    global left_fit
    global right_fit
    global left_polylines
    global right_polylines

    # 1. undistort image
    img = transform.undistort(img,mtx,dist)

    # 2. create combined threshold
    img_transform = transform.combined_thresh( img,
                                    sobel_kernel=3,
                                    sobel_thresh=(40,120),
                                    mag_thresh=(50,120),
                                    dir_thresh=(np.pi/6, np.pi/2),
                                    s_thresh=(140,255))

    # 3. transform image based on src and dst defined above & save
    img_transform = transform.transform(img_transform,M)

    [poly_left, poly_right, line_left, line_right] = 
    lane_detection.generate_polyline(img_transform, left_fit, right_fit,left_polylines,right_polylines)
    
    lane_img = create_lane_image(img,img_transform,left_fit,right_fit)

    return lane_img

# # loop over all test images
# for fname in test_images:
#     # load image
#     img = cv2.imread(fname)
#     out_fname = fname.split('\\')[1]

#     out_img=pipeline_img(img,out_fname)

output = 'project_video_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(pipeline_vid)
white_clip.write_videofile(output, audio=False)