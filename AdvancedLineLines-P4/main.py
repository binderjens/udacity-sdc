import pickle
import glob
import cv2
import transform
from Lane import Lane
import numpy as np
from moviepy.editor import VideoFileClip
from Lane import LaneSide
import lane_detection
import os

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

def get_save_name(filename,img_suffix):
    name = filename.split('.')[0]
    out_fname= name + '_' + img_suffix + '.jpg'
    return os.path.join(output_folder,out_fname)

def save_image(img,filename,img_suffix):
    name = get_save_name(filename,img_suffix)
    cv2.imwrite(name,img)

def create_lane_image(undist_img, warped, left, right):
    img_shape = warped.shape
    img_size = (img_shape[1], img_shape[0])
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    out_img = np.dstack((warp_zero, warp_zero, warp_zero))

    [left_line, ploty_left] = left.get_fit_line(warped.shape[0])
    [right_line, ploty_right] = right.get_fit_line(warped.shape[0])

    left_line_window = np.array(np.transpose(np.vstack([left_line, ploty_left])))
    right_line_window = np.array(np.flipud(np.transpose(np.vstack([right_line, ploty_right]))))
    line_points = np.vstack((left_line_window, right_line_window))
    cv2.fillPoly(out_img, np.int_([line_points]), [0,255, 0])
    unwarped = cv2.warpPerspective(out_img, M_inv, img_size , flags=cv2.INTER_LINEAR)
    result = cv2.addWeighted(undist_img, 1, unwarped, 0.3, 0)
    return result

def pipeline_img(img, img_name):
    LeftLane = Lane(LaneSide.Left)
    RightLane = Lane(LaneSide.Right)

    # undistort image
    img = transform.undistort(img,mtx,dist)
    save_image(img,img_name,'undistort')

    # save example image with mask lines
    img_lines = img.copy()
    cv2.polylines(img_lines,[src.astype(int)],True,(0,0,255),3)
    save_image(img_lines,img_name,'undistort_lines')

    # create combined threshold
    img_transform = transform.img_transform = transform.combined_thresh(img)
    
    # save binary image to output folder
    save_image(img_transform*255,img_name,'binary')

    # transform image based on src and dst defined above & save
    img_transform = transform.transform(img_transform,M)
    save_image(img_transform*255,img_name,'transform')

    name = get_save_name(img_name,'lanes_found')

    LeftLane.find_lane_for_frame(img_transform)
    RightLane.find_lane_for_frame(img_transform)
    
    lane_detection.plot_polyline(img_transform, name, LeftLane, RightLane)

    lane_img = create_lane_image(img,img_transform,LeftLane,RightLane)

    save_image(lane_img,img_name,'result')

    return lane_img

#loop over all test images
for fname in test_images:
    # load image
    img = cv2.imread(fname)
    out_fname = fname.split(os.sep)[1]
    out_img=pipeline_img(img,out_fname)

#exit()

LeftLane = Lane(LaneSide.Left)
RightLane = Lane(LaneSide.Right)

def pipeline_vid(img):

    # 1. undistort image
    img = transform.undistort(img,mtx,dist)

    # 2. create combined threshold
    img_transform = transform.combined_thresh(img)

    # 3. transform image based on src and dst defined above & save
    img_transform = transform.transform(img_transform,M)

    # 4. create new polynom values for current image
    found = LeftLane.find_lane_for_frame(img_transform)
    if(found != True):
        LeftLane.find_lane_for_frame(img_transform)
    
    found = RightLane.find_lane_for_frame(img_transform)
    if(found != True):    
        RightLane.find_lane_for_frame(img_transform)

    lane_img = create_lane_image(img,img_transform,LeftLane,RightLane)

    return lane_img

output1 = 'project_video_output.mp4'
output2 = 'challenge_video_output.mp4'
output3 = 'harder_challenge_video_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
clip2 = VideoFileClip("challenge_video.mp4")
clip3 = VideoFileClip("harder_challenge_video.mp4")
white_clip = clip1.fl_image(pipeline_vid)
white_clip.write_videofile(output1, audio=False)
LeftLane.reset()
RightLane.reset()
white_clip = clip2.fl_image(pipeline_vid)
white_clip.write_videofile(output2, audio=False)
exit()
LeftLane.reset()
RightLane.reset()
white_clip = clip3.fl_image(pipeline_vid)
white_clip.write_videofile(output3, audio=False)
