import pickle
import glob
import cv2
import transform
from Lane import RightLane, LeftLane
import numpy as np
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
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
    
    left_curve = left.calculate_new_radius()
    right_curve = right.calculate_new_radius()
    mean_curvature = (left_curve+right_curve)/2
    
    # Calculate the vehicle position relative to the center of the lane
    position = np.mean((right_line+left_line)/2)
    distance_from_center = abs((640 - position)*3.7/700) 
    
    # Print radius of curvature on video
    cv2.putText(undist_img, 'Curvature {}(m)'.format(int(mean_curvature)), (120,140), fontFace = 16, fontScale = 2, color=(255,255,255), thickness = 2)
    
    # Print distance from center on video
    if position > 640:
        cv2.putText(undist_img, 'Vehicle is {:.2f}m left of center'.format(distance_from_center), (100,80), fontFace = 16, fontScale = 2, color=(255,255,255), thickness = 2)
    else:
        cv2.putText(undist_img, 'Vehicle is {:.2f}m right of center'.format(distance_from_center), (100,80), fontFace = 16, fontScale = 2, color=(255,255,255), thickness = 2)
    
    result = cv2.addWeighted(undist_img, 1, unwarped, 0.3, 0)
    
    return result

def plot_polyline(binary_warped, name, left_lane, right_lane):

    [left_fitx, ploty] = left_lane.get_fit_line(binary_warped.shape[0])
    [right_fitx, _] = right_lane.get_fit_line(binary_warped.shape[0])
    
    # [left_fitx, right_fitx, ploty] = generate_polyline_from_polynom(binary_warped.shape[0],left_fit,right_fit)
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[left_lane.y, left_lane.x] = [255, 0, 0]
    out_img[right_lane.y, right_lane.x] = [0, 0, 255]
    
    margin = 100
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    
    plt.savefig(name)
    plt.close()

def pipeline_img(img, img_name):
    left_lane = LeftLane()
    right_lane = RightLane()

    left_lane.debug = True
    right_lane.debug = True
    left_lane.name=get_save_name(img_name,'windowed_left')
    right_lane.name=get_save_name(img_name,'windowed_right')

    # undistort image
    img = transform.undistort(img,mtx,dist)
    save_image(img,img_name,'undistort')

    # save example image with mask lines
    img_lines = img.copy()
    cv2.polylines(img_lines,[src.astype(int)],True,(0,0,255),3)
    save_image(img_lines,img_name,'undistort_lines')

    # transform image based on src and dst defined above & save
    img_transform = transform.transform(img,M)
    # save example image with mask lines for transformed image
    img_lines = img_transform.copy()
    cv2.polylines(img_lines,[dst.astype(int)],True,(0,0,255),3)
    save_image(img_lines,img_name,'transformed_lines')

    # create combined threshold
    img_transform = transform.combined_thresh(img_transform)
    
    # save binary image to output folder
    save_image(img_transform*255,img_name,'binary')

    name = get_save_name(img_name,'lanes_found')

    left_lane.find_lane_for_frame(img_transform)
    right_lane.find_lane_for_frame(img_transform)
    
    plot_polyline(img_transform, name, left_lane, right_lane)

    lane_img = create_lane_image(img,img_transform,left_lane,right_lane)

    save_image(lane_img,img_name,'result')

    return lane_img

#loop over all test images
for fname in test_images:
    # load image
    img = cv2.imread(fname)
    out_fname = fname.split(os.sep)[1]
    out_img=pipeline_img(img,out_fname)

#exit()

left_lane = LeftLane()
right_lane = RightLane()

def pipeline_vid(img):

    # 1. undistort image
    img = transform.undistort(img,mtx,dist)

    # 3. transform image based on src and dst defined above & save
    img_transform = transform.transform(img,M)

    # 2. create combined threshold
    img_transform = transform.combined_thresh(img_transform)

    # 4. create new polynom values for current image
    found = left_lane.find_lane_for_frame(img_transform)
    if(found != True):
        left_lane.find_lane_for_frame(img_transform)
    
    found = right_lane.find_lane_for_frame(img_transform)
    if(found != True):    
        right_lane.find_lane_for_frame(img_transform)

    lane_img = create_lane_image(img,img_transform,left_lane,right_lane)

    return lane_img

output1 = 'project_video_output.mp4'

clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(pipeline_vid)
white_clip.write_videofile(output1, audio=False)

exit()

left_lane.reset()
right_lane.reset()
output2 = 'challenge_video_output.mp4'
clip2 = VideoFileClip("challenge_video.mp4")
white_clip = clip2.fl_image(pipeline_vid)
white_clip.write_videofile(output2, audio=False)

exit()

LeftLane.reset()
RightLane.reset()
output3 = 'harder_challenge_video_output.mp4'
clip3 = VideoFileClip("harder_challenge_video.mp4")
white_clip = clip3.fl_image(pipeline_vid)
white_clip.write_videofile(output3, audio=False)
