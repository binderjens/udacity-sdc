import pickle
import matplotlib.pyplot as plt
import glob
import cv2
import transform
import lane_detection
import numpy as np

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

def save_image(img,filename,img_suffix):
    name = filename.split('.')[0]
    out_fname= name + '_' + img_suffix + '.jpg'
    cv2.imwrite(output_folder+'\\'+out_fname,img)

# loop over all test images
for fname in test_images:
    # load image
    img = cv2.imread(fname)
    out_fname = fname.split('\\')[1]

    # undistort image
    img = transform.undistort(img,mtx,dist)
    save_image(img,out_fname,'undistort')

    # save example image with mask lines
    img_lines = img.copy()
    cv2.polylines(img_lines,[src.astype(int)],True,(0,0,255),3)
    save_image(img_lines,out_fname,'undistort_lines')

    # create combined threshold
    img = transform.combined_thresh( img,
                                    sobel_kernel=5,
                                    sobel_thresh=(50,100),
                                    mag_thresh=(50,100),
                                    dir_thresh=(np.pi/6, np.pi/2),
                                    s_thresh=(150,255))
    # save binary image to output folder                                              
    save_image(img*255,out_fname,'binary')
    
    # transform image based on src and dst defined above & save
    img = transform.transform(img,src,dst)
    save_image(img*255,out_fname,'transform')
