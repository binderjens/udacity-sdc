import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def create_double_plot(file1,file2, desc1, desc2,out):
    fig = plt.figure(dpi=250)

    a=fig.add_subplot(1,2,1)
    img1 = mpimg.imread(file1)
    imgplot = plt.imshow(img1)
    a.set_title(desc1)
    plt.axis('off')

    img2 = mpimg.imread(file2)
    a=fig.add_subplot(1,2,2)
    imgplot = plt.imshow(img2,cmap='gray')
    a.set_title(desc2)

    plt.axis('off')
    plt.savefig(out, bbox_inches='tight')

create_double_plot('./output_images/straight_lines1_1_undistort.jpg',
                   './output_images/straight_lines1_3_binary.jpg',
                   'undistorted image',
                   'binary thresholded',
                   './output_images/straight_lines1_binary_double.png')

create_double_plot('./output_images/test5_1_undistort.jpg',
                   './output_images/test5_3_binary.jpg',
                   'undistorted image',
                   'binary thresholded',
                   './output_images/test5_binary_double.png')

create_double_plot('./output_images/straight_lines1_2_undistort_lines.jpg',
                   './output_images/straight_lines1_5_transformed_lines.jpg',
                   'original undistorted',
                   'warped image',
                   './output_images/straight_lines1_transform_double.png')

create_double_plot('./output_images/straight_lines1_7_windowed_left.jpg',
                   './output_images/straight_lines1_7_windowed_right.jpg',
                   'left lane windows',
                   'right lane windows',
                   './output_images/straight_lines1_windowed.png')

create_double_plot('./output_images/straight_lines1_9_result.jpg',
                   './output_images/test5_9_result.jpg',
                   'straight_lines1',
                   'test5',
                   './output_images/result.png')
