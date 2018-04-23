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

create_double_plot('./output_images/straight_lines1_undistort.jpg',
                   './output_images/straight_lines1_binary.jpg',
                   'warped image',
                   'binary thresholded',
                   './output_images/straight_lines1_binary_double.png')
