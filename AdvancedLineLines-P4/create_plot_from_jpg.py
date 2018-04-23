import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

file1 = sys.argv[1]
file2 = sys.argv[2]
desc1 = sys.argv[3]
desc2 = sys.argv[4]

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
plt.savefig(sys.argv[5],  bbox_inches='tight')