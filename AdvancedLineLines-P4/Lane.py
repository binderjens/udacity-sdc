import numpy as np
from collections import deque
from enum import Enum

class LaneSide(Enum):
    Left = 1
    Right = 2

class Lane:
    def __init__(self, side):
        self.found_last = False
        
        # enum for the side of the lane
        self.side = side
        
        # member variable for polyfit values - store last n values
        self.polyfit_0 = deque(maxlen=10)
        self.polyfit_1 = deque(maxlen=10)
        self.polyfit_2 = deque(maxlen=10)

        self.radius = None

        # last line
        self.x = None
        self.y = None

    def find_lane_with_preknowledge(self,x,y):
        margin = 100
        lane_inds = ((x > (np.mean(self.polyfit_0)*(y**2) + np.mean(self.polyfit_1)*y + np.mean(self.polyfit_2) - margin)) 
                    &(x < (np.mean(self.polyfit_0)*(y**2) + np.mean(self.polyfit_1)*y + np.mean(self.polyfit_2) + margin))) 
        return lane_inds
    
    def find_lane_windowed(self, binary_warped):
        histogram=np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        midpoint = np.int(histogram.shape[0]/2)
    
        # Find the peak of the halves of the histogram
        # These will be the starting point for the left and right lines    
        if self.side == LaneSide.Left:
            x_base = np.argmax(histogram[:midpoint])
        else:
            x_base = np.argmax(histogram[midpoint:]) + midpoint    
    
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        x_current = x_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        lane_inds = []
            
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin
            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]
            # Append these indices to the lists
            lane_inds.append(good_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > minpix:
                x_current = np.int(np.mean(nonzerox[good_inds])) 
    
        # Concatenate the arrays of indices
        lane_inds = np.concatenate(lane_inds)
        return lane_inds
    
    def calculate_new_radius(self):
        ym_per_pix = 30./720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        xvals = self.x
        yvals = self.y
        fit_cr = np.polyfit(yvals*ym_per_pix, xvals*xm_per_pix, 2)
        self.radius = ((1 + (2*fit_cr[0]*np.max(yvals) + fit_cr[1])**2)**1.5) \
                                     /np.absolute(2*fit_cr[0])
        return self.radius
    
    def find_lane_for_frame(self,new_binary_warped):
        nonzero = new_binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        if self.found_last == True:
            lane_inds = self.find_lane_with_preknowledge(nonzerox,nonzeroy)
        else:
            lane_inds = self.find_lane_windowed(new_binary_warped)
        
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]
        
        if np.sum(x) > 0:
            self.found_last = True
            self.x = x
            self.y = y
            fit = np.polyfit(self.x,self.y,2)
        
            self.polyfit_0.append(fit[0])
            self.polyfit_1.append(fit[1])
            self.polyfit_2.append(fit[2])
        else:
            self.found_last = False
        
        return self.found_last
    
    def get_fit_line(self,y):
        ploty = np.linspace(0, y-1, y )
        fitx  = np.mean(self.polyfit_0)*ploty**2 + np.mean(self.polyfit_1)*ploty + np.mean(self.polyfit_2)
        return [fitx, ploty]
        
    