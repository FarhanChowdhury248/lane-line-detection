import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# PARAMETERS
VAL = 1
IMAGE_FILE_NAME = './v1/photos/lanes{}.jpg'.format(VAL)
IMAGE_DEST_NAME = './v1/results/lanes{}.png'.format(VAL)

BLUR_KERNEL_SIZE = 5

CANNY_EDGE_LOW_T = 50
CANNY_EDGE_HIGH_T = 200
SOBEL_L_THRESHOLD = 150

PERSPECTIVE_WARP_SRC=np.float32([(0.4,0.3),(0.6,0.3),(0.1,1),(1,1)])
PERSPECTIVE_WARP_DST=np.float32([(0,0), (1, 0), (0,1), (1,1)])

# UTILS
def get_image(file_name=IMAGE_FILE_NAME, show=False):
  image = cv.imread(file_name)
  image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
  if show:
    plt.imshow(image)
    plt.show()
  return image

# PIPELINE
def get_grayscale(image, show=False):
  grayscale = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
  if show:
    plt.imshow(grayscale, cmap='gray')
    plt.show()
  return grayscale

def get_blur(image, kernel=(BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), show=False):
  blur = cv.GaussianBlur(image, kernel, 0)
  if show:
    plt.imshow(blur, cmap='gray')
    plt.show()
  return blur

def get_sobel(img, s_thresh=(100, 255), sx_thresh=(CANNY_EDGE_LOW_T, CANNY_EDGE_HIGH_T), l_thresh=SOBEL_L_THRESHOLD, show=False):
    # Convert to HLS color space and separate the V channel
    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS).astype(float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]

    # increase contrast
    min_l = l_thresh
    if min_l is not None: l_channel = np.clip(l_channel, None, min_l)
    # plt.imshow(l_channel, cmap='gray')
    # plt.show()

    # Sobel x
    sobelx = cv.Sobel(l_channel, cv.CV_64F, 1, 1) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    if show:
      plt.imshow(combined_binary, cmap='gray')
      plt.show()
    return combined_binary

def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist

def sliding_window(img, nwindows=9, margin=150, minpix = 1, draw_windows=True):
    left_a, left_b, left_c = [],[],[]
    right_a, right_b, right_c = [],[],[]

    left_fit_= np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img))*255

    histogram = get_hist(img)
    # find peaks of left and right halves
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    
    # Set height of windows
    window_height = int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        if draw_windows == True:
            cv.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (100,255,255), 3) 
            cv.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (100,255,255), 3) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])
    
    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])
    
    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])
    
    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2]
    right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]
    
    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty

def draw_lanes(img, left_fit, right_fit):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    color_img = np.zeros_like(img)
    
    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right))
    
    cv.fillPoly(color_img, np.int_(points), (0,200,255))
    inv_perspective = perspective_warp(color_img, PERSPECTIVE_WARP_DST, PERSPECTIVE_WARP_SRC)
    inv_perspective = cv.addWeighted(img, 1, inv_perspective, 0.7, 0)
    return inv_perspective

def perspective_warp(img, src, dst, show=False):
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src = src * img_size
    dst_size = (img.shape[1], img.shape[0])
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv.warpPerspective(img, M, dst_size)

    if show:
      plt.imshow(warped, cmap='gray')
      plt.show()
    return warped

# MAIN

def main_sobel(vals=range(1, 10)):
  fig = plt.figure()
  SOBEL_VALS = [100, 125, 150, 200, None]
  fig_shape = (1+len(SOBEL_VALS), len(vals))
  fig_title = 'Sobel L Thresholds: 100, 125, 150, 200, None        Canny Thresh: ({}, {})'.format(
    CANNY_EDGE_LOW_T, 
    CANNY_EDGE_HIGH_T)
  fig.suptitle(fig_title)

  for idx, i in enumerate(vals):
    img_title = './v1/photos/lanes{}.jpg'.format(i)
    print("Processing {}".format(img_title))

    image = get_image(img_title)
    ax = plt.subplot2grid(fig_shape, (0, idx))
    ax.imshow(image)
    plt.axis('off')

    # grayscale = get_grayscale(image)
    # blur = get_blur(grayscale)
    for idx2, st in enumerate(SOBEL_VALS):
      edges = get_sobel(image, l_thresh=st)
      warped = perspective_warp(edges, PERSPECTIVE_WARP_SRC, PERSPECTIVE_WARP_DST)
      try:
        out_img, curves, lanes, ploty = sliding_window(warped)
        img_ = draw_lanes(image, curves[0], curves[1])
        ax = plt.subplot2grid(fig_shape, (idx2+1, idx))
        ax.imshow(img_, cmap='hsv')
        plt.axis('off')
      except:
        print('\tError')
  plt.subplots_adjust(left=0,bottom=0,right=1,top=0.95,wspace=0,hspace=0)
  plt.show()

def main(vals):
  fig = plt.figure()
  fig_shape = (3, len(vals))
  # fig.suptitle('Sobel L Thresholds: 100, 125, 150, None')

  for idx, i in enumerate(vals):
    img_title = './v1/photos/lanes{}.jpg'.format(i)
    print("Processing {}".format(img_title))

    image = get_image(img_title)
    ax = plt.subplot2grid(fig_shape, (0, idx))
    ax.imshow(image)
    plt.axis('off')

    # grayscale = get_grayscale(image)
    # blur = get_blur(grayscale)
    edges = get_sobel(image)
    ax = plt.subplot2grid(fig_shape, (1, idx))
    ax.imshow(edges, cmap='gray')
    plt.axis('off')
    warped = perspective_warp(edges, PERSPECTIVE_WARP_SRC, PERSPECTIVE_WARP_DST)
    out_img, curves, lanes, ploty = sliding_window(warped)
    img_ = draw_lanes(image, curves[0], curves[1])
    ax = plt.subplot2grid(fig_shape, (2, idx))
    ax.imshow(img_, cmap='hsv')
    plt.axis('off')
  plt.subplots_adjust(left=0,bottom=0,right=1,top=0.95,wspace=0,hspace=0)
  plt.show()

if __name__ == "__main__":
  main_sobel(range(2, 9))
  # main([2])