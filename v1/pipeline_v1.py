
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('..')

# PARAMETERS
VAL = 10
IMAGE_FILE_NAME = './v1/photos/lanes{}.jpg'.format(VAL)
IMAGE_DEST_NAME = './v1/results/lanes{}.png'.format(VAL)

BLUR_KERNEL_SIZE = 5 # must be odd
CANNY_EDGE_LOW_T = 70
CANNY_EDGE_HIGH_T = 150

HOUGH_LINE_RHO = 3
HOUGH_LINE_THETA = np.pi / 180
HOUGH_LINE_THRESHOLD = 15
HOUGH_LINE_MIN_LINE_LEN = 100
HOUGH_LINE_MAX_LINE_GAP = 60

LINE_SLOPE_MAX = 10
LINE_SLOPE_MIN = 0.5
LINE_SLOPE_BOTTOM = 0.8

# UTILS
def get_image(file_name=IMAGE_FILE_NAME, show=False):
  image = cv.imread(file_name)
  if show:
    plt.imshow(image)
    plt.show()
  return image

def draw_lane_lines(img, lines, color=[255, 0, 0], thickness=7, debug=False):
  x_bottom_pos = []
  x_upper_pos = []
  x_bottom_neg = []
  x_upper_neg = []

  height = img.shape[0]
  y_bottom = height
  y_upper = height * 0.66

  for line in lines:
    for x1,y1,x2,y2 in line:
      slope = (y2 - y1) / (x2 - x1)
      b = y1 - slope*x1

      if debug:
        print("Line Info")
        print("\tCoords: ({}, {}) -> ({}, {})".format(x1,y1,x2,y2))
        print("\tLine: {}, {}".format(slope, b))
        print("\tIs Bottom: {}".format(max(y1, y2) >= LINE_SLOPE_BOTTOM * height))

      # skip lines that do not start near the bottom
      if max(y1, y2) < LINE_SLOPE_BOTTOM * height: continue
      
      # test and filter values to slope
      if LINE_SLOPE_MIN < slope < LINE_SLOPE_MAX:
        x_bottom_pos.append((y_bottom - b) / slope)
        x_upper_pos.append((y_upper - b) / slope)     
      elif -LINE_SLOPE_MAX < slope < -LINE_SLOPE_MIN:
        x_bottom_neg.append((y_bottom - b) / slope)
        x_upper_neg.append((y_upper - b) / slope)

  # a new 2d array with means
  if debug:
    print(len(x_bottom_pos))
    print(len(x_upper_pos))
    print(len(x_bottom_neg))
    print(len(x_upper_neg))
  lines_mean = np.array([[int(np.mean(x_bottom_pos)), int(np.mean(y_bottom)), int(np.mean(x_upper_pos)), int(np.mean(y_upper))], 
                         [int(np.mean(x_bottom_neg)), int(np.mean(y_bottom)), int(np.mean(x_upper_neg)), int(np.mean(y_upper))]])

  # Draw the lines
  for i in range(len(lines_mean)):
    cv.line(img, (lines_mean[i, 0], lines_mean[i, 1]), (lines_mean[i, 2], lines_mean[i, 3]), color, thickness)

def draw_all_lines(img, lines, color=[0, 255, 0], thickness=7):
  for line in lines:
    for x1,y1,x2,y2 in line:
      cv.line(img, (x1, y1), (x2, y2), color, thickness)

# PIPELINE

def get_grayscale(image, show=False):
  grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
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

def get_canny_edges(image, low_t=CANNY_EDGE_LOW_T, high_t=CANNY_EDGE_HIGH_T, show=False):
  edges = cv.Canny(image, low_t, high_t)
  if show:
    plt.imshow(edges, cmap='gray')
    plt.show()
  return edges

def get_masked(image, show=False):
  # mask to bottom half
  mask = np.zeros_like(image)
  height, width = mask.shape
  for y in range(height // 2, height):
    mask[y, :] = np.ones(width)
  masked_image = cv.bitwise_and(image, mask)
  if show:
    plt.imshow(masked_image, cmap='gray')
    plt.show()
  return masked_image

def get_hough_lines(image, rho=HOUGH_LINE_RHO, theta=HOUGH_LINE_THETA, threshold=HOUGH_LINE_THRESHOLD, minLineLen=HOUGH_LINE_MIN_LINE_LEN, maxLineGap=HOUGH_LINE_MAX_LINE_GAP):
  lines = cv.HoughLinesP(
    image, 
    rho, 
    theta, 
    threshold, 
    np.array([]), 
    minLineLength=minLineLen, 
    maxLineGap=maxLineGap)
  return lines

def get_drawn_image(image, lines, dest=None, debug=False):
  rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
  draw_all_lines(rgb_image, lines)
  try:
    draw_lane_lines(rgb_image, lines, debug=debug)
  except:
    print('Error in drawing summary lines')
  plt.imshow(rgb_image)
  if dest: plt.savefig(dest)
  else: plt.show()
  return rgb_image

def do_all():
  for i in range(1, 10):
    img_title = './v1/photos/lanes{}.jpg'.format(i)
    print("Processing {}".format(img_title))

    image = get_image(img_title)

    grayscale = get_grayscale(image)
    blur = get_blur(grayscale)
    canny_edges = get_canny_edges(blur)
    masked = get_masked(canny_edges)
    lines = get_hough_lines(masked)

    result = get_drawn_image(image, lines)

# MAIN

# image = get_image()

# grayscale = get_grayscale(image)
# blur = get_blur(grayscale)
# canny_edges = get_canny_edges(blur)
# masked = get_masked(canny_edges)
# lines = get_hough_lines(masked)

# result = get_drawn_image(image, lines, dest=None, debug=False)
do_all()