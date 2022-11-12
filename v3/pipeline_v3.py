import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# PARAMETERS

BLUR_KERNEL_SIZE = 21

CANNY_EDGE_LOW_T = 75
CANNY_EDGE_HIGH_T = 100
SOBEL_L_THRESHOLD = None

PERSPECTIVE_WARP_SRC=np.float32([(0.4,0.3),(0.6,0.3),(0.1,1),(1,1)])
PERSPECTIVE_WARP_DST=np.float32([(0,0), (1, 0), (0,1), (1,1)])

# UTILS
def get_image(file_name, show=False):
  image = cv.imread(file_name)
  image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
  if show:
    plt.imshow(image)
    plt.show()
  return image

# PIPELINE
def get_grayscale(image):
  grayscale = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
  return grayscale

def get_blur(image, kernel=(BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE)):
  blur = cv.medianBlur(image, kernel[0])
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
    abs_sobelx = sobelx #np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
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

def filter_image(img):
  filtered_img = img.copy()

  # increase contrast
  lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)
  l, a, b = cv.split(lab)
  
  clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
  filtered_img = cv.merge((clahe.apply(l), a, b))
  filtered_img = cv.cvtColor(filtered_img, cv.COLOR_LAB2RGB)

  # push whiteish pixels to full white
  thresh = 230
  gray = get_grayscale(img)
  for y in range(img.shape[0]):
    for x in range(img.shape[1]):
      if gray[y, x] > thresh: filtered_img[y, x, :] = [255, 255, 255]
  return filtered_img

def apply_mask(img):
  masked_img = img.copy()
  for y in range(int(img.shape[0] * 0.4)):
    masked_img[y, :] = np.zeros((img.shape[1:]))
  return masked_img

def extract_points(img):
  num_partitions = 8
  clustered_img = np.zeros((img.shape[0], img.shape[1], 3))
  for p in range(num_partitions):
    starty, endy = img.shape[0]*p//num_partitions, img.shape[0]*(p+1)//num_partitions
    vals = img[starty:endy, :]
    points = np.argwhere(vals)[:, 1]
    if points.shape[0] == 0: continue

    clusters, eps, sorted_points = [], 20, np.sort(points)
    cur_point, cur_cluster = sorted_points[0], [sorted_points[0]]
    for point in sorted_points[1:]:
      if point <= cur_point + eps: cur_cluster.append(point)
      else:
          clusters.append(cur_cluster)
          cur_cluster = [point]
      cur_point = point
    clusters.append(cur_cluster)

    midy = (starty + endy) // 2
    for cluster in clusters:
      avgx = sum(cluster) // len(cluster)
      clustered_img = cv.circle(
        clustered_img, 
        (avgx, midy), 
        10, 
        [1, 0, 0], 
        -1)

  return clustered_img

# MAIN

def main(vals):
  fig = plt.figure()
  fig_shape = (4, len(vals))
  # fig.suptitle('Sobel L Thresholds: 100, 125, 150, None')

  for idx, i in enumerate(vals):
    img_title = './v1/photos/lanes{}.jpg'.format(i)
    print("Processing {}".format(img_title))

    image = get_image(img_title)
    ax = plt.subplot2grid(fig_shape, (0, idx))
    ax.imshow(image)
    plt.axis('off')

    blur = get_blur(image, (15, 15))
    filtered_image = filter_image(blur)
    ax = plt.subplot2grid(fig_shape, (1, idx))
    ax.imshow(filtered_image)
    plt.axis('off')
    blur = get_blur(filtered_image, (5, 5))
    
    edges = apply_mask(cv.Canny(blur, CANNY_EDGE_LOW_T, CANNY_EDGE_HIGH_T, L2gradient=True))
    ax = plt.subplot2grid(fig_shape, (2, idx))
    ax.imshow(edges, cmap='gray')
    plt.axis('off')
    
    clustered = extract_points(edges)
    ax = plt.subplot2grid(fig_shape, (3, idx))
    ax.imshow(clustered, cmap='gray')
    plt.axis('off')
  plt.subplots_adjust(left=0,bottom=0,right=1,top=0.95,wspace=0,hspace=0)
  plt.show()

if __name__ == "__main__":
  main(range(2, 9))
  # main([2])