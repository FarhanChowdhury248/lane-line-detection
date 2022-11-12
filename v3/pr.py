import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def get_image(file_name, show=False):
  image = cv.imread(file_name)
  image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
  if show:
    plt.imshow(image)
    plt.show()
  return image

def apply_mask(img):
  masked_img = img.copy()
  for y in range(int(img.shape[0] * 0.4)):
    masked_img[y, :] = np.zeros((img.shape[1:]))
  return masked_img

vals = range(2, 9)
fig = plt.figure()
fig_shape = (3, len(vals))

for idx, i in enumerate(vals):
  img_title = './v1/photos/lanes{}.jpg'.format(i)
  print("Processing {}".format(img_title))
  
  img = apply_mask(get_image(img_title))
  ax = plt.subplot2grid(fig_shape, (0, idx))
  ax.imshow(img)
  plt.axis('off')

  lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)
  l,a,b = cv.split(lab)
  
  ax = plt.subplot2grid(fig_shape, (1, idx))
  ax.imshow(l, cmap="gray")
  plt.axis('off')

  print(np.max(l), np.min(l))
  filtered_l = l/255
  print(np.max(filtered_l), np.min(filtered_l))
  maxVal = np.max(filtered_l)
  for y in range(l.shape[0]):
    for x in range(l.shape[1]):
      filtered_l[y, x] = filtered_l[y, x] + 1 - maxVal
      filtered_l[y, x] = filtered_l[y, x]**2
      # filtered_l[y, x] = filtered_l[y, x] if filtered_l[y, x] > 0.8 else 0
  # filtered_l = l*255

  ax = plt.subplot2grid(fig_shape, (2, idx))
  ax.imshow(filtered_l, cmap="gray")
  plt.axis('off')

plt.subplots_adjust(left=0,bottom=0,right=1,top=0.95,wspace=0,hspace=0)
plt.show()