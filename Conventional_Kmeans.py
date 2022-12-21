import numpy as np
import cv2 as cv

img = cv.imread('./test_images_2/foto_mat (1).jpg')

image = img.reshape((-1,3))
K = 6

# convert to np.float32
image = np.float32(image)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, label, center = cv.kmeans(image, K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

# -=-=-=- AFBEELDINGEN OPSLAAN -=-=-=- #
cv.imwrite("./result_conventional/result1_Kmeans.jpg", res2)