import numpy as np    
import cv2    
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
import matplotlib.pyplot as plt
from skimage import io, color

#Loading original image
originImg = cv2.imread('c.jpg')
img_lab = cv2.cvtColor(originImg, cv2.COLOR_BGR2LAB)
img_rgb = cv2.cvtColor(originImg, cv2.COLOR_BGR2RGB)

# Shape of original image    
originShape = originImg.shape

# Converting image into array of dimension [nb of pixels in originImage, 3]
# based on r g b intensities    
flatImg=np.reshape(img_lab, [-1, 3])

# Estimate bandwidth for meanshift algorithm    
bandwidth = estimate_bandwidth(flatImg, quantile=0.1, n_samples=100)    
print(bandwidth)
ms = MeanShift(bandwidth = bandwidth*0.6, bin_seeding=True)

# Performing meanshift on flatImg    
ms.fit(flatImg)

# (r,g,b) vectors corresponding to the different clusters after meanshift    
labels=ms.labels_

# Remaining colors after meanshift    
cluster_centers = ms.cluster_centers_    

# Finding and diplaying the number of clusters    
labels_unique = np.unique(labels)    
n_clusters_ = len(labels_unique)    
print("number of estimated clusters : %d" % n_clusters_)    

# Displaying segmented image    
segmentedImg = cluster_centers[np.reshape(labels, originShape[:2])]
#l = segmentedImg[:,:,0]
#a = segmentedImg[:,:,1]
#b = segmentedImg[:,:,2]

seg = cv2.cvtColor(segmentedImg.astype(np.uint8), cv2.COLOR_LAB2RGB)

plt.subplot(121), plt.imshow(img_rgb)
plt.title('original'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(seg)
#plt.subplot(122), plt.imshow(np.reshape(labels, segmentedImg.shape[:2]))
plt.title('Mean shift'), plt.xticks([]), plt.yticks([])
plt.show()