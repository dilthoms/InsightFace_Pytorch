import cv2
import sys
img = cv2.imread(sys.argv[1],0)
img = np.stack((img,)*3,axis=-1)
cv2.imwrite(sys.argv[1],cv2.resize(img,(112,112))[:,:,::-1])
