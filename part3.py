import numpy as np
import cv2
import os


if not os.path.exists('clusteredImages'):
	os.makedirs('clusteredImages')

imgDirStr = './img-src'
imgDir = os.fsencode(imgDirStr)

for eachImg in os.listdir(imgDir):
	imgName = os.fsdecode(os.path.join(imgDir,eachImg))

	img = cv2.imread(imgName)
	z = img.reshape((-1,3))

	z = np.float32(z)
	
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) 
	K = 16 
	ret,label,center=cv2.kmeans(z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((img.shape)) 
	saveFileName = 'clustered' + os.fsdecode(eachImg)

	cv2.imwrite(os.path.join('clusteredImages',saveFileName),res2)
	cv2.waitKey(0)


