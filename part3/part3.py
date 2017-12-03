import numpy as np
import cv2
import os
import sys

if len(sys.argv) < 2:
    print("please provide a source directory and a value for k")
    exit()
if not os.path.exists("output"):
    os.makedirs("output")

sourcedir = sys.argv[1]
sourcedir = os.fsencode(sourcedir)
n_centroid = int(sys.argv[2])

for imgfile in os.listdir(sourcedir):

    path = os.fsdecode(os.path.join(sourcedir, imgfile))
    img = cv2.imread(path)
    imgshape = img.shape
    img = np.float32(img.reshape((-1,3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret, label, center = cv2.kmeans(img,n_centroid,None,criteria,10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    outp = center[label.flatten()].reshape((img.shape))
    outp = outp.reshape((imgshape))

    outpath = os.path.join("output", os.fsdecode(imgfile))
    cv2.imwrite(outpath , outp)
    cv2.waitKey(0)
