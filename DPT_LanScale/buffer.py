import cv2
import numpy as np

depth = cv2.imread("/media/home/zyzeng/code/LanScale/DPT/input/gt/rgb_00045/sync_depth_00045.png", -1)


depth = depth.astype(np.float32) / 1000.0

median=np.median(depth)

depth_sum=0
for row in depth:
    for col in row:
        depth_sum+=abs(col-median)

scale=depth_sum/(depth.shape[0]*depth.shape[1])

print("median:",round(median,4))
print("scale:",round(scale,4))