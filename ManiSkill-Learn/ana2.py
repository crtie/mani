import os
import cv2
import numpy as np
a = (0,0,1)
print (a)
R = cv2.Rodrigues(a)
print (R[0])
