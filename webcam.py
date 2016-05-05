import numpy as np
import cv2
import time

from imgprocess import *

cap = cv2.VideoCapture(0)

while(True):
	# Delay
	# time.sleep(1)

	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	get_objects_from_img(frame)

	# Display the resulting frame
	# cv2.imshow('Frame',gray)
	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()