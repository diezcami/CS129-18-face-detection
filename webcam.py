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

	get_objects_from_img(frame, webcam=True)

	# Display the resulting frame
	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()