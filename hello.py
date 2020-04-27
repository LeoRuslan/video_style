import numpy as np
import imutils
import time
import cv2
import os

full_path_to_video = '/Users/ruslanpalchuk/Documents/Work/github_project/video_style/input/111.MOV'
full_path_to_output = ''

model = 'models/eccv16/la_muse.t7'

vs = cv2.VideoCapture(full_path_to_video)
writer = None

# count of frames
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

print(total)

while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break


	(h, w) = frame.shape[:2]

	start = time.time()

	end = time.time()

	if writer is None:
		# initialize our video writer
		# fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		fourcc = cv2.VideoWriter_fourcc(*'MP4V')
		writer = cv2.VideoWriter(full_path_to_output, fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))

	# write the output frame to disk
	writer.write(frame)
