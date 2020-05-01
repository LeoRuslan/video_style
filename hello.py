import numpy as np
import imutils
import time
import cv2

full_path_to_video = '/Users/ruslanpalchuk/Documents/Work/github_project/video_style/input/veremiy.mp4'
full_path_to_output = '/Users/ruslanpalchuk/Documents/Work/github_project/video_style/output/test.avi'

model = '/Users/ruslanpalchuk/Documents/Work/github_project/video_style/models/instance_norm/mosaic.t7'
net = cv2.dnn.readNetFromTorch(model)

vs = cv2.VideoCapture(full_path_to_video)
writer = None

# count of frames
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
except:
	total = -1

print('total =', total)

t = 0
while True:
	print('t =', t)
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# frame = imutils.resize(frame, width=600)
	# frame = imutils.resize(frame)
	(h, w) = frame.shape[:2]

	start = time.time()
	blob = cv2.dnn.blobFromImage(frame, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
	net.setInput(blob)
	output = net.forward()

	output = output.reshape((3, output.shape[2], output.shape[3]))
	print(output.shape)
	output[0] += 103.939
	output[1] += 116.779
	output[2] += 123.680
	output /= 255.0
	output = output.transpose(1, 2, 0)
	# output *= 255
	print(output.shape)

	# cv2.imwrite('output/output_{}.png'.format(t), output*255)

	end = time.time()

	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
		writer = cv2.VideoWriter(full_path_to_output, fourcc, 30, (output.shape[1], output.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))

	# write the output frame to disk
	res = np.uint8(255 * output)
	writer.write(res)
	t += 1
