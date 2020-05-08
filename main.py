import numpy as np
import imutils
import time
import cv2

full_path_to_video = 'input/veremiy.mp4'
full_path_to_output = 'output/test_25.avi'

model = 'models/instance_norm/mosaic.t7'

def create_video_with_effect(full_path_to_video,
							 full_path_to_output,
							 model):

	writer = None
	cap = cv2.VideoCapture(full_path_to_video)
	net = cv2.dnn.readNetFromTorch(model)

	# TODO: get version of cv2
	# get count of fps
	fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
	# get count of frames
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	duration = frame_count / fps

	print(1 / 0)
	t = 0
	while True:
		print('t =', t)
		# read the next frame from the file
		(grabbed, frame) = cap.read()

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

		output[0] += 103.939
		output[1] += 116.779
		output[2] += 123.680
		output /= 255.0
		output = output.transpose(1, 2, 0)
		# output *= 255
		# print(output.shape)

		# cv2.imwrite('output/output_{}.png'.format(t), output*255)

		end = time.time()

		if writer is None:
			# initialize our video writer
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
			writer = cv2.VideoWriter(full_path_to_output, fourcc, fps, (output.shape[1], output.shape[0]), True)

			# some information on processing single frame
			if frame_count > 0:
				elap = (end - start)
				print("[INFO] single frame took {:.4f} seconds".format(elap))
				print("[INFO] estimated total time to finish: {:.4f}".format(elap * frame_count))

		# write the output frame to disk
		res = np.uint8(255 * output)
		writer.write(res)
		t += 1


create_video_with_effect(full_path_to_video, full_path_to_output, model)
