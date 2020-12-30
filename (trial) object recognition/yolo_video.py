import numpy as np
import imutils
import time
import cv2
import os 

inputVid = 'videos/cat.mp4'
outputVid = 'output/cat.avi'
yoloPath = 'yolo-coco'
minConfidence = 0.5
threshold = 0.3

labelsPath = 'yolo-coco/coco.names'
LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype='uint8')

weightsPath = 'yolo-coco/yolov3.weights'
configPath = 'yolo-coco/yolov3.cfg'
print("loading yolo...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

vid = cv2.VideoCapture(inputVid)
writer = None
(W, H) = (None, None)

try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vid.get(prop))
	print("The video has {} frames".format(total))
	
except:
	print('Failed to determine number of frames')
	total = -1 

count = 0	
while True:
	count +=1
	(grabbed, frame) = vid.read()
	
	if not grabbed:
		break
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
	
	boxes = []
	confidences = []
	classIDs = []
	
	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			
			if confidence > minConfidence:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height)  = box.astype("int")
				
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
				
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, minConfidence, threshold)
	
	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			
	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(outputVid, fourcc, 30, (frame.shape[1], frame.shape[0]), True)
	if total > 0:
		elap = (end - start)
		if count % 10 == 0:
			print("frame {}/{} took {:.4f} seconds".format(count,total, elap))
			print("ETA: {:.4f}".format(elap * (total - count)))

	writer.write(frame)
# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vid.release()

cap = cv2.VideoCapture(outputVid)
if (cap.isOpened()== False): 
	print("Error opening video stream or file")
while(cap.isOpened()):
	ret, frame = cap.read()
	if ret:
		cv2.imshow('Frame',frame)
	else:
		cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
		
	if cv2.waitKey(60) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows() 
