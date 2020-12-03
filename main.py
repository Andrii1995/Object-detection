import cv2

cap = cv2.VideoCapture(0)

classNames = []
classFile = 'Include\coco.names'
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split('\n')

configPath = 'venv/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'venv/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs,bbox = net.detect(img,confThreshold=0.5)

    if len(classIds) !=0:
        for classId ,confidence, box in zip(classIds.flatten(),conf.flatten(),bbox):
            cv2.rectangle(cap,box,color(0,255,0),4)
            cv2.putText(cap,classNames[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)

    cv2.imshow("Video", img)
    cv2.waitKey(1)