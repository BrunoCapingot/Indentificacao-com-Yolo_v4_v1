import cv2

def print_hi(name):
 colors = ([0,255,255],[255,255,0],[255,0,0])
 classNames = []
 with open('coco.names', 'r') as name:
  classNames = [cname.strip() for cname in name.readlines()]
 cap = cv2.VideoCapture("andando.mp4")
 net = cv2.dnn.readNet("yolov4-tiny.weights","yolov4-tiny.cfg")

 model = cv2.dnn_DetectionModel(net)
 model.setInputParams(size=(416,416),scale = 1/255)

 while True:
  _,frame = cap.read()
  classes,scores,boxes = model.detect(frame,0.1,0.2)

  for (classId,score,box) in zip(classes,scores,boxes):
   color = colors[int(classId)%len(colors)]
   label = f"{ classNames[0] }:{score}"
   cv2.rectangle(frame,box,color,2)
   cv2.putText(frame,label,(box[0],box[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

   cv2.imshow('detections',frame)
  if cv2.waitKey(1) == 27:
   break
 cap.release()



if __name__ == '__main__':
 print_hi('PyCharm')

