'''
General object detection using OpenCV and a pre-trained deep neural network (DNN).
Trained on the COCO dataset to recognise 80 possible objects. Used on a video stream.
'''
import cv2
import numpy as np

def detect():

    config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    frozen_model = "frozen_inference_graph.pb"

    # define a model variable based on the above DNN
    model = cv2.dnn_DetectionModel(frozen_model, config_file)

    model.setInputSize(320, 320)
    model.setInputScale(1.0/127.5)
    model.setInputMean( (127.5, 127.5, 127.5) )
    model.setInputSwapRB(True)


    # read in label names into the list
    labels = []
    labelfile = "coco_label_names.txt"
    with open(labelfile, 'rt') as f:
        labels = f.read().splitlines()

    # video capture
    vid = cv2.VideoCapture("bird_vid.mp4")
    # vid.set(3, 640)
    # vid.set(4, 480)

    if not vid.isOpened():
        raise IOError("cannot open video.")
    
    font = cv2.FONT_HERSHEY_COMPLEX

    success, frame = vid.read()

    while success:
        classIDs, confidences, bboxes = model.detect(frame, confThreshold= 0.55)

        bboxes = list(bboxes)
        confidences = list(np.array(confidences).reshape(1, -1)[0])
        confidences = list(map(float, confidences))

        bboxIdx = cv2.dnn.NMSBoxes(bboxes, confidences, score_threshold= 0.5, nms_threshold= 0.2)

        if len(bboxIdx) != 0:
            for i in range(0, len(bboxIdx)):
                bbox = bboxes[ np.squeeze(bboxIdx[i]) ]
                classConfidence = confidences[ np.squeeze(bboxIdx[i]) ]
                classLabelID = np.squeeze( classIDs[ np.squeeze(bboxIdx[i]) ] )
                classLabel = labels[classLabelID-1]

                displayText = "{} : {:.2f}".format(classLabel, classConfidence)

                x, y, w, h = bbox

                #cv2.rectangle(frame, (start_x,start_ y), width & height tuple ie (x+w, y+h), RGB_color, thickness)
                #cv2.putText(image_to_apply_to, text, text_offset, font, fontScale, color, thickness)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0, 250), thickness=2)
                cv2.putText(
                    frame, displayText, (x+ 10, y-20), font, fontScale=1.3, color=(200, 0, 0), thickness=3
                )  
        
        cv2.imshow("object_detected_frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        (success, frame) = vid.read()


    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect()
