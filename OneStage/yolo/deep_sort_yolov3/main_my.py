#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os
import datetime
from timeit import time
import warnings
import cv2
import numpy as np
import argparse
from PIL import Image
from yolo_my import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque
from keras import backend

backend.clear_session()
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",help="path to input video", default = "./test_video/test.avi")
ap.add_argument("-c", "--class",help="name of class", default = "person")
ap.add_argument("-o", "--output",help="path to output folder", default = "./output")
ap.add_argument("--interval", help="interval for extraction", default = 20)
ap.add_argument("--fps", help="video fps", default = 15)
args = vars(ap.parse_args())

pts = [deque(maxlen=30) for _ in range(9999)]
warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")

def main(yolo):

    start = time.time()
    #Definition of the parameters
    max_cosine_distance = 0.5 #余弦距离的控制阈值
    nn_budget = None
    nms_max_overlap = 0.3 #非极大抑制的阈值

    counter = []
    #deep_sort
    model_filename = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True
    logTrack_flag = True
    logDetect_flag = True
    extractImg_flag = False
    extractFrame = int(args["interval"]) * int(args["fps"])
    #video_path = "./output/output.avi"
    video_capture = cv2.VideoCapture(args["input"])
    w = int(video_capture.get(3))
    h = int(video_capture.get(4))
    frame_index = -1

    if not os.path.exists(args["output"]):
        os.makedirs(args["output"])

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        output_fps = int(args["fps"])
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # out = cv2.VideoWriter("tmpfile.avi", fourcc, 15, (w, h))
        out = cv2.VideoWriter(args["output"] + "/tmpfile.avi", fourcc, output_fps, (w, h)) 
        #    './output/'+args["input"][43:57]+ "_" + args["class"] + '_output.avi', fourcc, 15, (w, h))
        
        if logDetect_flag:
          list_file = open(args["output"] + '/detection.txt', 'w')
  
    if logTrack_flag:
        list_track_file = open(args["output"] +'/track.txt', 'w')

    if extractImg_flag:
    # folder for extracted images
        path_img = args["output"] + "/images/" 
        if not os.path.exists(path_img):
            os.makedirs(path_img)        

    fps = 0.0

    while True:
        frame_index = frame_index + 1
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()
        row_frame = frame.copy()

       # image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        # print('Image',image)
        boxs,class_names = yolo.detect_image(image)
        features = encoder(frame,boxs)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        i = int(0)
        indexIDs = []
        c = []
        boxes = []
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            #boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr().astype(int)
            boxes.append(bbox)
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3)
            track_id = track.track_id
            cv2.putText(frame,str(track.track_id),(int(bbox[0]), int(bbox[1] -50)),0, 5e-3 * 150, (color),2)
            if len(class_names) > 0:
               class_name = class_names[0]
               cv2.putText(frame, str(class_names[0]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (color),2)

            i += 1
            #bbox_center_point(x,y)
            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
            #track_id[center]
            pts[track.track_id].append(center)
            thickness = 5
            #center point
            cv2.circle(frame,  (center), 1, color, thickness)

	    #draw motion path
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                   continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(frame,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(color),thickness)
                #cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)

        count = len(set(counter))
        cv2.putText(frame, "Total Object Counter: "+str(count),(int(20), int(120)),0, 5e-3 * 200, (0,255,0),2)
        cv2.putText(frame, "Current Object Counter: "+str(i),(int(20), int(80)),0, 5e-3 * 200, (0,255,0),2)
        cv2.putText(frame, "FPS: %f"%(fps),(int(20), int(40)),0, 5e-3 * 200, (0,255,0),3)
        
        # Removed display-related methods by owd
        #cv2.namedWindow("YOLO3_Deep_SORT", 0);
        #cv2.resizeWindow('YOLO3_Deep_SORT', 1024, 768);
        #cv2.imshow('YOLO3_Deep_SORT', frame)

        if writeVideo_flag:
            #save a frame
            out.write(frame)
            if logDetect_flag:
              list_file.write(str(frame_index)+' ')
              if len(boxs) != 0:
                  for i in range(0,len(boxs)):
                      list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
              list_file.write('\n')

        if logTrack_flag: 
            if len(boxes) != 0:
                for i in range(0,len(boxes)):
                    list_track_file.write(str(frame_index)+',')
                    list_track_file.write(str(boxes[i][0])+','+str(boxes[i][1])+','+str(boxes[i][2])+','+str(boxes[i][3])+',')
                    list_track_file.write(str(indexIDs[i]))
                    list_track_file.write('\n')
        
        if extractImg_flag and (frame_index - 100) % extractFrame == 0:
            if len(boxes) != 0: # crop tracked object
                cv2.imwrite(path_img + str(frame_index) + ".jpg", row_frame) # whole image
                label_file = open(path_img + str(frame_index) + ".txt", 'w')
                for i in range(0,len(boxes)):    
                    extract_img = row_frame \
                      [max(0, boxes[i][1]) : min(h, boxes[i][3]), max(0, boxes[i][0]) : min(w, boxes[i][2]),...]
                    cv2.imwrite(path_img + str(frame_index) + "_" + str(indexIDs[i]) + ".jpg", extract_img)
                    x_center = (max(0, boxes[i][0]) + min(w, boxes[i][2])) / 2.0 / w
                    y_center = (max(0, boxes[i][1]) + min(h, boxes[i][3])) / 2.0 / h
                    width = float(min(w, boxes[i][2]) - max(0, boxes[i][0])) / w
                    height = float(min(h, boxes[i][3]) - max(0, boxes[i][1])) / h 
                    label_file.write(str(indexIDs[i]) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height) + '\n')
                label_file.close()

        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        #print(set(counter))

        # Removed UI-related methods by owd
        # Press Q to stop!
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
    print(" ")
    print("[Finish]")
    end = time.time()

    '''
    if len(pts[track.track_id]) != None:
       print(args["input"]+": "+ str(count) + " " + str(class_name) +' Found')
    else:
       print("[No Found]")
    '''


    video_capture.release()

    if writeVideo_flag:
        out.release()
        if logDetect_flag:
          list_file.close()
    if logTrack_flag:
        list_track_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
