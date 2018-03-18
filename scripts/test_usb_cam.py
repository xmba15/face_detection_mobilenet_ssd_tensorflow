#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import tensorflow as tf
from io import StringIO
import cv2

from config import Config

# add frozen models and text file of class names
PATH_TO_CKPT = os.path.join(Config.model_dir, Config.face_model)
PATH_TO_LABELS = os.path.join(Config.model_dir, Config.face_label)

# num of classes
category_index = Config.parse_label_files(PATH_TO_LABELS)

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# print TEST_IMAGE_PATHS
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    list_cam = Config.get_usb_cam()
    if list_cam is not None and list_cam[0] != "":
      cam_index = int(list_cam[0][-1])
      cap = cv2.VideoCapture(cam_index)
      ret, frame = cap.read()
      img_height = frame.shape[0]
      img_width = frame.shape[1]
      print ("Start streaming")

    while (ret):

      image_np_expanded = np.expand_dims(frame, axis=0)

      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

      final_result = Config.bbox_result(img_width, img_height, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, use_normalized_coordinates=True)

      for result in final_result:
        class_name = result[0]
        confidence = result[1]
        x_min = int(result[2])
        y_min = int(result[3])
        x_max = int(result[4])
        y_max = int(result[5])
        # print result
        _text = class_name + "_" + str(confidence)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,0,255), 8)
        cv2.putText(frame, _text, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

      cv2.imshow("object_detection", frame)
      ret, frame = cap.read()
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
