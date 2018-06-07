#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import tensorflow as tf
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

# TEST_IMAGE_PATHS = [os.path.join(Config.image_dir, "ss.jpg")]
TEST_IMAGE_PATHS = [os.path.join(Config.image_dir, "temp.jpg")]

# print TEST_IMAGE_PATHS
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    for image_path in TEST_IMAGE_PATHS:
      image_np = cv2.imread(image_path)
      img_height = image_np.shape[0]
      img_width = image_np.shape[1]

      image_np_expanded = np.expand_dims(image_np, axis=0)

      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

      final_result = Config.bbox_result(img_width, img_height, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, use_normalized_coordinates=True, min_score_thresh = .7)

      for result in final_result:
        class_name = result[0]
        confidence = result[1]
        x_min = int(result[2])
        y_min = int(result[3])
        x_max = int(result[4])
        y_max = int(result[5])
        print result
        _text = class_name + "_" + str(confidence)
        if class_name == "person":
          cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0,0,255), 8)
          cv2.putText(image_np, _text, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

      cv2.imshow("object_detection", image_np)
      cv2.waitKey(0)
