#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

directory_root = os.path.dirname(os.path.realpath(__file__)) + "/../"
model_path = os.path.join(directory_root, "models")
image_path = os.path.join(directory_root, "images")

class Config:

    model_dir = model_path
    image_dir = image_path
    face_model = "frozen_inference_graph_face.pb"
    face_label = "face_label.pbtxt"

    @classmethod
    def parse_label_files(cls, file_path):

        f = open(file_path)
        lines = f.read().splitlines()
        lines = [l.strip() for l in lines]
        f.close()
        try:
            lines.remove("")
        except ValueError:
            pass

        ids = []
        classes = []

        for line in lines:
            if ":" in line:
                current_line_split = line.split(" ")
                try:
                    current_line_split.remove("")
                except ValueError:
                    pass

                if "id:" in current_line_split:
                    ids.append(current_line_split[1])
                if "name:" in current_line_split:
                    classes.append(current_line_split[1].strip("'"))

        result = {}
        for i in range(len(ids)):
            result[i + 1] = {"id": ids[i], "name": classes[i]}

        return result

    @classmethod
    def bbox_result(cls, img_width, img_height, boxes, classes, scores,
                category_index, use_normalized_coordinates = False,
                max_boxes_to_draw = 20, min_score_thresh = .7, agnostic_mode = False):

        import collections
        box_to_display_str_map = collections.defaultdict(list)
        if not max_boxes_to_draw:
            max_boxes_to_draw = boxes.shape[0]
            
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > min_score_thresh:
                box = tuple(boxes[i].tolist())
            if scores is None:
                return None
            else:
                if not agnostic_mode:
                    if classes[i] in category_index.keys():
                        class_name = category_index[classes[i]]['name']
                    else:
                        class_name = 'N/A'
                    display_str = '{}: {}%'.format(
                        class_name,
                        int(100*scores[i]))
                else:
                    display_str = 'score: {}%'.format(int(100 * scores[i]))
            
                box_to_display_str_map[box].append(display_str)
    
        results = []
        for _key, _val in box_to_display_str_map.items():
            y_min, x_min, y_max, x_max = _key
    
            name_confidence = _val[0]
            name_confidence_split = name_confidence.split(":")
            class_name = name_confidence_split[0]
            confidence = int(name_confidence_split[1][:-1])
            if use_normalized_coordinates:
                (x_min, x_max, y_min, y_max) = (x_min * img_width, x_max * img_width,
                                                y_min * img_height, y_max * img_height)

            results.append([class_name, confidence, x_min, y_min, x_max, y_max])  
        
        return results
