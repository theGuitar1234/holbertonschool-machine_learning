#!/usr/bin/env python3
'''
Initialize Yolo
'''

from tensorflow import keras as K
import numpy as np


class Yolo:
    '''
    Yolo class
    '''

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        '''
        model_path is the path to where a Darknet Keras model is stored

        classes_path is the path to where the list of class names
        used for the Darknet model, listed in order of index, can be found

        class_t is a float representing the box score threshold
        for the initial filtering step

        nms_t is a float representing the IOU
        threshold for non-max suppression

        anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
        containing all of the anchor boxes:
            outputs is the number of outputs (predictions)
            made by the Darknet model
            anchor_boxes is the number of
            anchor boxes used for each prediction
            2 => [anchor_box_width, anchor_box_height]

        Public instance attributes:
            model: the Darknet Keras model
            class_names: a list of the class names for the model
            class_t: the box score threshold for the initial filtering step
            nms_t: the IOU threshold for non-max suppression
            anchors: the anchor boxes
        '''

        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, z):
        '''
        sigmoid function
        '''
        return 1 / (1 + np.exp(-z))

    def process_outputs(self, outputs, image_size):
        '''
        outputs - a list of ndarrays containing the predictions from
        Darknet model for a single image:
            Each output will have the shape
            (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
                grid_height & grid_width =>
                the height and width of the grid used for the output

                anchor_boxes => the number of anchor boxes used

                4 => (t_x, t_y, t_w, t_h)

                1 => box_confidence

                classes => class probabilities for all classes

        image_size is a numpy.ndarray containing
        the image's original size [image_height, image_width]

        Returns a tuple of (boxes, box_confidences, box_class_probs):
            boxes: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 4)
            containing the processed boundary boxes for each output,
            respectively:
                4 => (x1, y1, x2, y2)
                (x1, y1, x2, y2) should represent the
                boundary box relative to original image

            box_confidences: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 1)
            containing the box confidences for each output, respectively

            box_class_probs: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, classes)
            containing the box's class probabilities for each output,
            respectively
        '''
        boxes = [pred[:, :, :, 0:4] for pred in outputs]
        for ipred, pred in enumerate(boxes):
            for grid_h in range(pred.shape[0]):
                for grid_w in range(pred.shape[1]):
                    bx = ((self.sigmoid(
                        pred[grid_h, grid_w, :, 0]
                        ) + grid_w) / pred.shape[1])

                    by = ((self.sigmoid(
                        pred[grid_h, grid_w, :, 1]
                        ) + grid_h) / pred.shape[0])

                    anchor_tensor = self.anchors[ipred].astype(float)
                    anchor_tensor[:, 0] *= \
                        np.exp(pred[grid_h, grid_w, :,
                               2]) / self.model.input.shape[1]
                    anchor_tensor[:, 1] *= \
                        np.exp(pred[grid_h, grid_w, :,
                               3]) / self.model.input.shape[2]

                    pred[grid_h, grid_w, :, 0] = \
                        (bx - (anchor_tensor[:, 0] / 2)) * \
                        image_size[1]
                    pred[grid_h, grid_w, :, 1] = \
                        (by - (anchor_tensor[:, 1] / 2)) * \
                        image_size[0]
                    pred[grid_h, grid_w, :, 2] = \
                        (bx + (anchor_tensor[:, 0] / 2)) * \
                        image_size[1]
                    pred[grid_h, grid_w, :, 3] = \
                        (by + (anchor_tensor[:, 1] / 2)) * \
                        image_size[0]

        box_confidences = [self.sigmoid(pred[:, :, :,
                                        4:5]) for pred in outputs]

        box_class_probs = [self.sigmoid(pred[:, :, :,
                                        5:]) for pred in outputs]

        return boxes, box_confidences, box_class_probs
