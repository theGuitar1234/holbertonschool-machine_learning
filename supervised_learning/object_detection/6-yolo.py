#!/usr/bin/env python3
'''
Initialize Yolo
'''

from tensorflow import keras as K
import numpy as np
import glob
import cv2


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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        '''
        boxes: a list of numpy.ndarrays of shape
        (grid_height, grid_width, anchor_boxes, 4) containing the
        processed boundary boxes for each output, respectively

        box_confidences: a list of numpy.ndarrays of shape
        (grid_height, grid_width, anchor_boxes, 1) containing the box
        confidences for each output, respectively

        box_class_probs: a list of numpy.ndarrays of shape
        (grid_height, grid_width, anchor_boxes, classes) containing the
        box's class probabilities for each output, respectively

        Returns a tuple of (filtered_boxes, box_classes, box_scores):
            filtered_boxes: a numpy.ndarray of shape (?, 4) containing all
            of the filtered bounding boxes:

                ? => the number of boxes that met the filtering criteria

                4 => (x1, y1, x2, y2)

            box_classes: a numpy.ndarray of shape (?,) containing the class
            number that each box in filtered_boxes predicts, respectively
            box_scores: a numpy.ndarray of shape (?) containing the box
            scores for each box in filtered_boxes, respectively
        '''
        box_scores = [box_confidences[i] * box_class_probs[i]
                      for i in range(len(box_confidences))]
        box_classes = [np.argmax(box_scores[i], axis=-1)
                       for i in range(len(box_scores))]
        box_scores = [np.max(box_scores[i], axis=-1)
                      for i in range(len(box_scores))]
        filtered_boxes = [boxes[i][box_scores[i] >= self.class_t]
                          for i in range(len(boxes))]
        box_classes = [box_classes[i][box_scores[i] >= self.class_t]
                       for i in range(len(box_classes))]
        box_scores = [box_scores[i][box_scores[i] >= self.class_t]
                      for i in range(len(box_scores))]
        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        '''
        filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of
        the filtered bounding boxes:

            ? => the number of boxes that met the filtering criteria

            4 => (x1, y1, x2, y2)
        box_classes: a numpy.ndarray of shape (?,) containing the class
        number that each box in filtered_boxes predicts, respectively
        box_scores: a numpy.ndarray of shape (?) containing the box scores
        for each box in filtered_boxes, respectively
        Returns a tuple of (box_predictions, predicted_box_classes,
        predicted_box_scores):
            box_predictions: a numpy.ndarray of shape (?, 4) containing all
            of the predicted bounding boxes ordered by class and box score
            (highest to lowest):

                ? => the number of boxes that will be kept

                4 => (x1, y1, x2, y2)
            predicted_box_classes: a numpy.ndarray of shape (?,) containing the
            class number for box_predictions, respectively
            predicted_box_scores: a numpy.ndarray of shape (?) containing the
            box scores for box_predictions, respectively
        '''
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []
        for c in set(box_classes):
            idx = np.where(box_classes == c)
            b = filtered_boxes[idx]
            s = box_scores[idx]
            sorted_indices = np.argsort(s)[::-1]
            b = b[sorted_indices]
            s = s[sorted_indices]
            while len(b) > 0:
                box_predictions.append(b[0])
                predicted_box_classes.append(c)
                predicted_box_scores.append(s[0])
                if len(b) == 1:
                    break
                x1 = np.maximum(b[0, 0], b[1:, 0])
                y1 = np.maximum(b[0, 1], b[1:, 1])
                x2 = np.minimum(b[0, 2], b[1:, 2])
                y2 = np.minimum(b[0, 3], b[1:, 3])
                intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
                area1 = (b[0, 2] - b[0, 0]) * (b[0, 3] - b[0, 1])
                area2 = (b[1:, 2] - b[1:, 0]) * (b[1:, 3] - b[1:, 1])
                union = area1 + area2 - intersection
                ious = intersection / union
                b = b[1:][ious < self.nms_t]
                s = s[1:][ious < self.nms_t]
        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)
        return box_predictions, predicted_box_classes, predicted_box_scores

    def load_images(self, folder_path):
        '''
        folder_path is a string representing the path to the folder holding
        all the images to predict upon

        Returns a tuple of (images, image_paths):
            images: a list of images as numpy.ndarrays
            image_paths: a list of paths to the individual images in images
        '''

        images = []
        image_paths = glob.glob(folder_path + '/*')
        for path in image_paths:
            img = cv2.imread(path)
            images.append(img)
        return images, image_paths

    def preprocess_images(self, images):
        '''
        images is a list of images as numpy.ndarrays

        Returns a tuple of (pimages, image_shapes):
            pimages: a numpy.ndarray of shape
            (ni, input_h, input_w, 3) containing the preprocessed images,
            respectively:

                ni => the number of images

                input_h => the input height for the Darknet model

                input_w => the input width for the Darknet model

                3 => number of color channels

            image_shapes: a numpy.ndarray of shape (ni, 2) containing the
            original height and width of the images, respectively
        '''
        pimages = []
        shapes = []
        input_h = self.model.input.shape[2]
        input_w = self.model.input.shape[1]
        for i in images:
            img_shape = i.shape[0], i.shape[1]
            shapes.append(img_shape)
            image = cv2.resize(i, (input_w, input_h),
                               interpolation=cv2.INTER_CUBIC)
            image = image / 255
            pimages.append(image)
        pimages = np.array(pimages)
        image_shapes = np.array(shapes)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        '''
        image: a numpy.ndarray containing an unprocessed image

        boxes: a numpy.ndarray of shape (?, 4) containing the predicted
        bounding boxes for the image

        box_classes: a numpy.ndarray of shape (?,) containing the class
        number for each box in boxes, respectively

        box_scores: a numpy.ndarray of shape (?) containing the box scores
        for each box in boxes, respectively

        file_name: the file path where the original image is stored

        Displays the original image with all bounding boxes, class names,
        and box scores drawn on it. The window should be titled with file_name
        '''
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            class_name = self.class_names[box_classes[i]]
            score = box_scores[i]
            label = '{} {:.2f}'.format(class_name, score)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2)
        cv2.imshow(file_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
