from Data.SyntheticData.euclidean_utils import *


class DetectionPostProcessing:
    def __init__(self, bounding_box_decision_methods, classes_names, **kwargs):
        self.bounding_box_decision_methods = bounding_box_decision_methods
        self.classes_names = classes_names
        self.kwargs = kwargs

    def post_process(self, image_paths_to_lines_and_words):
        images_paths_to_bounding_boxes_post = {}
        for k in image_paths_to_lines_and_words.keys():
            images_paths_to_bounding_boxes_post[k] = {'image_size': images_paths_to_bounding_boxes[k]['image_size']}
            for class_name in self.classes_names[1:]:
                images_paths_to_bounding_boxes_post[k][class_name] = []
                bounding_boxes = images_paths_to_bounding_boxes[k][class_name]
                indexes_to_delete = []
                for bounding_box_decision_method in self.bounding_box_decision_methods:
                    for idx1, bb1 in enumerate(bounding_boxes):
                        if idx1 not in indexes_to_delete:
                            indexes_to_delete = self._get_indexes_to_delete(bounding_boxes, idx1, indexes_to_delete, bounding_box_decision_method)
                for idx1, bb1 in enumerate(bounding_boxes):
                    if idx1 not in indexes_to_delete:
                        images_paths_to_bounding_boxes_post[k][class_name].append(bb1)
        return images_paths_to_bounding_boxes_post

    def _get_indexes_to_delete(self, bounding_boxes, index, indexes_to_delete, bounding_box_decision_method):
        if bounding_box_decision_method == 'area_threshold':
            indexes_to_delete = self._area_threshold(bounding_boxes, index, indexes_to_delete)
        if bounding_box_decision_method == 'length_threshold':
            indexes_to_delete = self._length_threshold(bounding_boxes, index, indexes_to_delete)
        if bounding_box_decision_method == 'size_based_nms':
            indexes_to_delete = self._size_based_nms(bounding_boxes, index, indexes_to_delete)
        indexes_to_delete = list(set(indexes_to_delete))
        return indexes_to_delete

    def _area_threshold(self, bounding_boxes, index, indexes_to_delete):
        if area_2D(bounding_boxes[index]) < self.kwargs['area_threshold']:
            indexes_to_delete.append(index)
        return indexes_to_delete

    def _length_threshold(self, bounding_boxes, index, indexes_to_delete):
        if min(bounding_boxes[index][2] - bounding_boxes[index][0] + 1, bounding_boxes[index][3] - bounding_boxes[index][1] + 1) < self.kwargs['length_threshold']:
            indexes_to_delete.append(index)
        return indexes_to_delete

    def _size_based_nms(self, bounding_boxes, index, indexes_to_delete):
        bb1 = bounding_boxes[index]
        for index2, bb2 in enumerate(bounding_boxes):
            if index != index2 and (index not in indexes_to_delete and index2 not in indexes_to_delete) and intersection_2D(bb1, bb2) >= self.kwargs['intersection_thr'] * min(area_2D(bb1), area_2D(bb2)):
                if area_2D(bb1) > area_2D(bb2):
                    indexes_to_delete.append(index2)
                else:
                    indexes_to_delete.append(index)
        return indexes_to_delete

