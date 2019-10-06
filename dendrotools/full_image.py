import os
import json
import math
from subprocess import call
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000

MAX_PIXEL_DIM_WITHOUT_OVERLAP = 5700
MAX_PIXEL_DIM = 6000
DETECTOR_PATH = '/home/bunjake/dev/darknet/build/darknet/x64/darknet'
DETECTOR_COMMAND = './darknet detector test data/obj.data yolo-obj-detect-full.cfg backup/yolo-obj_final.weights -ext_output -out result.json  < data/new_train.txt'
OVERLAP = 300


class FullImage:
    """

    """
    def __init__(self, full_image_path, results_path, detector_path=None):
        self.detector = DETECTOR_PATH if not detector_path else detector_path
        self.path = full_image_path
        self.image_folder = os.path.dirname(full_image_path)
        self.result_path = results_path
        self.name = os.path.basename(os.path.normpath(self.path))
        with Image.open(self.path) as img:
            self.width = img.size[0]
            self.height = img.size[1]
        self.num_sections = None
        self.sections_text = "{}//{}_sections.txt".format(self.image_folder, self.name)
        self.sections_json = "{}//{}_section_results.json".format(self.image_folder, self.name)
        self.section_size = self._calc_section_size()
        self._break_up_image()
        # self._run_detection_sections()

    def _calc_section_size(self):
        self.num_sections = math.ceil(self.width / MAX_PIXEL_DIM_WITHOUT_OVERLAP)
        section_length = self.width / self.num_sections + OVERLAP
        if self.height < MAX_PIXEL_DIM:
            section_height = (0, self.height)
        else:
            bottom = (self.height-MAX_PIXEL_DIM)/2
            section_height = (bottom, self.height - bottom)
        return section_length, section_height

    def _break_up_image(self):
        # include overlap
        self.section_widths = []
        with Image.open(self.path) as full_img:
            with open(self.sections_text, 'w') as sections:
                excess_length = int((self.width - (self.num_sections * self.section_size[0])) / 2)
                leftover_length = self.section_size[0] - excess_length
                increment = self.section_size[0] - OVERLAP
                self.section_widths.append(0)
                self._create_section(full_img, 0, (self.section_widths[0], leftover_length), sections)
                for i in range(1, self.num_sections-1):
                    self.section_widths.append(i * increment - excess_length)
                    self._create_section(full_img, i,
                                         (self.section_widths[i],
                                          self.section_widths[i] + self.section_size[0]), sections)
                self.section_widths.append(self.width - leftover_length)
                self._create_section(full_img, self.num_sections,
                                     (self.section_widths[self.num_sections - 1], self.width),
                                     sections)

    def _create_section(self, full_img, num, width_dims, list_file):
        section_name = self.image_folder + "//{}_section_{}.jpg".format(self.name, (num + 1))
        full_img.crop((
            width_dims[0], self.section_size[1][0],
            width_dims[1], self.section_size[1][1]
        )).save(section_name)
        list_file.write('{}\n'.format(section_name))

    def _run_detection_sections(self):
        sections_results = "{}//{}_section_results.json".format(self.image_folder, self.name)
        cmd = ['.' + DETECTOR_PATH, 'detector', 'test', '.data', '.cfg', '.weights', '-ext_output',
               '-dont_show', '-out', self.sections_json,
               '<', self.sections_text]

        call(cmd)

        full_result = self._merge_sections_detections(sections_results)

        #self._delete_sections()

        self._save_results(full_result)

    def _bb_intersection_over_union(self, box_a, box_b):
        # determine the (x, y)-coordinates of the intersection rectangle
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])

        # compute the area of intersection rectangle
        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
        box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        # return the intersection over union value
        return iou

    def _detection_overlap(self, first, second):
        first_right = first.left_x + first.width
        second_right = second.left_x + second.width
        first_bottom = first.top_y + first.height
        second_bottom = second.top_y + second.height

        overlap_1 = (((first.left_x < second.left_x) and (first_right > second.left_x)) and
                     ((first.top_y > second.top_y) and (first_bottom < second.top_y)))

        overlap_2 = (((second.left_x < first.left_x) and (second_right > first.left_x)) and
                     ((second.top_y > first.top_y) and (second_bottom < first.top_y)))

        if overlap_1 or overlap_2:
            iou = self._bb_intersection_over_union(first.get_box(), second.get_box()) #*******************************#
            min_percent_detection = (first.percent
                                     if first.percent < second.percent
                                     else second.percent)

            return True, iou, min_percent_detection
        return False, None, None

    def _merge_sections_detections(self, sections_results_path):
        sections_results = json.load(open(sections_results_path, 'r'))
        detections = []
        dims = (self.section_size[0], self.section_size[1][1] - self.section_size[1][0])
        for section in sections_results:
            for detection in section["objects"]:
                num_section = section["frame_id"]
                detections.append(Detection(num_section, self.section_widths[num_section],
                                            detection, dims))

        i = 0
        while i < len(detections):
            j = i + 1
            while j < len(detections):
                overlap, iou, min_percent_detection = self._detection_overlap(detections[i],
                                                                              detections[j])
                if overlap:
                    if iou > 50:
                        detections.remove(min_percent_detection)
                j += 1
            i += 1

        json_dict = {"frame_id": 1, "filename": (self.image_folder + self.name), "object": []}
        for detection in detections:
            json_dict["objects"].append(detection.info)

        return [json_dict]

    def _save_results(self, results):
        json.dump(results, open(self.result_path, 'w'))

    def _delete_sections(self):
        for section in open(self.sections_text, 'r'):
            os.remove(section)

        os.remove(self.sections_json)


class Detection:

    def __init__(self, section, offset, detection_info, dims):
        # make placement change here
        # self.section_placement = offset
        self.section_num = section
        self.percent = detection_info['confidence']
        self.info = detection_info
        self.info['relative_coordinates']['height'] *= dims[1]
        self.info['relative_coordinates']['width'] *= dims[0]
        self.height = self.info['relative_coordinates']['height']
        self.width = self.info['relative_coordinates']['width']
        half_width = (.5 * self.width)
        self.info['relative_coordinates']['width'] = (self.info['relative_coordinates']['width'] *
                                                      dims[0] - half_width + offset)
        self.info['relative_coordinates']['height'] = (self.info['relative_coordinates']['height'] *
                                                       dims[1] - half_width)
        self.left_x = detection_info['relative_coordinates']['width']
        self.top_y = detection_info['relative_coordinates']['height']
        self.image_class = detection_info['name']

    def __str__(self):
        return (self.image_class + ": " + str(self.percent) + "%\t"
                + "(left_x:\t" + str(self.left_x) + "\ttop_y:\t"
                + str(self.top_y) + "\twidth:\t" + str(self.width)
                + "\theight:\t" + str(self.height) + ")\n")

    def get_box(self):
        return [self.left_x, self.top_y,
                self.left_x + self.width,
                self.top_y + self.height]
