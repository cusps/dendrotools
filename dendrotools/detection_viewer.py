import os
import shutil

from PIL import Image, ImageDraw
from json import load


def create_all_previews(json_folder, viewer_folder_path, images_folder):
    pass

def create_view(json_path, viewer_folder_path=None, image_path=None):
    json_dict = load(open(json_path, 'r'))
    if not image_path:
        image_path = json_dict['filename']
    if not viewer_folder_path:
        viewer_folder_path = '{}/image_previews'.format((os.path.dirname(image_path)))

    image_name = os.path.basename(os.path.normpath(image_path))
    image_path = shutil.copyfile(image_path, '{}/{}'.format(viewer_folder_path, image_name))

    with Image.open(image_path) as img:
        width = img.size[0]
        height = img.size[1]
        for detection in json_dict['objects']:
            detection_width = detection['relative_coordinates']['width']*width
            detection_height = detection['relative_coordinates']['height']*height
            detection_x = detection['relative_coordinates']['center_x']*width - detection_width/2
            detection_y = detection['relative_coordinates']['center_y']*height - detection_height/2
            rect = [(detection_x, detection_y + detection_height), (detection_y, detection_x + detection_width)]

            draw = ImageDraw.Draw(img)
            draw.rectangle(rect, outline="red")
