import os
import tkinter as tk
import subprocess
import shutil
import math

from PIL import Image
from tkinter import filedialog

DATA_DIRECTORY = r'I:\Research\data'
SOURCE_IMAGES_DIRECTORY = r'Conifer\source_images\test'
IMAGEJ_SCRIPTS_DIRECTORY = r'imagej_scripts'
IMAGEJ = "I:\Fiji.app\ImageJ-win64.exe"
IMAGEJ_O_TO_CROP_SCRIPT = "imagej_original_to_cropped.py"

# TODO: make config file for this in source files
CORE_SIZE = (3000, 0)
RING_SIZE = (2000, 2000)
VESSEL_SIZE = (500, 500)
CONIFER_SIZE = (400, 80)
CONIFER = 1

"""
Process: make directory for each image and split it up here. make json file that contains the pixels for each 
image piece for the ability to later reconstruct. 

Ideas:
*get images from the elevator website? Might be hard as there are some not wanted for training. Maybe just 
function/script for adding new images from the website
*split into both training for vessel and ring boundary training
*
"""


def make_dirs_from_filenames():
    """Make directories from sources images.

    Source images are the full tiff images.
    """
    root = tk.Tk()
    root.withdraw()

    dirname = os.path.join(DATA_DIRECTORY, SOURCE_IMAGES_DIRECTORY)

    file_list = os.listdir(dirname)

    for file in file_list:
        if file.split('.')[-1] == "tiff":
            new_dirname = file.split('.')[0]
            os.mkdir(os.path.join(dirname, file.split('.')[0]))
            os.replace(os.path.join(dirname, file), os.path.join(dirname, new_dirname, file))


def split_original_to_training():
    """Splits the large original image into the smaller training images.

    Actually calls another function `split_into_parts` to split the original
    image. This function also calls an imagej script to allow the user
    to crop it.
    """
    ImageJ_org_to_crop_macro = os.path.join(os.curdir, IMAGEJ_SCRIPTS_DIRECTORY, IMAGEJ_O_TO_CROP_SCRIPT)


    # dirname = tk.filedialog.askdirectory(parent=root, initialdir="/", title='PLease select a directory')
    if not CONIFER:
        source_image_path = r'I:\Research\data\source_images\test'
    else:
        source_image_path = r'I:\Research\data\Conifer\source_images\test'
    vessel_training_path = r'I:\Research\data\vessels\training'
    ring_training_path = r'I:\Research\data\rings\training'


    # essentially check for new images not in a dir
    make_dirs_from_filenames()

    dir_list = os.listdir(source_image_path)

    for dir in dir_list:
        # checks if this file has its labels yet
        has_jsons = (os.path.exists(os.path.join(source_image_path, dir, "{}_{}.json".format(dir, "vessels"))) +
                     os.path.exists(os.path.join(source_image_path, dir, "{}_{}.json".format(dir, "rings"))) +
                     os.path.exists(os.path.join(source_image_path, dir, "{}_{}.json".format(dir, "core"))) +
                     os.path.exists(os.path.join(source_image_path, dir, "{}_{}.json".format(dir, "core"))))

        # if it doesn't have its labels
        if has_jsons == 0:

            img_full_path = os.path.join(source_image_path, dir, "{}.tiff".format(dir)).replace('\\', "/")

            if os.path.exists(img_full_path):
                if not CONIFER:
                    # split_into_parts(img_full_path, 'core', CORE_SIZE)
                    x = subprocess.check_output([IMAGEJ, "-macro", ImageJ_org_to_crop_macro, img_full_path])
                    core_cropped_img_full_path = img_full_path.replace(
                        os.path.basename(img_full_path),
                        "{}_cropped.tiff".format(os.path.split(os.path.basename(img_full_path))))
                    split_into_parts(core_cropped_img_full_path, 'rings', RING_SIZE)
                    split_into_parts(core_cropped_img_full_path, 'vessels', VESSEL_SIZE)
                else:
                    # x = subprocess.check_output([IMAGEJ, "-macro", ImageJ_org_to_crop_macro, img_full_path])
                    # core_cropped_img_full_path = img_full_path.replace(
                    #     os.path.basename(img_full_path),
                    #     "{}_cropped.tiff".format(os.path.split(os.path.basename(img_full_path))))
                    split_into_parts(img_full_path, 'tracheids', CONIFER_SIZE)

        else:
            if has_jsons < 3 and has_jsons > 0:
                pass
                # TODO: report major error here. may need manual checking, could do some error logging about whats wrong


def split_into_parts(img_full_path, img_type, max_size):
    """Crops the image into parts based on the max size.

    :param str img_full_path: path to the full image
    :param str img_type: either tracheid or vessel image
    :param tuple max_size: max dimension size of the smaller images
    :return:
    """
    if not img_type == 'tracheids':
        training_path = r'I:\Research\data\{}\training'.format(img_type)
    else:
        training_path = r'I:\Research\data\Conifer\{}\training'.format(img_type)

    core = img_type == 'core'

    shutil.copy(img_full_path, training_path)
    training_img_full = os.path.basename(img_full_path)
    if '_cropped' in img_full_path:
        training_img_full_name = training_img_full.split('_cropped.')[0]
    else:
        training_img_full_name = training_img_full.split('.')[0]
    img = Image.open(os.path.join(training_path, training_img_full))
    w, h = img.size

    w_crops = math.ceil(w / max_size[0])
    h_crops = math.ceil(h / max_size[1])

    # TODO: save pixels into json
    for row in range(h_crops):
        if core:
            h_start = 0
            h_end = h
        elif row == h_crops - 1:
            h_start = row * max_size[1]
            h_end = h
        else:
            h_start = row * max_size[1]
            h_end = (row + 1) * max_size[1]

        for col in range(w_crops):
            w_start = col * max_size[0]
            w_end = (col + 1) * max_size[0]

            if col == w_crops - 1:
                w_end = w

            crop_path = os.path.join(training_path,
                                     "{}_{}_{}_{}.tiff".format(training_img_full_name, img_type, row, col))
            shutil.copyfile(os.path.join(training_path, training_img_full), crop_path)
            crop_img = Image.open(crop_path)
            crop_img = crop_img.crop((w_start, h_start, w_end, h_end))
            crop_img.save(os.path.join(training_path, "{}_{}_{}_{}.jpeg".format(training_img_full_name, img_type, row, col)))
            os.remove(crop_path)
    os.remove(os.path.join(training_path, training_img_full))


split_original_to_training()
