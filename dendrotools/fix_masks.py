import os
import numpy as np
from PIL import Image
from pprint import pprint

OLD_MASKS = r'I:\Research\data\Conifer\tracheids\masks_RGB'
NEW_MASKS = r'I:\Research\data\Conifer\tracheids\masks'
ANNOTATED_IMGS = r'I:\Research\data\Conifer\tracheids\training'
UNANNOTATED = r'I:\Research\data\Conifer\tracheids\unannotated'
IN_CVAT = r'I:\Research\data\Conifer\tracheids\in_cvat'


# TODO: find out what exactly this did and document
def fix_masks():
    """Selects only the first number in the 3 digit color for each of the masks."""
    walk = tuple(os.walk(OLD_MASKS))
    for mask_path in walk[0][2]:

        mask = Image.open(os.path.join(OLD_MASKS, mask_path))
        mask = np.array(mask)
        obj_ids = np.array(np.unique(mask.reshape(-1, mask.shape[2]), axis=0))
        obj_ids = obj_ids[1:]
        new_colors = np.zeros(len(obj_ids))
        mask_count = 0
        for i in range(len(new_colors)):
            new_colors[i] = 255 - mask_count
            mask_count += 1
        obj_ids = obj_ids.tolist()
        mask = mask.tolist()

        for i in range(len(mask)):
            for j in range(len(mask[i])):

                for id_num in range(len(obj_ids)):
                    # print(mask[i][j])
                    # print(obj_ids[id_num])
                    if obj_ids[id_num] == mask[i][j]:
                        mask[i][j] = [new_colors[id_num], 0, 0]
                        break

        mask = np.asarray(mask)
        mask = np.dot(mask[..., :3], [1, 0, 0])
        mask = np.asarray(mask)
        img = Image.fromarray(mask)
        # img.show()
        img = img.convert('RGB')
        img.save(os.path.join(NEW_MASKS, mask_path))


def find_unannotated():
    """Goes through each of the smaller photos to see if it has an annotation yet.

    If they are unannotated, they are moved to the unannotated folder.
    """
    files_moved = []
    img_walk = tuple(os.walk(ANNOTATED_IMGS))
    mask_walk = tuple(os.walk(NEW_MASKS))[0][2]
    for img_path in img_walk[0][2]:
        if not "{}.png".format(os.path.basename(img_path).split('.')[0]) in mask_walk:
            os.rename(os.path.join(ANNOTATED_IMGS, img_path), os.path.join(UNANNOTATED, img_path))
            files_moved.append(os.path.join(UNANNOTATED, img_path))

    pprint(files_moved)


def find_annotated():
    """Goes through each of the smaller photos to see if it has an annotation yet.

    If they are annotated, they are moved into the annotated folder.
    """
    files_moved = []
    img_walk = tuple(os.walk(UNANNOTATED))
    mask_walk = tuple(os.walk(NEW_MASKS))[0][2]
    for img_path in img_walk[0][2]:
        if "{}.png".format(os.path.basename(img_path).split('.')[0]) in mask_walk:
            os.rename(os.path.join(UNANNOTATED, img_path), os.path.join(ANNOTATED_IMGS, img_path))
            files_moved.append(os.path.join(ANNOTATED_IMGS, img_path))

    pprint(files_moved)


def find_annotated_in_cvat():
    """Moves images that were marked as 'in_cvat' into annotated if they have a mask."""
    files_moved = []
    mask_walk = tuple(os.walk(NEW_MASKS))[0][2]  # This gets the files, ignores any folders
    in_cvat_walk = tuple(os.walk(IN_CVAT))[0][2]

    for img_path in in_cvat_walk:
        if "{}.png".format(os.path.basename(img_path).split('.')[0]) in mask_walk:
            os.rename(os.path.join(IN_CVAT, img_path), os.path.join(ANNOTATED_IMGS, img_path))
            files_moved.append(os.path.join(ANNOTATED_IMGS, img_path))

    pprint(files_moved)


def find_unannotated_in_cvat():
    """Removes images from unannotated that are now in 'in_cvat'."""
    files_removed = []
    unannotated_walk = tuple(os.walk(UNANNOTATED))[0][2]  # This gets the files, ignores any folders
    in_cvat_walk = tuple(os.walk(IN_CVAT))[0][2]

    for img_path in in_cvat_walk:
        if img_path in unannotated_walk:
            os.remove(os.path.join(UNANNOTATED, img_path))
            files_removed.append(os.path.join(UNANNOTATED, img_path))

    pprint(files_removed)


def organize_training_images():
    """Goes through images in 'in_cvat', 'annotated', and 'training', ensuring they are organized.

    For each image in 'in_cvat', it makes sure it has been deleted from annotated.
    For each image in 'training' that doesn't have an annotated (mask), it puts it in unannotated, and vice versa.
    """

    # As a note to developer, masks are png but training are jpeg.

    print("find_unannotated: \n")
    find_unannotated()
    print("\nfind_annotated: \n")
    find_annotated()
    print("\nfind_annotated_in_cvat: \n")
    find_annotated_in_cvat()
    print("\nfind_unannotated_in_cvat: \n")
    find_unannotated_in_cvat()


organize_training_images()
