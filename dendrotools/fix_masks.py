import os
import numpy as np
from PIL import Image

OLD_MASKS = r'I:\Research\data\Conifer\tracheids\masks_RGB'
NEW_LOC = r'I:\Research\data\Conifer\tracheids\masks'
ANNOTATED_IMGS = r'I:\Research\data\Conifer\tracheids\training'
UNANNOTATED = r'I:\Research\data\Conifer\tracheids\unannotated'


def fix_masks():
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
        img.save(os.path.join(NEW_LOC, mask_path))


def find_unannotated():
    img_walk = tuple(os.walk(ANNOTATED_IMGS))
    mask_walk = tuple(os.walk(NEW_LOC))[0][2]
    for img_path in img_walk[0][2]:
        if not "{}.png".format(os.path.basename(img_path).split('.')[0]) in mask_walk:
            os.rename(os.path.join(ANNOTATED_IMGS, img_path), os.path.join(UNANNOTATED, img_path))


find_unannotated()