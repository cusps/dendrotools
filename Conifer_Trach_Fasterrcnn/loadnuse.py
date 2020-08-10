import torch
import numpy as np
from PIL import Image, ImageDraw
from Conifer_Trach_Fasterrcnn import transforms as T

from Conifer_Trach_Fasterrcnn.start import VesselDataset, get_transform, get_instance_seg_model
from Conifer_Trach_Fasterrcnn import utils




def get_prediction_boxes(image_name, max_tran_size=8100, model=None):
    if model is None:
        model = get_instance_seg_model(2, backbone=0)
        model.load_state_dict(torch.load(r'I:\Research\dendrotools\Conifer_Trach_Fasterrcnn\trach_90.pt'))
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    img, _ = T.ToTensor()(Image.open(image_name).convert("RGB"), None)

    model.transform.max_size = max_tran_size
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])
    bboxes = prediction[0]['boxes'].cpu()
    return bboxes


def test_model():
    dataset_test = VesselDataset('../../data/Conifer/tracheids', get_transform(train=False))
    dataset = VesselDataset('../../data/Conifer/tracheids', get_transform(train=False))
    indices = torch.randperm(len(dataset)).tolist()
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    model = get_instance_seg_model(2, backbone=0)

    model.load_state_dict(torch.load('trach_90.pt'))
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    img, _ = T.ToTensor()(Image.open(r"I:\Research\herotest_pmc10b_1571-90_5x_03-12-2020_18-14-1test2.jpg").convert("RGB"), None)
    # pick one image from the test set
    # for img, _ in dataset_test:
    # img, _ = dataset_test[0]
    #
    # put the model in evaluation mode
    model.transform.max_size = 8100
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])
    # masks2 = prediction[0]['masks'].cpu().numpy()
    img_img = (Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy()))
    # img_img.show()

    # img_img = Image.open(r"I:\Research\CCS717B_1910_780pmm_edit crop.jpg").convert("RGB")
    box_img = ImageDraw.Draw(img_img)
    # img_masks = torch.zeros(img.size()[1:], dtype=torch.int32)
    # for mask in prediction[0]['masks']:
    #         img_masks = img_masks | mask[0].type(torch.int32).cpu()

    for box in prediction[0]['boxes']:
        box_img.rectangle(box.cpu().numpy(), outline="red", width=2)

    img_img.show()
    # img_img.save(r"../../conifer_bbox_demonstration_3.PNG")
    #
    # # make mask outlines to place on pic
    # img_masks_arr = img_masks.cpu().numpy()
    # img_outline = ImageDraw.Draw(img_img)
    # points = []
    # for x in range(img_masks_arr.shape[0]):
    #     for y in range(img_masks_arr.shape[1]):
    #         if img_masks_arr[x][y] == 1 and (img_masks_arr[x+1][y] == 0 or
    #                                          img_masks_arr[x-1][y] == 0 or
    #                                          img_masks_arr[x][y+1] == 0 or
    #                                          img_masks_arr[x][y-1] == 0):
    #             # img_outline.point([x,y], fill="Red")
    #             points.append((y, x))
    #
    # for point in points:
    #     img_outline.ellipse((point[0] - 1, point[1] - 1, point[0] + 1, point[1] + 1), fill="red")
    # img_outline.ellipse((100 - 4, 100 - 4, 100 + 4, 100 + 4), fill="green")
    # img_outline.point(points, fill='Red')
    # img_outline.point((100, 100), fill='Green')
    # img_outline.rectangle([(500, 600), (1000, 800)], fill='Red')
    # img_img.show()
    # img_img.save(r"../../conifer_bbox_demonstration_3_with_outline.PNG")
    # img_img.save(r"../../vessel_bbox_mask_demonstration_CCS717B_1910_780pmm.PNG")
    #
    # im = (Image.fromarray(img_masks.type(torch.float32).mul(255).byte().cpu().numpy()))
    # im.show()
    # im.save(r"../../conifer_bbox_demonstration_3_mask.PNG")
    # im.save(r"../../vessel_mask_demonstration_CCS717B_1910_780pmm.PNG")
    #
    print("hi")
