# dataset imports
import os
import numpy as np
import torch
import json
from PIL import Image, ImageOps

# getting instance imports
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# training imports
from Vessel_Maskrcnn import transforms as T
from Vessel_Maskrcnn import utils
from Vessel_Maskrcnn.engine import train_one_epoch, evaluate


def get_rid_of_white_boundary(mask):
    mask[0] = 3
    mask[-1] = 3
    mask[:, 0] = 3
    mask[:, -1] = 3


class RingMaskDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "imgs"))))
        self.masks = {}
        with open(os.path.join(root, "imgs", "annotations.json")) as f:
            self.masks = json.load(f)
        # self.masks = [self.masks[x] for x in self.masks]

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "imgs", self.imgs[idx])
        img_filename = os.path.split(img_path)[-1]
        # mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = self.masks["{}{}".format(img_path, os.path.getsize(img_path))]
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background

        # mask = Image.open(mask_path)
        # mask = ImageOps.grayscale(mask)
        # mask.show()
        # mask = np.array(mask)

        # get_rid_of_white_boundary(mask)
        # instances are encoded as different colors
        # obj_ids = np.unique(mask)
        # first id is the background, so remove it
        # obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        mask_points = [(mask["regions"][i]["shape_attributes"]["all_points_x"][i],
                        mask["regions"][i]["shape_attributes"]["all_points_x"][i])
                       for i in range(len(mask["regions"]["shape_attributes"]["all_points_x"]))]

        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(mask["regions"])
        boxes = []
        for i in range(num_objs):
            # pos = np.where(masks[i])
            xmin = np.min(mask["regions"][i]["shape_attributes"]["all_points_x"])
            xmax = np.max(mask["regions"][i]["shape_attributes"]["all_points_x"])
            ymin = np.min(mask["regions"][i]["shape_attributes"]["all_points_y"])
            ymax = np.max(mask["regions"][i]["shape_attributes"]["all_points_y"])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_instance_seg_model(num_classes):
    # load an instance segmantation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the num of input features for classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the num of input features for mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # replace mask predictor with new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


def get_transform(train):
    transforms = list()
    # convert image (PIL) to Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training we want to randomly flip training
        # images and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def train_model():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2

    dataset = RingMaskDataset('data', get_transform(train=True))
    dataset_test = RingMaskDataset('data', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices)
    dataset_test = torch.utils.data.Subset(dataset_test, indices)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_instance_seg_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=3,
    #                                                gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10000

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        # lr_scheduler.step(epoch)
        # evaluate on the test dataset
        if (epoch + 1) % 100 == 0:
            evaluate(model, data_loader_test, device=device)

        if (epoch+1) % 1000 == 0:
            torch.save(model.state_dict(), "./vessel_{}.pt".format(epoch+1))

    print("That's it!")

    # pick one image from the test set
    img, _ = dataset_test[0]
    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])
    print('hi')


if __name__ == '__main__':
    train_model()
