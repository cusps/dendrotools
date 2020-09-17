# dataset imports
import os
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw

# getting instance imports
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# training imports
from Conifer_Trach_Fasterrcnn import transforms as T
from Conifer_Trach_Fasterrcnn import utils
from Conifer_Trach_Fasterrcnn.engine import train_one_epoch, evaluate


def get_rid_of_white_boundary(mask):
    mask[0] = 3
    mask[-1] = 3
    mask[:, 0] = 3
    mask[:, -1] = 3


class VesselDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "training"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "training", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background

        mask = Image.open(mask_path)
        # mask = ImageOps.grayscale(mask)
        # mask.show()
        # mask = np.array(mask)

        # get_rid_of_white_boundary(mask)
        # instances are encoded as different colors
        # obj_ids = np.unique(mask)
        # obj_ids = np.array(np.unique(mask.reshape(-1, mask.shape[2]), axis=0))
        # first id is the background, so remove it
        # obj_ids = obj_ids[1:]
        # new_colors = np.zeros(len(obj_ids))
        # mask_count = 0
        # for i in range(len(new_colors)):
        #     new_colors[i] = 255-mask_count
        #     mask_count += 1
        # obj_ids = obj_ids.tolist()
        # mask = mask.tolist()
        # for id in obj_ids:
        #     mask = np.where(mask == id, (200-mask_count, 0, 0), mask, axis=-1)
        #     # mask = np.where((mask[:, :, 0] == id[0]) & (mask[:, :, 1] == id[1]) & (mask[:, :, 2] == id[2]))
        #     # for loc in mask_locs:
        #     #     None
        #     mask_count += 1
        #
        # for i in range(len(mask)):
        #     for j in range(len(mask[i])):
        #         for id_num in range(len(obj_ids)):
        #             # print(mask[i][j])
        #             # print(obj_ids[id_num])
        #             if obj_ids[id_num] == mask[i][j]:
        #                 mask[i][j] = [new_colors[id_num], 0, 0]
        #                 break


        mask = np.array(ImageOps.grayscale(mask))
        # Image.fromarray(mask).show()
        # mask = np.asarray(mask)
        # mask = np.dot(mask[..., :3], [1, 0, 0])
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # box_img = ImageDraw.Draw(img)
        # for box in boxes:
        #     box_img.rectangle(box)
        # img.show()

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


def get_instance_seg_model(num_classes, backbone=1):
    if not backbone:
        # load a model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # replace the classifier with a new one, that has
        # num_classes which is user-defined
        num_classes = 2  # 1 class (person) + background
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        # load a pre-trained model for classification and return
        # only the features
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        # FasterRCNN needs to know the number of
        # output channels in a backbone. For mobilenet_v2, it's 1280
        # so we need to add it here
        backbone.out_channels = 1280

        # let's make the RPN generate 5 x 3 anchors per spatial
        # location, with 5 different sizes and 3 different aspect
        # ratios. We have a Tuple[Tuple[int]] because each feature
        # map could potentially have different sizes and
        # aspect ratios
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))

        # let's define what are the feature maps that we will
        # use to perform the region of interest cropping, as well as
        # the size of the crop after rescaling.
        # if your backbone returns a Tensor, featmap_names is expected to
        # be [0]. More generally, the backbone should return an
        # OrderedDict[Tensor], and in featmap_names you can choose which
        # feature maps to use.
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                        output_size=7,
                                                        sampling_ratio=2)

        # put the pieces together inside a FasterRCNN model
        model = FasterRCNN(backbone,
                           num_classes=2,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler,
                           max_size=1600)
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

    # dataset = VesselDataset('data', get_transform(train=True))
    # dataset_test = VesselDataset('data', get_transform(train=False))

    dataset = VesselDataset('../../data/Conifer/tracheids', get_transform(train=True))
    dataset_test = VesselDataset('../../data/Conifer/tracheids/', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices)
    dataset_test = torch.utils.data.Subset(dataset_test, indices)



    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_instance_seg_model(num_classes, backbone=0)


    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0005,
                                momentum=0.5, weight_decay=0.0005)
    # and a learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=1,
    #                                                gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 200

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        # lr_scheduler.step(epoch)
        # evaluate on the test dataset
        if (epoch + 1) % 10 == 0:
            evaluate(model, data_loader_test, device=device)

        if epoch+1 == 5:
            torch.save(model.state_dict(), "./trach_5.pt")

        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), "./trach_{}.pt".format(epoch+1))


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