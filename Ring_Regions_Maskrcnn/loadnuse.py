import torch
from PIL import Image, ImageDraw
from Ring_Maskrcnn import transforms as T

from Ring__Regions_Maskrcnn.start import RingMaskDataset, get_transform, get_instance_seg_model
from Ring__Regions_Maskrcnn import utils

dataset_test = RingMaskDataset('data', get_transform(train=False))
dataset = RingMaskDataset('data', get_transform(train=True))
indices = torch.randperm(len(dataset)).tolist()
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

model = get_instance_seg_model(2)

model.load_state_dict(torch.load('ring_3000.pt'))
model.eval()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
# img, _ = T.ToTensor()(Image.open(r"I:\Research\Research Images\testfull-sixteenth1-half-half.jpg").convert("RGB"), None)
img, _ = dataset_test[0]
# pick one image from the test set
# for img, _ in dataset_test:
# img, _ = dataset_test[0]

# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])

model.transform.max_size = 2000

org_img = (Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy()))
box_img = ImageDraw.Draw(org_img)
for box in prediction[0]['boxes']:
    box_img.rectangle(box.cpu().numpy(), outline="Red")
# org_img.show()
points=[]
masks2 = prediction[0]['masks'].cpu().numpy()
x = masks2.max()
img_masks = torch.zeros(img.size()[1:], dtype=torch.int32)
# for mask in prediction[0]['masks']:
#     points.append([(i, o) for i, y in enumerate(mask[0].cpu().numpy()) for o, t in enumerate(y) if t != 0])

# for line in points:
#     box_img.line(line)
org_img.show()
im = (Image.fromarray(img_masks.type(torch.float32).mul(255).byte().cpu().numpy()))
# im.show()

print("hi")
