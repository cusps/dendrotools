import torch
from PIL import Image
from Ring_NoMask_FRONLY import transforms as T

from Vessel_Maskrcnn.start import VesselDataset, get_transform, get_instance_seg_model
from Vessel_Maskrcnn import utils

dataset_test = VesselDataset('data', get_transform(train=False))
dataset = VesselDataset('data', get_transform(train=True))
indices = torch.randperm(len(dataset)).tolist()
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

model = get_instance_seg_model(2)

model.load_state_dict(torch.load('vessel_2000.pt'))
model.eval()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
img, _ = T.ToTensor()(Image.open(r"I:\Research\Research Images\testfull-sixteenth1-half-half.jpg").convert("RGB"), None)
# pick one image from the test set
# for img, _ in dataset_test:
# img, _ = dataset_test[0]

# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])

(Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())).show()
img_masks = torch.zeros(img.size()[1:], dtype=torch.int32)
for mask in prediction[0]['masks']:
    img_masks = img_masks | mask[0].type(torch.int32).cpu()

im = (Image.fromarray(img_masks.type(torch.float32).mul(255).byte().cpu().numpy()))
im.show()

print("hi")
