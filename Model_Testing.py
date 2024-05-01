import torch 
from UNET import UNET
from PIL import Image
import torchvision
from torchvision import transforms 
from torchvision.utils import draw_segmentation_masks
from Images import Images

imagepath = "/Users/nivaankaushal/Downloads/TrashApp/TrashAppV2 (2024)/Bottles"
maskpath = "/Users/nivaankaushal/Downloads/TrashApp/TrashAppV2 (2024)/SegmentationObject"
model = UNET(in_channels=3, out_channels=1)
model.load_state_dict(torch.load("UNET_Testing/model.pt"))
images = Images(x_directory=imagepath, y_directory=maskpath)



model.eval()

model.eval()
with torch.inference_mode():
    for x, y in images: 
        res = model(x.unsqueeze(dim=0))
        res = torch.sigmoid(res)
        res = (res > 0.9).float()
        res = draw_segmentation_masks((x.clamp(0, 1) * 255).to(torch.uint8), (res>0.5).squeeze(), 0.5, colors= (0, 237, 156))
        pilimg = transforms.ToPILImage()
        img = pilimg(res)
        img.show()
