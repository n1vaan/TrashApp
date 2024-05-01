if __name__ == '__main__':
    import torch
    from torch import nn
    from matplotlib import pyplot as plt
    from PIL import Image
    from UNET import UNET
    from tqdm import tqdm 
    from Images import Images
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms 
    from torch.optim.lr_scheduler import StepLR

    model = UNET(in_channels=3, out_channels=1)
    imagepath = ""
    maskpath = ""
    images = Images(x_directory=imagepath, y_directory=maskpath)

    epochs = 10

    data_loader = DataLoader(images, batch_size=8, shuffle=True, num_workers=2)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr = 0.01)
    counter = 0
    for epoch in tqdm(range(epochs)):
        for x,y in tqdm(data_loader):
            model.train()
            pred = model(x)
            loss = loss_fn(y, pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            counter+=1
            print(f"{counter}: loss {loss.item()}")
            
            if counter % 2 == 0:
                model.eval()
                with torch.inference_mode():
                    res = model(images[0][0].unsqueeze(dim=0))
                    res = torch.sigmoid(res)
                    res = (res > 0.9).float()
                    pilimg = transforms.ToPILImage()
                    img = pilimg(res.squeeze())
                    img.show()


    torch.save(model.state_dict(), 'model.pt')
        


