import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class Images(Dataset):
    
    def __init__(self, x_directory, y_directory):
        self.x_directory = x_directory
        self.y_directory = y_directory
        self.x_files = os.listdir(x_directory)
        self.y_files = os.listdir(y_directory)
        self.x_files.sort()
        self.y_files.sort()

        if ".DS_Store" in self.x_files:
            self.x_files.remove(".DS_Store")
        if ".DS_Store" in self.y_files:
            self.y_files.remove(".DS_Store")            


    def __getitem__(self, index):

        x = Image.open(self.x_directory+"/"+self.x_files[index])
        y = Image.open(self.y_directory+"/"+self.y_files[index])
        y = y.convert("L")

        threshold = 10  
        y = y.point(lambda p: 255 if p > threshold else 0)


        x = x.resize(size=(384, 512))
        y = y.resize(size=(384, 512))
        
        converter = transforms.ToTensor()

        return converter(x), converter(y)
    
    def get_files(self):
        return list(zip(self.x_files, self.y_files))
    
    def __len__(self):
        return len(self.x_files)
