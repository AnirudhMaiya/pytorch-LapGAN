import torch
import torch.nn as nn
import torch.nn.functional as F

class Gen_zero(nn.Module):
    def __init__(self):
        super(Gen_zero, self).__init__()
        self.c1 = nn.ConvTranspose2d(in_channels=2,out_channels = 256,kernel_size=3,padding = 1,bias = False)
        self.b1 = nn.BatchNorm2d(256)
        self.a1 =nn.LeakyReLU(0.2,inplace = True)

        self.c2 = nn.ConvTranspose2d(in_channels=256,out_channels = 128,kernel_size=3,padding = 1,bias = False)
        self.b2 = nn.BatchNorm2d(128)
        self.a2 = nn.LeakyReLU(0.2,inplace = True)

        self.c3 = nn.ConvTranspose2d(in_channels=128,out_channels = 64,kernel_size=3,padding = 1,bias = False)
        self.b3 = nn.BatchNorm2d(64)
        self.a3 = nn.LeakyReLU(0.2,inplace = True)

        self.c4 = nn.ConvTranspose2d(in_channels=64,out_channels = 1,kernel_size=3,padding = 1,bias = False)
        self.a4 = nn.Tanh()
    def forward(self,x,upsampled_img):
      x = torch.cat((upsampled_img,x),dim=1)
      x = self.a1(self.b1(self.c1(x)))
      x = self.a2(self.b2(self.c2(x)))
      x = self.a3(self.b3(self.c3(x)))
      x = self.a4(self.c4(x))
      return x


class Gen_one(nn.Module):
    def __init__(self):
        super(Gen_one, self).__init__()
        self.c1 = nn.ConvTranspose2d(in_channels=2,out_channels = 128,kernel_size=3,padding = 1,bias = False)
        self.b1 = nn.BatchNorm2d(128)
        self.a1 = nn.LeakyReLU(0.2,inplace = True)

        self.c2 = nn.ConvTranspose2d(in_channels=128,out_channels = 64,kernel_size=3,padding = 1,bias = False)
        self.b2 = nn.BatchNorm2d(64)
        self.a2 = nn.LeakyReLU(0.2,inplace = True)

        self.c3 = nn.ConvTranspose2d(in_channels=64,out_channels = 1,kernel_size=3,padding = 1,bias = False)
        self.a3 = nn.Tanh()
    def forward(self,x,upsampled_img):
      x = torch.cat((upsampled_img,x),dim=1)
      x = self.a1(self.b1(self.c1(x)))
      x = self.a2(self.b2(self.c2(x)))
      x = self.a3(self.c3(x))
      return x

class Gen_two(nn.Module):
    def __init__(self):
        super(Gen_two, self).__init__()
        self.fc1 = nn.Linear(100, 1200)
        self.a1 = nn.LeakyReLU(0.2,inplace = True)
        self.fc2 = nn.Linear(1200, 1200)
        self.a2 = nn.LeakyReLU(0.2,inplace = True)
        self.fc3 = nn.Linear(1200, 8*8*1)
    def forward(self,x,upsampled_img=None):
      x = self.a1(self.fc1(x))
      x = self.a2(self.fc2(x))
      x = torch.tanh(self.fc3(x))
      x = x.view(-1, 1, 8, 8)
      return x


class Disc_zero(nn.Module):
    def __init__(self):
        super(Disc_zero, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1,out_channels=2,kernel_size =4,stride = 2,padding = 1,bias= False)
        self.a1 = nn.LeakyReLU(0.2,inplace = True)

        self.c2 = nn.Conv2d(in_channels=2,out_channels=2,kernel_size =4,stride = 2,padding = 1,bias = False)
        self.b2 = nn.BatchNorm2d(2)
        self.a2 = nn.LeakyReLU(0.2,inplace = True)

        self.c3 = nn.Conv2d(in_channels=2,out_channels=2,kernel_size =4,stride = 2,padding = 1,bias = False)
        self.b3 = nn.BatchNorm2d(2)
        self.a3 = nn.LeakyReLU(0.2,inplace = True)

        self.c4 = nn.Conv2d(in_channels=2,out_channels=1,kernel_size =4,stride = 1,padding = 0,bias = False)
        self.a4 = nn.Sigmoid()
        
    def forward(self,x,edge_img):
      x = x + edge_img
      x = torch.clamp(x,min = -1,max = 1)
      x = self.a1(self.c1(x))
      x = self.a2(self.b2(self.c2(x)))
      x = self.a3(self.b3(self.c3(x)))
      x = self.a4(self.c4(x))
      return x


class Disc_one(nn.Module):
    def __init__(self):
        super(Disc_one, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size =4,stride = 2,padding = 1,bias= False)
        self.a1 = nn.LeakyReLU(0.2,inplace = True)

        self.c2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size =4,stride = 2,padding = 1,bias = False)
        self.b2 = nn.BatchNorm2d(16)
        self.a2 = nn.LeakyReLU(0.2,inplace = True)


        self.c3 = nn.Conv2d(in_channels=16,out_channels=1,kernel_size =4,stride = 1,padding = 0,bias = False)
        self.a3 = nn.Sigmoid()
        
    def forward(self,x,edge_img):
      x = x + edge_img
      x = torch.clamp(x,min = -1,max = 1)
      x = self.a1(self.c1(x))
      x = self.a2(self.b2(self.c2(x)))
      x = self.a3(self.c3(x))
      return x

class Disc_two(nn.Module):
    def __init__(self):
        super(Disc_two, self).__init__()
        self.fc1 = nn.Linear(1*8*8, 1200)
        self.a1 = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Linear(1200, 2000)
        self.a2 = nn.LeakyReLU(inplace=True)
        self.fc3 = nn.Linear(2000, 1)
    def forward(self,x,edge_img = None):
      x = x.view(-1, 1*8*8)
      x = F.leaky_relu(self.fc1(x))
      x = F.leaky_relu(self.fc2(x))
      x = F.sigmoid(self.fc3(x))
      return x
