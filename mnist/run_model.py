import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
import torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import numpy as np
import matplotlib.pyplot as plt
import cv2
from layers import *
from utils import *

batch_is = 128
transform_train = transforms.Compose([
    transforms.Resize((32,32), interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5], std = [0.5])
])

transform_test = transforms.Compose([
                                     transforms.Resize((32,32), interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5], std = [0.5])
])
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_is,
                                          shuffle=True, num_workers=1,pin_memory=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_is,
                                         shuffle=False, num_workers=1,pin_memory=True)

def sampling(n_samples):
  with torch.no_grad():
    noise_dim_in_each_level = [8,16,32]
    for i in range(level):
      if i == 0:
        noise_is = get_noise_linear(n_samples,noise_dim_in_each_level[i],device)
        pred = GenModels[level - 1 - i](noise_is)
      else:
        upsampled_img = []
        size_is = noise_dim_in_each_level[i-1]
        pred = pred.detach().cpu().numpy()

        for ind in range(n_samples):
          res = pred[ind].reshape(size_is,size_is)
          res = cv2.pyrUp(res)
          upsampled_img.append(res.reshape(1,res.shape[0],res.shape[0]))
        upsampled_img = torch.tensor(upsampled_img).to(device)
        noise_is = get_noise(n_samples,noise_dim_in_each_level[i],device)
        pred = GenModels[level - 1 - i](noise_is,upsampled_img)
        pred = pred + upsampled_img
  pred = pred.clamp(min=-1,max = 1)
  return pred.detach().cpu()

def call_mod(epoch):
  for idx, data in enumerate(trainloader, 0):
    inputs,_ = data[0].to(device),data[1].to(device)
    downsampled_images_list = []
    upsampled_images_list = []
    edge_images_list = []
    list_for_GLoss = []
    list_for_DLoss = []
    for l in range(level):  
      if l == 0:  
        for cv2_index in range(inputs.shape[0]):
          res_copy = inputs[cv2_index].reshape(1,32,32).cpu().numpy()
          res = inputs[cv2_index].reshape(1,32,32)
          res = res.detach().cpu().numpy()
          res_down = cv2.pyrDown(res.reshape(32,32))
          res_down = res_down.reshape(1,16,16)
          downsampled_images_list.append(res_down)
          res_up = cv2.pyrUp(res_down)
          res_up = res_up.reshape(1,32,32)
          upsampled_images_list.append(res_up)
          edge_images_list.append(res_copy - res_up)
      
        upsampled_images_tensor = torch.tensor(np.array(upsampled_images_list).copy()).to(device)
        edge_images_tensor = torch.tensor(np.array(edge_images_list).copy()).to(device)
      
        #Disc 0
        DiscModels[l].zero_grad()
        label = torch.full((inputs.shape[0],), real_label, device=device)
        output = DiscModels[l](upsampled_images_tensor,edge_images_tensor).view(-1)
        D_real = DiscLosses[l](output,label)
        D_real.backward()

        noise = get_noise(inputs.shape[0],32,device)
        fake = GenModels[l](noise,upsampled_images_tensor)
        label.fill_(fake_label)
        output = DiscModels[l](upsampled_images_tensor,fake.detach()).view(-1)
        D_fake = DiscLosses[l](output,label)
        D_fake.backward()
        total = D_real + D_fake
        DiscOpts[l].step()

        GenModels[l].zero_grad()
        label.fill_(real_label)
        output = DiscModels[l](upsampled_images_tensor,fake).view(-1)
        G_loss = GenLosses[l](output,label)
        G_loss.backward()
        GenOpts[l].step()
        list_for_DLoss.append(total.item())
        list_for_GLoss.append(G_loss.item())

      elif l == 1:
        for cv2_index in range(inputs.shape[0]):
          res_copy = downsampled_images_list[cv2_index].reshape(1,16,16)
          res_down = cv2.pyrDown(downsampled_images_list[cv2_index].reshape(16,16))
          res_down = res_down.reshape(1,8,8)
          downsampled_images_list[cv2_index] = res_down
          res_up = cv2.pyrUp(downsampled_images_list[cv2_index].reshape(8,8))
          res_up = res_up.reshape(1,16,16)
          upsampled_images_list[cv2_index] =  res_up
          edge_images_list[cv2_index] = res_copy - res_up
        upsampled_images_tensor = torch.tensor(np.array(upsampled_images_list).copy()).to(device)
        edge_images_tensor = torch.tensor(np.array(edge_images_list).copy()).to(device)

        #Disc 1
        DiscModels[l].zero_grad()
        label_2 = torch.full((inputs.shape[0],), real_label, device=device)
        output = DiscModels[l](upsampled_images_tensor,edge_images_tensor).view(-1)
        D_real = DiscLosses[l](output,label_2)
        D_real.backward()

        noise = get_noise(inputs.shape[0],16,device)
        fake = GenModels[l](noise,upsampled_images_tensor)
        label_2.fill_(fake_label)
        output = DiscModels[l](upsampled_images_tensor,fake.detach()).view(-1)
        D_fake = DiscLosses[l](output,label_2)
        D_fake.backward()
        total = D_real + D_fake
        DiscOpts[l].step()

        #Gen 1
        GenModels[l].zero_grad()
        label_2.fill_(real_label)
        output = DiscModels[l](upsampled_images_tensor,fake).view(-1)
        G_loss = GenLosses[l](output,label_2)
        G_loss.backward()
        GenOpts[l].step()
        list_for_DLoss.append(total.item())
        list_for_GLoss.append(G_loss.item())

      elif l == 2:
        downsampled_images_tensor = torch.tensor(downsampled_images_list).to(device)
        #Disc 2
        DiscModels[l].zero_grad()
        label_3 = torch.full((inputs.shape[0],), real_label, device=device)
        output = DiscModels[l](downsampled_images_tensor).view(-1)
        D_real = DiscLosses[l](output,label_3)
        D_real.backward()

        noise = get_noise_linear(inputs.shape[0],8,device)
        fake = GenModels[l](noise)
        label_3.fill_(fake_label)
        output = DiscModels[l](fake.detach()).view(-1)
        D_fake = DiscLosses[l](output,label_3)
        D_fake.backward()
        total = D_real + D_fake
        DiscOpts[l].step()

        #Gen 2
        GenModels[l].zero_grad()
        label_3.fill_(real_label)
        output = DiscModels[l](fake).view(-1)
        G_loss = GenLosses[l](output,label_3)
        G_loss.backward()
        GenOpts[l].step()
        list_for_DLoss.append(total.item())
        list_for_GLoss.append(G_loss.item())
    
    if(idx %200 == 0):
      print('Iter %d of [%d/%d]'%(idx,epoch,total_epochs))
      print("Disc 0 : %f and Gen 0 : %f"%(list_for_DLoss[0],list_for_GLoss[0]))
      print("Disc 1 : %f and Gen 1 : %f"%(list_for_DLoss[1],list_for_GLoss[1]))
      print("Disc 2 : %f and Gen 2 : %f"%(list_for_DLoss[2],list_for_GLoss[2]))
    
  dd = sampling(n_samples)
  dd = dd.numpy()
  plot_layout(dd,int(n_samples**0.5),epoch)

total_epochs = 50
level = 3
n_samples = 25 
real_label = 1
fake_label = 0
DiscModels = []
GenModels = []
DiscLosses = []
GenLosses = []
DiscOpts = []
GenOpts = []

disc_0 = Disc_zero()
disc_0 = disc_0.apply(weights_init).to(device)
disc_1 = Disc_one()
disc_1 = disc_1.apply(weights_init).to(device)
disc_2 = Disc_two()
disc_2 = disc_2.apply(weights_init).to(device)
DiscModels.extend((disc_0,disc_1,disc_2))

gen_0 = Gen_zero()
gen_0 = gen_0.apply(weights_init).to(device)
gen_1 = Gen_one()
gen_1 = gen_1.apply(weights_init).to(device)
gen_2 = Gen_two()
gen_2 = gen_2.apply(weights_init).to(device)
GenModels.extend((gen_0,gen_1,gen_2))

DiscLosses.extend((nn.BCELoss(),nn.BCELoss(),nn.BCELoss()))
GenLosses.extend((nn.BCELoss(),nn.BCELoss(),nn.BCELoss()))

DiscOpts.extend((torch.optim.Adam(DiscModels[0].parameters(),lr = 9e-6,betas = (0.5,0.999)),
                 torch.optim.Adam(DiscModels[1].parameters(),lr = 1e-3,betas = (0.5,0.999)),
                 torch.optim.Adam(DiscModels[2].parameters(),lr = 1e-3,betas = (0.5,0.999))))

GenOpts.extend((torch.optim.Adam(GenModels[0].parameters(),lr = 1e-3,betas = (0.5,0.999)),
                 torch.optim.Adam(GenModels[1].parameters(),lr = 1e-3,betas = (0.5,0.999)),
                 torch.optim.Adam(GenModels[2].parameters(),lr = 1e-4,betas = (0.5,0.999))))

def run_mod(total_epochs):
  for epoch in range(total_epochs):
    for i in range(level):
      DiscModels[i].train()
      GenModels[i].train()
    call_mod(epoch)

run_mod(total_epochs)