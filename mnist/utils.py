import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def get_noise_linear(n_samples,dim_is,device):
  noise  = torch.randn(n_samples,100,device = device)
  return noise
def get_noise(n_samples,dim_is,device):
  noise  = torch.randn(n_samples,1,dim_is,dim_is,device = device)
  return noise

def weights_init(m): #https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def plot_layout(preds, n_samples,epoch):
  preds = preds.reshape(preds.shape[0],32,32)
  preds = (preds + 1) / 2
  for i in range(n_samples * n_samples):
    plt.subplot(n_samples, n_samples, 1 + i)
    plt.axis('off')
    plt.imshow(preds[i, :, :])
  plt.savefig('/content/results/Epoch%d.png'%(epoch))
  plt.show()