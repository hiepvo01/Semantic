import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object
import pandas as pd
from copy import deepcopy


# PyTorch Lightning
try:
    import pytorch_lightning as pl
except ModuleNotFoundError: # Google Colab does not have PyTorch Lightning installed by default. Hence, we do it here if necessary
    #!pip install --quiet pytorch-lightning>=1.4
    import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

class SimCLR(pl.LightningModule):
    
    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500, c_hid=32):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, 'The temperature must be a positive float!'
        # Base model f(.)
        self.convnet = torchvision.models.resnet18(num_classes=4*hidden_dim)  # Output of last linear layer
        self.convnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # The MLP for g(.) consists of Linear->ReLU->Linear 
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4*hidden_dim, hidden_dim)
        )
        
    def forward(self, x):
        return self.convnet(x)
    

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), 
                                lr=self.hparams.lr, 
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr/50)
        return [optimizer], [lr_scheduler]
        
    def info_nce_loss(self, batch, mode='train'):
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)
        
        # Encode all images
        feats = self.convnet(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()
        
        # Logging loss
        self.log(mode+'_loss', nll)
        # Get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
                              cos_sim.masked_fill(pos_mask, -9e15)], 
                             dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode+'_acc_top1', (sim_argsort == 0).float().mean())
        self.log(mode+'_acc_top5', (sim_argsort < 5).float().mean())
        self.log(mode+'_acc_mean_pos', 1+sim_argsort.float().mean())
        
        return nll
        
    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode='train')
        
    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode='val')

def prepareData(source):
    source = source
    if source == "CIFAR10":
        transform = transforms.Compose(
                [transforms.ToTensor()])

        batch_size = 128

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
        
        classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    elif source == "MNIST":
        transform = transforms.Compose(
                [transforms.ToTensor(),])

        batch_size = 128

        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
        trainsize = 60000
        
    elif source == "STL10":
        trainsize = 100000
        transform = transforms.Compose(
                [transforms.ToTensor(),])

        batch_size = 128
        
        trainset = torchvision.datasets.STL10(root='./data', split='unlabeled',
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.STL10(root='./data', split='test',
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
    return trainset, testset, trainloader, testloader



class Decoder(nn.Module):
    
    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder model
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2*12*12*c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
            nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 12, 12)
        x = self.net(x)
        return x
    
class MNISTDecoder(nn.Module):
    
    def __init__(self, encoded_space_dim=4):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 
            stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
            padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, 
            padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def plot_ae_outputs(encoder,decoder, test_dataset, source, epoch,n=10):
    plt.figure(figsize=(16,4.5))
    
    try:
        targets = test_dataset.targets.numpy()
    except:
        # targets = np.array(test_dataset.targets)
        targets = np.array(test_dataset.labels) 

    t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
         rec_img  = decoder(encoder(img))
      try:
        plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      except:
        plt.imshow(img.T.cpu().squeeze().numpy(), cmap='gist_gray')

      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      try:
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
      except:
        plt.imshow(rec_img.T.cpu().squeeze().numpy(), cmap='gist_gray')

      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.savefig('./figures/MNIST/' + source + '_epoch_' +str(epoch) +'.png')


def loop(trainset, testset, trainloader, testloader, source):
    # model = models.resnet18()
    # num_features = model.fc.in_features     #extract fc layers features
    # model.fc = nn.Linear(num_features, 512)
    simclr_model = SimCLR( 
            hidden_dim=128, 
            lr=5e-4, 
            temperature=0.07, 
            weight_decay=1e-4, 
            max_epochs=100)

    simclr_model.convnet.load_state_dict(
        torch.load('./results/simclrMNIST-200.pt')
    )

    model = deepcopy(simclr_model.convnet)
    model.fc = nn.Identity()  # Removing projection head g(.)
    model.eval()
    model.to(device)
    num_features = 512
    decoder = Decoder(num_input_channels=3, base_channel_size=96, latent_dim=num_features)
    if source == "MNIST":
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        decoder = MNISTDecoder(encoded_space_dim=num_features)
    
    model = model.to(device) 
    decoder = decoder.to(device)
    criterion = nn.MSELoss()       #(set loss function)
    
    
    params_to_optimize = [
        {'params': model.parameters()},
        {'params': decoder.parameters()}
    ]

    optimizer = torch.optim.Adam(params_to_optimize, lr=0.001, weight_decay=1e-05)
    
     
    num_epochs = 30   #(set no of epochs)
    start_time = time.time() #(for showing time)
    
    times = []
    epochs = []
    ssims = []
    
    for epoch in range(num_epochs): #(loop for every epoch)
        ssim_epoch = 0
        print("Epoch {} running".format(epoch)) #(printing message)``
        """ Training Phase """
        model.train()    #(training model)
        decoder.train()
        running_loss = 0.   #(set loss 0)
        # load a batch data of images
        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            # forward inputs and get output
            optimizer.zero_grad()
            outputs = model(inputs)
            reconstructs = decoder(outputs)
                        
            loss = criterion(reconstructs, inputs)
            # get loss value and update the model weights
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(trainset)
        print('[Train #{}] Loss: {:.4f} % Time: {:.4f}s'.format(epoch, epoch_loss, time.time() -start_time))
        
        plot_ae_outputs(model, decoder, testset, source, epoch)
        
        """ Testing Phase """
        model.eval()
        decoder.eval()
        with torch.no_grad():
            running_loss = 0.
            for inputs, labels in testloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                reconstructs = decoder(outputs)
                
                if source == 'MNIST':
                    trainsize = 10000
                    ssim_batch = ssim(reconstructs, inputs, win_size=11,
                            size_average=False, data_range=1)
                elif source == 'STL10':
                    trainsize = 8000    
                    ssim_batch = ssim(reconstructs, inputs, win_size=11,
                            size_average=False, data_range=255)
                ssim_epoch += torch.sum(ssim_batch).item()
                
                loss = criterion(reconstructs, inputs)
                running_loss += loss.item() * inputs.size(0)
                
            times.append(time.time() -start_time)
            ssims.append(ssim_epoch/trainsize)
            epochs.append(epoch)
            epoch_loss = running_loss / len(testset)
            print('[Test #{}] Loss: {:.4f} SSIM: {:.4f} % Time: {:.4f}s'.format(epoch, epoch_loss, ssim_epoch/trainsize ,time.time()- start_time))
            
        # Calling DataFrame constructor after zipping
        # both lists, with columns specified
        df = pd.DataFrame(list(zip(epochs, ssims, times)),
                    columns =['Epoch', 'SSIM', 'Time'])
        df.to_csv('./results/MNIST/' + source + '_SIMCLR.csv')
            
        
    save_path = './results/MNIST/' + source +  'encoder.pt'
    torch.save(model.state_dict(), save_path)
    save_path = './results/MNIST/' + source +  'decoder.pt'
    torch.save(decoder.state_dict(), save_path)


if __name__ == '__main__':    
    source = "MNIST" # Choose between STL10 or MNIST
    os.makedirs("./results/traditional", exist_ok=True)
    trainset, testset, trainloader, testloader = prepareData(source)
    loop(trainset, testset, trainloader, testloader, source)
