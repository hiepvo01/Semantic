import os
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from util import EarlyStopping, MetricMonitor
from util import TwoCropTransform, SupCon, SupConLoss, save_model
from simclrMNIST import Encoder

class Decoder(torch.nn.Module):
    "Decoder network"
    def __init__(self):
        super(Decoder, self).__init__()
        # L1 (?, 28, 28, 1) -> (?, 28, 28, 32) -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2)
            )
        # L2 (?, 14, 14, 32) -> (?, 14, 14, 64) -> (?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2)
            )
        # L3 (?, 7, 7, 64) -> (?, 7, 7, 128) -> (?, 4, 4, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2)
            )
        self._to_linear = 7 * 7 * 128

    def forward(self, x):
        x = x.view(-1, 128, 7, 7) # Flatten them for FC
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

def decoder_training(epoch, model, decoder, train_loader, optimizer, criterion):
    "Training over an epoch"
    metric_monitor = MetricMonitor()
    model.eval()
    decoder.train()
    for batch_idx, (data,labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            data,labels = data.cuda(), labels.cuda()
        data = torch.autograd.Variable(data,False)
        with torch.no_grad():
            features = model.encoder(data)
        output = decoder(features.float())
        loss = criterion(output, data)
        metric_monitor.update("Loss", loss.item())
        data.detach()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("[Epoch: {epoch:03d}] Train      | {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
    return metric_monitor.metrics['Loss']['avg']


def decoder_validation(epoch, model, decoder, valid_loader, criterion):
    "Validation over an epoch"
    metric_monitor = MetricMonitor()
    model.eval()
    decoder.eval()
    with torch.no_grad():
        for batch_idx, (data,labels) in enumerate(valid_loader):
            if torch.cuda.is_available():
                data,labels = data.cuda(), labels.cuda()
            data = torch.autograd.Variable(data,False)
            features = model.encoder(data)
            output = decoder(features.float())
            loss = criterion(output,data) 
            metric_monitor.update("Loss", loss.item())
            data.detach()
    print("[Epoch: {epoch:03d}] Validation | {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
    return metric_monitor.metrics['Loss']['avg']


def main():
    
    num_epochs = 100
    use_early_stopping = True
    use_scheduler = True
    head_type = 'mlp' # choose among 'mlp' and 'linear"
    method = 'SimCLR' # choose among 'SimCLR' and 'SupCon'
    save_file = os.path.join('./results/', 'simclrMNIST.pth')
    if not os.path.isdir('./results/'):
         os.makedirs('./results/')
    
    train_transform = transforms.Compose([
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,)),
                                       ])
    valid_transform = transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,)),
                                       ])
    
    train_set = datasets.MNIST('./data', download=True, train=True, transform=train_transform)
    valid_set = datasets.MNIST('./data', download=True, train=False, transform=valid_transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=64, shuffle=True)
    
    # Part 2
    encoder = Encoder()
    model = SupCon(encoder, head=head_type, feat_dim=128)
    decoder = Decoder()
    criterion = torch.nn.MSELoss()
    
    ckpt = torch.load(save_file, map_location='cpu')
    state_dict = ckpt['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict
    model.load_state_dict(state_dict)
    
    
    if torch.cuda.is_available():
        model = model.cuda()
        decoder = decoder.cuda()
        criterion = criterion.cuda()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
        
    train_losses = []
    valid_losses = []
    
    if use_early_stopping:
        early_stopping = EarlyStopping(patience=30, verbose=False, delta=1e-4)
 
    for epoch in range(1, num_epochs+1):
        
        train_loss = decoder_training(epoch, model, decoder, train_loader, optimizer, criterion)
        valid_loss = decoder_validation(epoch, model, decoder, valid_loader, criterion)

        
        if use_scheduler:
            scheduler.step()
            
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
             
        if use_early_stopping: 
            early_stopping(valid_loss, model)
            
            if early_stopping.early_stop:
                print('Early stopping at epoch', epoch)
                #model.load_state_dict(torch.load('checkpoint.pt'))
                break
    
    save_model(decoder, optimizer, num_epochs, './results/decoderMNIST.pth')

     
    plt.plot(range(1,len(train_losses)+1), train_losses, color='b', label = 'training loss')
    plt.plot(range(1,len(valid_losses)+1), valid_losses, color='r', linestyle='dashed', label = 'validation loss')
    plt.legend(), plt.ylabel('loss'), plt.xlabel('epochs'), plt.title('Loss'), plt.show()


    counts = 0
    for batch_idx, (data,labels) in enumerate(train_loader):
        if counts == 3:
            break
        if torch.cuda.is_available():
            data,labels = data.cuda(), labels.cuda()
        data, labels = torch.autograd.Variable(data,False), torch.autograd.Variable(labels)
        with torch.no_grad():
            features = model.encoder(data)
        output = decoder(features.float())
        counts += 1
        
        data = data.cpu().detach().numpy().reshape(-1, 28, 28)
        plt.imshow(data[0])
        plt.show()
        output = output.cpu().detach().numpy().reshape(-1, 28, 28)
        plt.imshow(output[0])
        plt.show()

        
    
if __name__ == "__main__":
    main()

