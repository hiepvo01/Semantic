import os
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from util import EarlyStopping, MetricMonitor
from util import TwoCropTransform, SupCon, SupConLoss, save_model

   
class Encoder(torch.nn.Module):
    "Encoder network"
    def __init__(self):
        super(Encoder, self).__init__()
        # L1 (?, 28, 28, 1) -> (?, 28, 28, 32) -> (?, 14, 14, 32)
        self.convnet = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(8, 8, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(8, 16, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(16, 16, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=0),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(32, 32, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(3 * 3 * 32, 128),
            # torch.nn.ReLU(True),
            # torch.nn.Linear(128, 4)
        )
        
        self._to_linear = 128
        
        
    def forward(self, x):
        x = self.convnet(x)
        return x

def pretraining(epoch, model, contrastive_loader, optimizer, criterion, method='SimCLR'):
    "Contrastive pre-training over an epoch"
    metric_monitor = MetricMonitor()
    model.train()
    for batch_idx, (data,labels) in enumerate(contrastive_loader):
        data = torch.cat([data[0], data[1]], dim=0)
        if torch.cuda.is_available():
            data,labels = data.cuda(), labels.cuda()
        data, labels = torch.autograd.Variable(data,False), torch.autograd.Variable(labels)
        bsz = labels.shape[0]
        features = model(data)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if method == 'SupCon':
            loss = criterion(features, labels)
        elif method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.format(method))
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Learning Rate", optimizer.param_groups[0]['lr'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("[Epoch: {epoch:03d}] Contrastive Pre-train | {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
    return metric_monitor.metrics['Loss']['avg'], metric_monitor.metrics['Learning Rate']['avg']

def main():
    
    num_epochs = 100
    use_early_stopping = True
    use_scheduler = True
    head_type = 'mlp' # choose among 'mlp' and 'linear"
    method = 'SimCLR' # choose among 'SimCLR' and 'SupCon'
    save_file = os.path.join('./results/', 'simclrMNIST.pth')
    if not os.path.isdir('./results/'):
         os.makedirs('./results/')
    
    contrastive_transform = transforms.Compose([
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,)),
                                       ])
    train_transform = transforms.Compose([
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,)),
                                       ])
    valid_transform = transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,)),
                                       ])
    
    contrastive_set = datasets.MNIST('./data', download=True, train=True, transform=TwoCropTransform(contrastive_transform))
    train_set = datasets.MNIST('./data', download=True, train=True, transform=train_transform)
    valid_set = datasets.MNIST('./data', download=True, train=False, transform=valid_transform)
    
    contrastive_loader = torch.utils.data.DataLoader(contrastive_set, batch_size=64, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=64, shuffle=True)
    
    # Part 1
    encoder = Encoder()
    model = SupCon(encoder, head=head_type, feat_dim=128)
    criterion = SupConLoss(temperature=0.07)
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()   
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    contrastive_loss, contrastive_lr = [], []
    
    for epoch in range(1, num_epochs+1):
        loss, lr = pretraining(epoch, model, contrastive_loader, optimizer, criterion, method=method)
        if use_scheduler:
            scheduler.step()
        contrastive_loss.append(loss)
        contrastive_lr.append(lr)
    
    save_model(model, optimizer, num_epochs, save_file)
    
    plt.plot(range(1,len(contrastive_lr)+1),contrastive_lr, color='b', label = 'learning rate')
    plt.legend(), plt.ylabel('loss'), plt.xlabel('epochs'), plt.title('Learning Rate'), plt.show()
    
    plt.plot(range(1,len(contrastive_loss)+1),contrastive_loss, color='b', label = 'loss')
    plt.legend(), plt.ylabel('loss'), plt.xlabel('epochs'), plt.title('Loss'), plt.show()
    
if __name__ == "__main__":
    main()

