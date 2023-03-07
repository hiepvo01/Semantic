import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from util import TwoCropTransform, SupCon, SupConLoss, save_model


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

class Encoder(torch.nn.Module):
    "Encoder network"
    def __init__(self):
        super(Encoder, self).__init__()
        # L1 (?, 28, 28, 1) -> (?, 28, 28, 32) -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.2)
            )
        # L2 (?, 14, 14, 32) -> (?, 14, 14, 64) -> (?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.2)
            )
        # L3 (?, 7, 7, 64) -> (?, 7, 7, 128) -> (?, 4, 4, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.Dropout(p=0.2)
            )
        self._to_linear = 7 * 7 * 128

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1) # Flatten them for FC
        return x

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
        x = x.view(-1, 128, 7, 7)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.enc = Encoder()
        self.dec = Decoder()

    def forward(self, x):
        x = self.enc(x)
        return self.dec(x)
    
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    
model = AutoEncoder()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e8)

if torch.cuda.is_available():
    model = model.cuda()
    loss_function = loss_function.cuda()
        

epochs = 100
outputs = []
losses = []
for epoch in range(epochs):
    print("Epoch " + str(epoch) + " :" )
    for (image, _) in train_set:
        
        if torch.cuda.is_available():
            image = image.cuda()
        
        image = image.view(-1, 1, 28, 28) # Flatten them for FC 
        reconstructed = model(image)
        
        loss = loss_function(reconstructed, image)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss)
        
    outputs.append((epochs, image, reconstructed))

save_model(model, optimizer, epochs, './results/autoencoderMNIST.pth')

plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.plot(losses[-100:])

for i, item in enumerate(image):
   item = item.reshape(-1, 28, 28)
   plt.imshow(item[0])
   
for i, item in enumerate(reconstructed):
   item = item.reshape(-1, 28, 28)
   plt.imshow(item[0])