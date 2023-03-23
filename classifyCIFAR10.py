import os
from copy import deepcopy

## Imports for plotting
import matplotlib.pyplot as plt
plt.set_cmap('cividis')
# %matplotlib inline
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.set()

## tqdm for loading bars
from tqdm.notebook import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

## Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms

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
        # The MLP for g(.) consists of Linear->ReLU->Linear 
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4*hidden_dim, hidden_dim)
        )

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

# Import tensorboard
# %load_ext tensorboard

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "./data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "../results/CIFAR10"
# In this notebook, we use data loaders with heavier computational processing. It is recommended to use as many
# workers as possible in a data loader, which corresponds to the number of CPU cores
NUM_WORKERS = os.cpu_count()

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
print("Number of workers:", NUM_WORKERS)


class LogisticRegression(pl.LightningModule):
    
    def __init__(self, feature_dim, num_classes, lr, weight_decay, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        # Mapping from representation h to classes
        self.model = nn.Linear(feature_dim, num_classes)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), 
                                lr=self.hparams.lr, 
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                                      milestones=[int(self.hparams.max_epochs*0.6), 
                                                                  int(self.hparams.max_epochs*0.8)], 
                                                      gamma=0.1)
        return [optimizer], [lr_scheduler]
        
    def _calculate_loss(self, batch, mode='train'):
        feats, labels = batch
        preds = self.model(feats)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(mode + '_loss', loss)
        self.log(mode + '_acc', acc)
        return loss        
        
    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode='train')
        
    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='val')
        
    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='test')

"""The data we use is the training and test set of CIFAR10. The training contains 500 images per class, while the test set has 800 images per class."""

img_transforms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5,), (0.5,))])

train_img_data = CIFAR10(root=DATASET_PATH, train=True, download=True,
                       transform=img_transforms)
test_img_data = CIFAR10(root=DATASET_PATH, train=False, download=True,
                      transform=img_transforms)

print("Number of training examples:", len(train_img_data))
print("Number of test examples:", len(test_img_data))

"""Next, we implement a small function to encode all images in our datasets. The output representations are then used as inputs to the Logistic Regression model."""

@torch.no_grad()
def prepare_data_features(model, dataset):
    # Prepare model
    network = deepcopy(model.convnet)
    network.fc = nn.Identity()  # Removing projection head g(.)
    network.eval()
    network.to(device)
    
    # Encode all images
    data_loader = data.DataLoader(dataset, batch_size=64, num_workers=NUM_WORKERS, shuffle=False, drop_last=False)
    feats, labels = [], []
    for batch_imgs, batch_labels in tqdm(data_loader):
        batch_imgs = batch_imgs.to(device)
        batch_feats = network(batch_imgs)
        feats.append(batch_feats.detach().cpu())
        labels.append(batch_labels)
    
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    
    # Sort images by labels
    labels, idxs = labels.sort()
    feats = feats[idxs]
    
    return data.TensorDataset(feats, labels)


simclr_model = SimCLR( 
                    hidden_dim=384, 
                    lr=5e-4, 
                    temperature=0.07, 
                    weight_decay=1e-4, 
                    max_epochs=100)

simclr_model.convnet.load_state_dict(
    torch.load('./results/simclrCIFAR10-384.pt')
)

"""Let's apply the function to both training and test set below."""

train_feats_simclr = prepare_data_features(simclr_model, train_img_data)
test_feats_simclr = prepare_data_features(simclr_model, test_img_data)

"""Finally, we can write a training function as usual. We evaluate the model on the test set every 10 epochs to allow early stopping, but the low frequency of the validation ensures that we do not overfit too much on the test set."""

def train_logreg(batch_size, train_feats_data, test_feats_data, model_suffix, max_epochs=100, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "LogisticRegression"),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc'),
                                    LearningRateMonitor("epoch")],
                         enable_progress_bar=False,
                         check_val_every_n_epoch=10)
    trainer.logger._default_hp_metric = None
    
    # Data loaders
    train_loader = data.DataLoader(train_feats_data, batch_size=batch_size, shuffle=True, 
                                   drop_last=False, pin_memory=True, num_workers=0)
    test_loader = data.DataLoader(test_feats_data, batch_size=batch_size, shuffle=False, 
                                  drop_last=False, pin_memory=True, num_workers=0)

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"LogisticRegression_{model_suffix}.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # model = LogisticRegression.load_from_checkpoint(pretrained_filename)
    else:
        print("Training logreg cifar10")
        pl.seed_everything(42)  # To be reproducable
        model = LogisticRegression(**kwargs)
        trainer.fit(model, train_loader, test_loader)
        
        torch.save(model.model.state_dict(), "logregCIFAR10.pt")
        
        trainer.save_checkpoint("'./results/CIFAR10/logregCIFAR10.ckpt")
        model = LogisticRegression.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on train and validation set
    train_result = trainer.test(model, train_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"train": train_result[0]["test_acc"], "test": test_result[0]["test_acc"]}
        
    return model, result

"""Despite the training dataset of CIFAR10 already only having 500 labeled images per class, we will perform experiments with even smaller datasets. Specifically, we train a Logistic Regression model for datasets with only 10, 20, 50, 100, 200, and all 500 examples per class. This gives us an intuition on how well the representations learned by contrastive learning can be transfered to a image recognition task like this classification. First, let's define a function to create the intended sub-datasets from the full training set:"""

def get_smaller_dataset(original_dataset, num_imgs_per_label):
    new_dataset = data.TensorDataset(
        *[t.unflatten(0, (10, -1))[:,:num_imgs_per_label].flatten(0, 1) for t in original_dataset.tensors]
    )
    return new_dataset

"""Next, let's run all models. Despite us training 6 models, this cell could be run within a minute or two without the pretrained models. """

results = {}
for num_imgs_per_label in [10, 20, 50, 100, 200, 500, 2000, 5000]:
    sub_train_set = get_smaller_dataset(train_feats_simclr, num_imgs_per_label)
    _, small_set_results = train_logreg(batch_size=64,
                                        train_feats_data=sub_train_set,
                                        test_feats_data=test_feats_simclr,
                                        model_suffix=num_imgs_per_label,
                                        feature_dim=train_feats_simclr.tensors[0].shape[1],
                                        num_classes=10,
                                        lr=1e-3,
                                        weight_decay=1e-3)
    results[num_imgs_per_label] = small_set_results

"""Finally, let's plot the results."""

dataset_sizes = sorted([k for k in results])
test_scores = [results[k]["test"] for k in dataset_sizes]

fig = plt.figure(figsize=(6,4))
plt.plot(dataset_sizes, test_scores, '--', color="#000", marker="*", markeredgecolor="#000", markerfacecolor="y", markersize=16)
plt.xscale("log")
plt.xticks(dataset_sizes, labels=dataset_sizes)
plt.title("CIFAR10 classification over dataset size", fontsize=14)
plt.xlabel("Number of images per class")
plt.ylabel("Test accuracy")
plt.minorticks_off()
plt.savefig('./figures/CIFAR10test.png', format="png")
plt.show()

dataset_sizes = sorted([k for k in results])
train_scores = [results[k]["train"] for k in dataset_sizes]

fig = plt.figure(figsize=(6,4))
plt.plot(dataset_sizes, train_scores, '--', color="#000", marker="*", markeredgecolor="#000", markerfacecolor="y", markersize=16)
plt.xscale("log")
plt.xticks(dataset_sizes, labels=dataset_sizes)
plt.title("CIFAR10 classification over dataset size", fontsize=14)
plt.xlabel("Number of images per class")
plt.ylabel("Train accuracy")
plt.minorticks_off()
plt.savefig('./figures/CIFAR10train.png', format="png")
plt.show()


for k, score in zip(dataset_sizes, test_scores):
    print(f'Test accuracy for {k:3d} images per label: {100*score:4.2f}%')
for k, score in zip(dataset_sizes, train_scores):
    print(f'Test accuracy for {k:3d} images per label: {100*score:4.2f}%')

"""As one would expect, the classification performance improves the more data we have. However, with only 10 images per class, we can already classify more than 60% of the images correctly. This is quite impressive, considering that the images are also higher dimensional than e.g. CIFAR10. With the full dataset, we achieve an accuracy of 81%. The increase between 50 to 500 images per class might suggest a linear increase in performance with an exponentially larger dataset. However, with even more data, we could also finetune $f(\cdot)$ in the training process, allowing for the representations to adapt more to the specific classification task given.

To set the results above into perspective, we will train the base network, a ResNet-18, on the classification task from scratch.

## Baseline

As a baseline to our results above, we will train a standard ResNet-18 with random initialization on the labeled training set of CIFAR10. The results will give us an indication of the advantages that contrastive learning on unlabeled data has compared to using only supervised training. The implementation of the model is straightforward since the ResNet architecture is provided in the torchvision library.
"""

# class ResNet(pl.LightningModule):
    
#     def __init__(self, num_classes, lr, weight_decay, max_epochs=100):
#         super().__init__()
#         self.save_hyperparameters()
#         self.model = torchvision.models.resnet18(num_classes=num_classes)

#     def configure_optimizers(self):
#         optimizer = optim.AdamW(self.parameters(), 
#                                 lr=self.hparams.lr, 
#                                 weight_decay=self.hparams.weight_decay)
#         lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
#                                                       milestones=[int(self.hparams.max_epochs*0.7), 
#                                                                   int(self.hparams.max_epochs*0.9)], 
#                                                       gamma=0.1)
#         return [optimizer], [lr_scheduler]

#     def _calculate_loss(self, batch, mode='train'):
#         imgs, labels = batch
#         preds = self.model(imgs)
#         loss = F.cross_entropy(preds, labels)
#         acc = (preds.argmax(dim=-1) == labels).float().mean()

#         self.log(mode + '_loss', loss)
#         self.log(mode + '_acc', acc)
#         return loss

#     def training_step(self, batch, batch_idx):
#         return self._calculate_loss(batch, mode='train')

#     def validation_step(self, batch, batch_idx):
#         self._calculate_loss(batch, mode='val')

#     def test_step(self, batch, batch_idx):
#         self._calculate_loss(batch, mode='test')

# """It is clear that the ResNet easily overfits on the training data since its parameter count is more than 1000 times larger than the dataset size. To make the comparison to the contrastive learning models fair, we apply data augmentations similar to the ones we used before: horizontal flip, crop-and-resize, grayscale, and gaussian blur. Color distortions as before are not used because the color distribution of an image showed to be an important feature for the classification. Hence, we observed no noticeable performance gains when adding color distortions to the set of augmentations. Similarly, we restrict the resizing operation before cropping to the max. 125% of its original resolution, instead of 1250% as done in SimCLR. This is because, for classification, the model needs to recognize the full object, while in contrastive learning, we only want to check whether two patches belong to the same image/object. Hence, the chosen augmentations below are overall weaker than in the contrastive learning case."""

# train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                        transforms.RandomResizedCrop(size=96, scale=(0.8, 1.0)),
#                                        transforms.RandomGrayscale(p=0.2),
#                                        transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 0.5)),
#                                        transforms.ToTensor(),
#                                        transforms.Normalize((0.5,), (0.5,))
#                                        ])

# train_img_aug_data = CIFAR10(root=DATASET_PATH, train=True, download=True,
#                            transform=train_transforms)

# """The training function for the ResNet is almost identical to the Logistic Regression setup. Note that we allow the ResNet to perform validation every 2 epochs to also check whether the model overfits strongly in the first iterations or not."""

# def train_resnet(batch_size, max_epochs=100, **kwargs):
#     trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "ResNet"),
#                          accelerator="gpu" if str(device).startswith("cuda") else "cpu",
#                          devices=1,
#                          max_epochs=max_epochs,
#                          callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
#                                     LearningRateMonitor("epoch")],
#                          check_val_every_n_epoch=2)
#     trainer.logger._default_hp_metric = None
    
#     # Data loaders
#     train_loader = data.DataLoader(train_img_aug_data, batch_size=batch_size, shuffle=True, 
#                                    drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
#     test_loader = data.DataLoader(test_img_data, batch_size=batch_size, shuffle=False, 
#                                   drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)

#     # Check whether pretrained model exists. If yes, load it and skip training
#     pretrained_filename = os.path.join(CHECKPOINT_PATH, "ResNet.ckpt")
#     if os.path.isfile(pretrained_filename):
#         print("Found pretrained model at %s, loading..." % pretrained_filename)
#         model = ResNet.load_from_checkpoint(pretrained_filename)
#     else:
#         pl.seed_everything(42) # To be reproducable
#         model = ResNet(**kwargs)
#         trainer.fit(model, train_loader, test_loader)
#         model = ResNet.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

#     # Test best model on validation set
#     train_result = trainer.test(model, train_loader, verbose=False)
#     val_result = trainer.test(model, test_loader, verbose=False)
#     result = {"train": train_result[0]["test_acc"], "test": val_result[0]["test_acc"]}
        
#     return model, result

# """Finally, let's train the model and check its results:"""

# resnet_model, resnet_result = train_resnet(batch_size=64,
#                                            num_classes=10,
#                                            lr=1e-3,
#                                            weight_decay=2e-4,
#                                            max_epochs=100)
# print(f"Accuracy on training set: {100*resnet_result['train']:4.2f}%")
# print(f"Accuracy on test set: {100*resnet_result['test']:4.2f}%")