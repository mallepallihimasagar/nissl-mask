import time
from collections import defaultdict
import torch.nn.functional as F
import torch
from loss import dice_loss
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torch.nn as nn

from nissl_dataset import Nissl_mask_dataset
from network import U_Net
from network import ResAttU_Net

# ------------------------parameters--------------------#
batch_size = 8
# ------------------------dataset-----------------------#
dataset = Nissl_mask_dataset()
dataset_len = dataset.__len__()
print(f"training with {dataset_len} images")
# train, val, test = random_split(dataset, [dataset_len-60, 30, 30])
# noinspection PyArgumentList
train, val, test = random_split(dataset, [dataset_len-60,40,20])
train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset=val, batch_size=batch_size//2, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test, batch_size=batch_size//2, shuffle=True, num_workers=4)

dataloaders = {
    'train': train_loader,
    'val': val_loader
}
# -----------------------training-----------------------#

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

num_class = 6
# -----------------------model--------------------------#
model = U_Net(UnetLayer=5, img_ch=3, output_ch=4).to(device)
# model2  = ResAttU_Net(UnetLayer=5,output_ch=4).to(device)


# freeze backbone layers
# Comment out to finetune further
# for l in model.base_layers:
#     for param in l.parameters():
#         param.requires_grad = False
model_path = "models/"
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=15)
torch.save(model.state_dict(),model_path)