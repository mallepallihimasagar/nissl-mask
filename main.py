from nissl_dataset import Nissl_mask_dataset
import matplotlib.pyplot as plt
from network import U_Net

from network import ResAttU_Net
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from tqdm import tqdm

# ------------------------parameters--------------------#
batch_size = 4
# ------------------------dataset-----------------------#
dataset = Nissl_mask_dataset()
dataset_len = dataset.__len__()
print(dataset_len)
# train, val, test = random_split(dataset, [dataset_len-60, 30, 30])
train, val, test = random_split(dataset, [1,dataset_len-31, 30])
train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(dataset=val, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True, num_workers=2)
# -----------------------model--------------------------#
model = U_Net(UnetLayer=5, img_ch=3, output_ch=4)
# model2  = ResAttU_Net(UnetLayer=5,output_ch=4)
# -----------------------training-----------------------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Model Created , training on {device}")

def dice_coeff_loss(Prediction_vector, GT_vector):
    smooth = 1
    intersection = (GT_vector * Prediction_vector).sum()
    return 1 - (2. * intersection + smooth) / (GT_vector.sum() + Prediction_vector.sum() + smooth)


def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()
    print("Training Started")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    model.train()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)


        running_loss = 0.0
        loop = tqdm(enumerate(train_loader),total=len(test_loader),leave=False)
        for batch_idx ,(input_images, target_masks) in loop:
            inputs = input_images
            labels = target_masks
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs.type(torch.float))
            print(f"input shape : {inputs.shape}")
            print(f"target shape : {labels.shape}")
            print(f"predicted shape : {outputs.shape}")
            # Flatten the prediction, target, and training field.
            SR = torch.sigmoid(outputs)
            GT = labels.type(torch.float)
            SR_flat = SR.view(SR.size(0), -1)
            GT_flat = GT.view(GT.size(0), -1)

            #loss = criterion(SR_flat, GT_flat)
            loss = dice_coeff_loss(SR_flat, GT_flat)

            running_loss += loss.item()  #* inputs.size(0)
            loss.backward()

            optimizer.step()

            loop.set_description(f"Epoch [{epoch}/{num_epochs}],Batch [{batch_idx}/{dataset_len//batch_size}]")
            loop.set_postfix(loss=loss.item())



        epoch_loss = running_loss / (dataset_len//batch_size)

        print('Loss: {:.4f}'.format(epoch_loss))

        # deep copy the model
        # if phase == 'val' and epoch_loss < best_loss:
        #     best_loss = epoch_loss
        #     best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    model_ft = train_model(model, criterion, optimizer_ft, num_epochs=1)
