from torchvision import transforms, models
from torch.utils.data import DataLoader
import torch
from torch.optim import lr_scheduler
import cv2
import numpy as np
import time
import copy
import argparse

from dataset.crack_images import CrackImagesForClassification


class TrainConfig():
    def __init__(self):
        parser = argparse.ArgumentParser(
            description='arguments for training the deep model')
        parser.add_argument(
            "-e",
            dest='epochs',
            help="number of training epoch",
            default=1,
            type=int)
        args = parser.parse_known_args()[0]
        self.epochs = args.epochs


if __name__ == "__main__":
    config = TrainConfig()

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    crack_dataset = CrackImagesForClassification(
        root_dir='/home/aicenter/Documents/hsu/data/SDNET2018/', transforms=data_transforms)
    indices = torch.randperm(len(crack_dataset)).tolist()
    split_pos = int(len(crack_dataset)*0.7)
    train_crack = torch.utils.data.Subset(crack_dataset, indices[:split_pos])
    val_crack = torch.utils.data.Subset(crack_dataset, indices[split_pos:])

    print('Training set= %s images' % (len*(train_crack)))
    print('Validation set= %s images' % (len*(val_crack)))

    labels = ['uncracked', 'cracked']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloaders = {'train': DataLoader(train_crack, batch_size=4, shuffle=True, num_workers=4),
                   'val': DataLoader(val_crack, batch_size=4, shuffle=True, num_workers=4)}
    dataset_sizes = {'train': len(train_crack), 'val': len(val_crack)}

    # Parameters of model
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2 : crack/uncrack
    model_ft.fc = torch.nn.Linear(num_ftrs, 2)
    model_ft = model_ft.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_ft = torch.optim.SGD(
        model_ft.parameters(), lr=0.001, momentum=0.9)  # Observe that all parameters are being optimized
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1)  # Decay LR by a factor of 0.1 every 7 epochs
    num_epochs = config.epochs

    # train conv model
    since = time.time()
    best_model_wts = copy.deepcopy(model_ft.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model_ft.train()  # Set model to training mode
            else:
                model_ft.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer_ft.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_ft(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer_ft.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                exp_lr_scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model_ft.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model_ft.load_state_dict(best_model_wts)
    localtm = time.localtime()
    model_name = '%02d%02d%02d%02d.pkl' % (
        localtm.tm_mon, localtm.tm_mday, localtm.tm_hour, localtm.tm_min)
    torch.save(model_ft.state_dict(), 'models/' + model_name)
