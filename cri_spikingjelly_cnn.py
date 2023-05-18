# imports
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os
import time
import argparse
from torch.cuda import amp
import sys
import datetime
from spikingjelly import visualizing
import shutil
from quant_layer import *
from cri_converter import BN_Folder, Quantize_Network, CRI_Converter
from torchsummary import summary
from timm.models import resume_checkpoint
#from spikingjelly.activation_based.model import spiking_resnet
import getopt
# from cri_test import train

class CIFAR10(nn.Module):
    def __init__(self, T: int, channels: int, features: int, use_cupy=False):
        super().__init__()
        self.T = T

        self.conv_fc = nn.Sequential(
        layer.Conv2d(channels, channels*features, kernel_size=3, padding=1, bias=False),
        layer.BatchNorm2d(channels*features),
        neuron.IFNode(surrogate_function=surrogate.ATan()),
        
        layer.Conv2d(channels*features, channels*features*2, kernel_size=3, padding=1, bias=False),
        layer.BatchNorm2d(channels*features*2),
        neuron.IFNode(surrogate_function=surrogate.ATan()),
            
        layer.Conv2d(channels*features*2, channels*features*4, kernel_size=3, padding=1, bias=False),
        layer.BatchNorm2d(channels*features*4),
        neuron.IFNode(surrogate_function=surrogate.ATan()),
            
        layer.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),  # 14 * 14
            
        
        layer.Conv2d(channels*features*4, channels*features*8, kernel_size=3, padding=1, bias=False),
        layer.BatchNorm2d(channels*features*8),
        neuron.IFNode(surrogate_function=surrogate.ATan()),
        
        layer.Conv2d(channels*features*8, channels*features*16, kernel_size=3, padding=1, bias=False),
        layer.BatchNorm2d(channels*features*16),
        neuron.IFNode(surrogate_function=surrogate.ATan()),
            
        layer.Conv2d(channels*features*16, channels*features*8, kernel_size=3, padding=1, bias=False),
        layer.BatchNorm2d(channels*features*8),
        neuron.IFNode(surrogate_function=surrogate.ATan()),
            
        layer.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),  # 14 * 14

        layer.Flatten(),
        layer.Dropout(0.5),
        layer.Linear(channels*features*8*64, 1000, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),
        
        layer.Dropout(0.5),
        layer.Linear(1000, 100, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),
    
        layer.Linear(100, 10, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),
        )
        
        functional.set_step_mode(self, step_mode='m')
        
    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        x_seq = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        x_seq = self.conv_fc(x_seq)
        fr = x_seq.mean(0)
        return fr

def validate(net, test_loader, device, out_dir):
    start_time = time.time()
    net.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    
    # writer = SummaryWriter(out_dir)
    
    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(device)
            label = label.to(device)
            label_onehot = F.one_hot(label, 10).float()
            out_fr = net(img)
            loss = F.cross_entropy(out_fr, label_onehot)

            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (out_fr.argmax(1) == label).float().sum().item()
            functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - start_time)
        test_loss /= test_samples
        test_acc /= test_samples
        # writer.add_scalar('test_loss', test_loss)
        # writer.add_scalar('test_acc', test_acc)
    
    print(f'test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}')
    print(f'test speed ={test_speed: .4f} images/s')

def validate_cri(test_loader, out_dir):
    start_time = time.time()
    net.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    
    # writer = SummaryWriter(out_dir+'/cri')
    
    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(device)
            label = label.to(device)
            label_onehot = F.one_hot(label, 10).float()
            out_fr = net(img)
            loss = F.cross_entropy(out_fr, label_onehot)

            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (out_fr.argmax(1) == label).float().sum().item()
            functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - start_time)
        test_loss /= test_samples
        test_acc /= test_samples
        # writer.add_scalar('test_loss', test_loss)
        # writer.add_scalar('test_acc', test_acc)
    
    print(f'test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}')
    print(f'test speed ={test_speed: .4f} images/s')
    
def main(argv):
    # resume_path, load_path, train_flag = "", "", False
    # # args = argv.split()
    # opts, args = getopt.getopt(argv,"tr:l:",["resume=", "load="], )
    # for opt, arg in opts:
    #     if opt in ('-r','--resume'):
    #         resume_path = str(arg)
    #     elif opt in ('-l', '--load'):
    #         load_path = str(arg)
    #     elif opt == '-t':
    #         train_flag = True 
            
    # dataloader arguments
    batch_size = 128
    data_path='/Volumes/export/isn/keli/code/data'
    out_dir = 'runs/cifar10'
    epochs = 10
    start_epoch = 0
    lr = 1e-1 #change the learning rate
    momentum = 0.9
    T = 4
    channels = 3
    features = 2
    max_test_acc = -1
    
    resume_path = ""
    
    opts, args = getopt.getopt(argv,"r:",["resume="])
    for opt, arg in opts:
        if opt in ('-r','--resume'):
            resume_path = str(arg)



    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # Prepare the dataset
    cifar_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
    cifar_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=True, drop_last=True)

    net = CIFAR10(T = T, channels = channels, features = features, use_cupy=False)
    print("before model: ", torch.cuda.memory_allocated())
    net.to(device)
    print("After model: ", torch.cuda.memory_allocated())
    print(net)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
#     net.to(device)
    
#     if resume_path != "" or train_flag:
#         print('Start Training')
#         train(argv, net, train_loader, test_loader, device, out_dir, epochs, resume_path)
#     elif load_path != "":
#         checkpoint = torch.load(load_path, map_location=device)
#         net.load_state_dict(checkpoint['net'])
#         validate(net, test_loader, device, out_dir)

    
    if resume_path != "":
        checkpoint = torch.load(resume_path, map_location=device)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        max_test_acc = checkpoint['max_test_acc']
        
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")

    
    # writer = SummaryWriter(out_dir)

    #Training
    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_loader:
            optimizer.zero_grad()
            img = img.to(device)
            label = label.to(device)
            label_onehot = F.one_hot(label, 10).float()
            out_fr = net(img)
            loss = F.cross_entropy(out_fr, label_onehot)
            loss.backward()
            optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        # writer.add_scalar('train_loss', train_loss, epoch)
        # writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0

        with torch.no_grad():
            for img, label in test_loader:
                img = img.to(device)
                label = label.to(device)
                label_onehot = F.one_hot(label, 10).float()
                out_fr = net(img)
                loss = F.cross_entropy(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
                
            test_time = time.time()
            test_speed = test_samples / (test_time - train_time)
            test_loss /= test_samples
            test_acc /= test_samples
            # writer.add_scalar('test_loss', test_loss, epoch)
            # writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')
    
    
    # #Save the trained model
    # if not os.path.exists('result'):
    #     os.makedirs('result')
    # fdir = 'result/'
    # if not os.path.exists(fdir):
    #     os.makedirs(fdir)
    # save_checkpoint({'state_dict': net.state_dict(),}, 0, fdir, len(net.state_dict()))

if __name__ == '__main__':
    main(sys.argv[1:])
