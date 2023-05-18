# imports
# import matplotlib.pyplot as plt
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
from cri_spikingjelly_train import CSNN, validate, save_checkpoint
from l2s.api import CRI_network
import cri_simulations
from cri_model import SSA,SPS
import getopt
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

#TODO: Train a small nn on mnist with residual connection and self attention

class Net(nn.Module):
    def __init__(self, T=1, embed_dims=4):
        super().__init__()
        self.T = T
        self.embed_dims = embed_dims
        
        self.proj_conv = nn.Conv2d(1, self.embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(self.embed_dims)
        self.proj_lif = neuron.LIFNode(tau=2.0,surrogate_function=surrogate.ATan())
        self.avgpool = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        
        # self.patch_embed = SPS(img_size_h=28,img_size_w=28, patch_size=4,in_channels=1,embed_dims=self.embed_dims)
        
        self.attn = SSA(dim=self.embed_dims,num_heads=1)
        self.linear = nn.Linear(self.embed_dims * 7 * 7,10)
        
    def forward(self,x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        # print(x.shape)
        T, B, C, H, W = x.shape
        x = self.proj_conv(x.flatten(0, 1)) # TB, C, H, W
        # print(f"proj_conv.shape: {x.shape}")
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous() 
        x = self.proj_lif(x).flatten(0, 1).contiguous() #TB, C, H, W
        x = self.avgpool(x)
        # print(f"maxpool.shape: {x.shape}")
        x = x.reshape(T, B, -1, self.embed_dims) #T, B, N, D
        
        # x = self.patch_embed(x)
        # print(x.shape)
        
        x = self.attn(x)
        # print(f"attn.shape: {x.shape}")
        x = self.linear(x.flatten(-2,-1))
        # print(f"linear: {x.shape}")
        
        return x.mean(0)
    
def train(argv, net, train_loader, test_loader, device, out_dir, epochs, resume_path=""):
    
    lr = 1e-2
    momentum = 0.9
    start_epoch = 0
    max_test_acc = -1
   
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
            
    if resume_path != "":
        checkpoint = torch.load(resume_path, map_location=device)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        max_test_acc = checkpoint['max_test_acc']
    
    # writer = SummaryWriter(out_dir)
    
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
    

    
def main(argv):
    resume_path, load_path, train_flag = "", "", False
    # args = argv.split()
    opts, args = getopt.getopt(argv,"tr:l:",["resume=", "load="], )
    for opt, arg in opts:
        if opt in ('-r','--resume'):
            resume_path = str(arg)
        elif opt in ('-l', '--load'):
            load_path = str(arg)
        elif opt == '-t':
            train_flag = True 
            
            
    # dataloader arguments
    batch_size = 128
    data_path='~/code/data/mnist'
    out_dir = 'runs/test'
    epochs = 15
    # lr = 0.1
    # momentum = 0.9
    # T = 4
    # channels = 8

    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    
    # Prepare the dataset
    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transforms.Compose(
        [transforms.Resize(14), transforms.ToTensor()]))
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transforms.Compose(
        [transforms.Resize(14), transforms.ToTensor()]))
    # Create DataLoaders
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # best_model_path = '/home/keli/code/CRI_Mapping/result/model_spikingjelly_14_s.pth.tar'
    # checkpoint = torch.load(best_model_path, map_location=device)
    # print(checkpoint['state_dict'].keys())
    
    
    net_1 = Net(T=1, embed_dims=2)
    
    print(net_1)
    # net_1.eval()
    net_1.to(device)
    n_parameters = sum(p.numel() for p in net_1.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
    
    if resume_path != "" or train_flag:
        print('Start Training')
        train(argv, net_1, train_loader, test_loader, device, out_dir, epochs, resume_path)
    elif load_path != "":
        checkpoint = torch.load(load_path, map_location=device)
        net_1.load_state_dict(checkpoint['net'])
        validate(net_1, test_loader, device, out_dir)
  
    
    bn = BN_Folder()  #Fold the BN layer 
    net_bn = bn.fold(net_1.eval())
    validate(net_bn, test_loader, device, out_dir)
    
    quan_fun = Quantize_Network(w_alpha = 10,dynamic_alpha = False) # weight_quantization
    net_quan = quan_fun.quantize(net_bn)
    # net_quan.to(device)
    validate(net_quan, test_loader, device, out_dir)
    # print(net_quan.attn.attn_lif.v_threshold)
    
    cri_convert = CRI_Converter(1, 0, 20, np.array(( 1, 14, 14)),'spikingjelly', int(quan_fun.v_threshold), 4) # num_steps, input_layer, output_layer, input_size, backend, v_threshold, embed_dim
    cri_convert.layer_converter(net_quan)
    
    config = {}
    config['neuron_type'] = "I&F"
    config['global_neuron_params'] = {}
    config['global_neuron_params']['v_thr'] = int(quan_fun.v_threshold)
    
    hardwareNetwork = CRI_network(axons=cri_convert.axon_dict,connections=cri_convert.neuron_dict,config=config,target='CRI', outputs = cri_convert.output_neurons,coreID=1)
    
    cri_convert.bias_start_idx = int(cri_convert.output_neurons[0])#add this to the end of conversion

    cri_convert.run(hardwareNetwork, test_loader, loss_fun)
    
    
    
if __name__ == '__main__':
    main(sys.argv[1:])