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
from cri_spikingjelly_train import CSNN, validate, save_checkpoint
from l2s.api import CRI_network
import cri_simulations

# def run_CRI_hw(inputList,output_offset,hardwareNetwork):
#     predictions = []
#     #each image
#     total_time_cri = 0
#     for currInput in inputList:
#         #initiate the softwareNetwork for each image
#         cri_simulations.FPGA_Execution.fpga_controller.clear(len(neuronsDict), False, 0)  ##Num_neurons, simDump, coreOverride
#         spikeRate = [0]*10
#         #each time step
#         for slice in currInput:
#             start_time = time.time()
#             hwSpike = hardwareNetwork.step(slice, membranePotential=False)
# #             print("Mem:",mem)
#             end_time = time.time()
#             total_time_cri = total_time_cri + end_time-start_time
#             print(hwSpike)
#             for spike in hwSpike:
#                 print(int(spike))
#                 spikeIdx = int(spike) - output_offset 
#                 if spikeIdx >= 0: 
#                     spikeRate[spikeIdx] += 1 
#         predictions.append(spikeRate.index(max(spikeRate))) 
#     print(f"Total execution time CRIFPGA: {total_time_cri:.5f} s")
#     return(predictions)

# def validate_cri(converter, net, hardwareNetwork,test_loader, out_dir, device,output_offset):
#     start_time = time.time()
#     net.eval() #TODO: add Net as paramter
#     test_loss = 0
#     test_acc = 0
#     test_samples = 0
#     test_loss_cri = 0
#     test_acc_cri = 0
    
    
#     writer = SummaryWriter(out_dir+'/cri')
#     with torch.no_grad():
#         for img, label in test_loader:

#             cri_input = converter.input_converter(img)
#             print(f'cri_input: {cri_input}')
#             img = img.to(device)
#             label = label.to(device)
#             label_onehot = F.one_hot(label, 10).float()
#             out_fr = net(img)
#             cri_out = torch.tensor(run_CRI_hw(cri_input,output_offset,hardwareNetwork)).to(device)
#             loss = F.mse_loss(out_fr, label_onehot)
#             loss_cri = F.mse_loss(cri_out, label_onehot)

#             test_samples += label.numel()
#             test_loss += loss.item() * label.numel()
#             test_loss_cri += loss_cri.item() * label.numel()
            
#             test_acc += (out_fr.argmax(1) == label).float().sum().item()
#             test_acc_cri += (cri_out.argmax(1) == label).float().sum().item()
#             functional.reset_net(net)
        
#         test_time = time.time()
#         test_speed = test_samples / (test_time - start_time)
#         test_loss /= test_samples
#         test_acc /= test_samples
#         writer.add_scalar('test_loss', test_loss)
#         writer.add_scalar('test_acc', test_acc)
#         writer.add_scalar('test_loss_cri', test_loss)
#         writer.add_scalar('test_acc_cri', test_acc)
    
#     print(f'test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}')
#     print(f'test speed ={test_speed: .4f} images/s')
    

def main():
    # dataloader arguments
    batch_size = 128
    data_path='~/code/data/mnist'
    out_dir = 'runs/transformers'
    epochs = 25
    start_epoch = 0
    lr = 0.1
    momentum = 0.9
    T = 4
    channels = 8
    max_test_acc = -1

    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    
    # Prepare the dataset
    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transforms.ToTensor())

    # Create DataLoaders
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)
    
    best_model_path = '/home/keli/code/CRI_Mapping/result/model_spikingjelly_14_s.pth.tar'
    checkpoint = torch.load(best_model_path, map_location=device)
    print(checkpoint['state_dict'].keys())
    net_1 = CSNN(T = T, channels = channels, use_cupy=False)
    net_1.load_state_dict(checkpoint['state_dict'])
    print(net_1)
    net_1.eval()
    net_1.to(device)
    validate(net_1, test_loader, device, out_dir)
    
    bn = BN_Folder()  #Fold the BN layer 
    net_bn = bn.fold(net_1.eval())
#     # validate(net_bn, test_loader, device, out_dir)
    
    quan_fun = Quantize_Network(dynamic_alpha = False) # weight_quantization
    net_quan = quan_fun.quantize(net_bn)
    net_quan.to(device)
    validate(net_quan, test_loader, device, out_dir)
    
    cri_convert = CRI_Converter(4, 0, 11, np.array(( 1, 28, 28)),'spikingjelly', int(quan_fun.v_threshold)) # num_steps, input_layer, output_layer, input_size
    cri_convert.layer_converter(net_quan)
    
    config = {}
    config['neuron_type'] = "I&F"
    config['global_neuron_params'] = {}
    config['global_neuron_params']['v_thr'] = int(quan_fun.v_threshold)
    
    hardwareNetwork = CRI_network(axons=cri_convert.axon_dict,connections=cri_convert.neuron_dict,config=config,target='CRI', outputs = cri_convert.output_neurons,coreID=1)
    
    cri_convert.bias_start_idx = int(cri_convert.output_neurons[0])

    cri_convert.validate_cri(net_1, hardwareNetwork, test_loader, out_dir, device)
    
    
    
if __name__ == '__main__':
    main()

    