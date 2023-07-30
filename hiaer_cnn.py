# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torchvision import datasets, transforms
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os
import time
import argparse
from spikingjelly import visualizing
from quant_network import Quantize_Network
from cri_converter import CRI_Converter
from bn_folder import BN_Folder
from torchsummary import summary
from l2s.api import CRI_network
import cri_simulations
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from helper import train, validate, run_CRI_hw
from models import FashionMnist, Mnist
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--resume_path', default='', type=str, help='checkpoint file')
parser.add_argument('--load_path', default='', type=str, help='checkpoint loading path')
parser.add_argument('--train', action='store_true', default=False, help='Train the network from stratch')
parser.add_argument('-b','--batch_size', default=32, type=int)
parser.add_argument('--data_path', default='/Volumes/export/isn/keli/code/data', type=str, help='path to dataset')
parser.add_argument('--out_dir', default='/Volumes/export/isn/keli/code/CRI/CRI_Mapping/runs/fashionMnist', type=str, help='dir path that stores the trained model checkpoint')
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('-m','--momentum', default=0.9, type=float)
parser.add_argument('-T','--num_steps', default=4, type=int)
parser.add_argument('-c','--channels', default=1, type=int)
parser.add_argument('--writer', action='store_true', default=False, help='Use torch summary')
parser.add_argument('--encoder',action='store_true',default=False, help='Using spike rate encoder to process the input')
parser.add_argument('--amp', action='store_true', default=False, help='Use mixed percision training')
parser.add_argument('--hardware',action='store_true', default=False, help='Run the network on FPGA')


def main():
    
    args = parser.parse_args()

    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    scaler = None
    if args.amp:
        scaler = amp.GradScaler()
        
    # #Prepare the dataset
    # mnist_train = datasets.MNIST(args.data_path, train=True, download=True, transform=transforms.Compose(
    #     [transforms.ToTensor()]))
    # mnist_test = datasets.MNIST(args.data_path, train=False, download=True, transform=transforms.Compose(
    #     [transforms.ToTensor()]))
     
    # Prepare the dataset
    mnist_train = datasets.FashionMNIST(args.data_path, train=True, download=True, transform=transforms.Compose(
        [transforms.ToTensor()]))
    mnist_test = datasets.FashionMNIST(args.data_path, train=False, download=True, transform=transforms.Compose(
        [transforms.ToTensor()]))
    
    # Create DataLoaders
    train_loader = DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # Initialize SnnTorch/SpikingJelly model
    # net_1 = Mnist()
    net_1 = FashionMnist()

    
    
    print(net_1)
    net_1.to(device)
    n_parameters = sum(p.numel() for p in net_1.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
    
    if args.resume_path != "" or args.train:
        print('Start Training')
        train(args, net_1, train_loader, test_loader, device, scaler)
    elif args.load_path != "":
        checkpoint = torch.load(args.load_path, map_location=device)
        net_1.load_state_dict(checkpoint['net'])
        # validate(args, net_1, test_loader, device)
  
    
    bn = BN_Folder()  #Fold the BN layer 
    net_bn = bn.fold(net_1.eval())
    # validate(args, net_bn, test_loader, device)
    
    quan_fun = Quantize_Network(w_alpha = 3,dynamic_alpha = False) # weight_quantization
    net_quan = quan_fun.quantize(net_bn)
    # validate(args, net_bn, test_loader, device)
    # print(net_quan.attn.attn_lif.v_threshold)
    
#     cri_convert = CRI_Converter(args.num_steps, 1, 11, (1, 28, 28),'spikingjelly', int(quan_fun.v_threshold), 4) # num_steps, input_layer, output_layer, input_size, backend, v_threshold, embed_dim
#     cri_convert.layer_converter(net_quan)
    
#     config = {}
#     config['neuron_type'] = "I&F"
#     config['global_neuron_params'] = {}
#     config['global_neuron_params']['v_thr'] = int(quan_fun.v_threshold)
    
#     hardwareNetwork, softwareNetwork = None, None
#     if args.hardware:
#         hardwareNetwork = CRI_network(axons=dict(cri_convert.axon_dict),connections=dict(cri_convert.neuron_dict),config=config,target='CRI', outputs = cri_convert.output_neurons,coreID=1)
#     else:
#         softwareNetwork = CRI_network(axons=dict(cri_convert.axon_dict),connections=dict(cri_convert.neuron_dict),config=config,target='simpleSim', outputs = cri_convert.output_neurons,coreID=1)

#     cri_convert.bias_start_idx = int(cri_convert.output_neurons[0])#add this to the end of conversion
#     loss_fun = nn.MSELoss()
#     start_time = time.time()
#     test_loss = 0
#     test_acc = 0
#     test_samples = 0
#     for img, label in tqdm(test_loader):
#         cri_input = cri_convert.input_converter(img)
#         # label_onehot = F.one_hot(label, 10).float()
#         output = None
#         if args.hardware:
#             output = torch.tensor(cri_convert.run_CRI_hw(cri_input,hardwareNetwork), dtype=float)
#         else:
#             output = torch.tensor(cri_convert.run_CRI_sw(cri_input,softwareNetwork), dtype=float)
        
#         print(f'output: {output}, label: {label}')
#         loss = loss_fun(output, label)
#         # print(f'loss: {loss}')
#         test_samples += label.numel()
#         test_loss += loss.item() * label.numel()
#         test_acc += (output == label).float().sum().item()
#         print(f'test_acc: {test_acc}')
#         break

#     test_time = time.time()
#     test_speed = test_samples / (test_time - start_time)
#     test_loss /= test_samples
#     test_acc /= test_samples

#     print(f'test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}')
#     print(f'test speed ={test_speed: .4f} images/s')
    
if __name__ == '__main__':
    main()