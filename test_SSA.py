# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torchvision import datasets, transforms
from spikingjelly.activation_based import neuron, functional, surrogate, layer, encoding
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
from hs_api.api import CRI_network
#import hs_bridge
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from utils import train, validate
from models import SSA
from tqdm import tqdm
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--resume_path', default='', type=str, help='checkpoint file')
parser.add_argument('--load_path', default='', type=str, help='checkpoint loading path')
parser.add_argument('--train', action='store_true', default=False, help='Train the network from stratch')
parser.add_argument('-b','--batch_size', default=32, type=int)

# parser.add_argument('--data_path', default='C:\\Users\\KeliWang\\CRI_Mapping\\data', type=str, help='path to dataset')
# parser.add_argument('--out_dir', default='C:\\Users\\KeliWang\\CRI_Mapping\\runs\\ssa', type=str, help='dir path that stores the trained model checkpoint')

parser.add_argument('--data_path', default='/Volumes/export/isn/keli/code/data', type=str, help='path to dataset')
parser.add_argument('--out_dir', default='/Volumes/export/isn/keli/code/HS/CRI_Mapping/runs/ssa', type=str, help='dir path that stores the trained model checkpoint')

parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('-m','--momentum', default=0.9, type=float)
parser.add_argument('-T','--num_steps', default=4, type=int)
parser.add_argument('-c','--channels', default=1, type=int)
parser.add_argument('--writer', action='store_true', default=False, help='Use torch summary')
parser.add_argument('--encoder',action='store_true',default=True, help='Using spike rate encoder to process the input')
parser.add_argument('--amp', action='store_true', default=False, help='Use mixed percision training')
parser.add_argument('--hardware',action='store_true', default=False, help='Run the network on FPGA')
parser.add_argument('--num_batches', default=4, type=int)

def main():
    
    args = parser.parse_args()

    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    scaler = None
    if args.amp:
        scaler = amp.GradScaler()
        
    #Prepare the dataset
    mnist_train = datasets.MNIST(args.data_path, train=True, download=True, transform=transforms.Compose(
        [transforms.ToTensor()]))
    mnist_test = datasets.MNIST(args.data_path, train=False, download=True, transform=transforms.Compose(
        [transforms.ToTensor()]))
     
    # Create DataLoaders
    train_loader = DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # Initialize SnnTorch/SpikingJelly model
    N = 4
    
    net = SSA(N = 4)
    net_test = SSA(N = 4)
    net_mul = SSA(N =4)
    
    
    # print(net_1)
    net.to(device)
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
    # print(net)
    
    if args.resume_path != "" or args.train:
        print('Start Training')
        train(args, net, train_loader, test_loader, device, scaler)
    elif args.load_path != "":
        checkpoint = torch.load(args.load_path, map_location=device)
        net.load_state_dict(checkpoint['net'])
        # Testing for conversion: 
        net_test.load_state_dict(checkpoint['net'])
        net_mul.load_state_dict(checkpoint['net'])
        validate(args, net, test_loader, device)
  
    
    bn = BN_Folder()  #Fold the BN layer 
    net_bn = bn.fold(net.eval())
    # validate(args, net_bn, test_loader, device)
    
    quan_fun = Quantize_Network(w_alpha = 3,dynamic_alpha = False) # weight_quantization
    net_quan = quan_fun.quantize(net_bn)
    # validate(args, net_bn, test_loader, device)
    # print(net_quan.attn.attn_lif.v_threshold)

    # CRI only spikes when the input > threshold
    # add offset to ensure the postneuron will spike 
    threshold_offset = 10
    
    cri_convert = CRI_Converter(args.num_steps, # num_steps
                                0, # input_layer
                                8, # output_layer
                                (1, 28, 28), # input_size
                                'spikingjelly', # backend
                                int(quan_fun.v_threshold) + threshold_offset , # used for the weight of the synapses
                                N) # embed_dim
    
    print(net_quan)
    cri_convert._attention_converter(net_quan)

    cri_convert._cri_fanout_test()
    
    
    config= {}
    config['neuron_type'] = "ANN" #memoryless neurons
    config['global_neuron_params'] = {}
    config['global_neuron_params']['v_thr'] = int(quan_fun.v_threshold)

    
    hardwareNetwork, softwareNetwork = None, None
    if args.hardware:
        hardwareNetwork = CRI_network(dict(cri_convert.mul_axon),
                                      connections=dict(cri_convert.mul_neuron),
                                      config=config,target='CRI', 
                                      outputs =cri_convert.mul_output,
                                      coreID=1, 
                                      perturbMag=8,#Zero randomness  
                                      leak=2**6)#IF
    else:
        # breakpoint()
        softwareNetwork = CRI_network(dict(cri_convert.mul_axon),
                                      connections=dict(cri_convert.mul_neuron),
                                      config=config,target='simpleSim', 
                                      outputs = cri_convert.mul_output,
                                      coreID=1, 
                                      perturbMag=8, #Zero randomness  
                                      leak=2**6)#IF

    cri_convert.bias_start_idx = 0 #add this to the end of conversion
    loss_fun = nn.MSELoss()
    start_time = time.time()
    test_loss = 0
    test_acc = 0
    test_loss_cri = 0
    test_acc_cri = 0
    test_samples = 0
    num_batches = 0
    encoder = encoding.PoissonEncoder()
    
    for img, label in tqdm(test_loader):
        img = img.to(device) #one batch
        label = label.to(device)
        label_onehot = F.one_hot(label, 10).float()
        out_fr = 0.
        out_cri = 0.
        for t in range(args.num_steps):
            encoded_img = encoder(img)
            out_fr += net(encoded_img)
            
            
            q,k,v = net_test.forward_qkv(encoded_img)
            # breakpoint()
            cri_input = cri_convert.input_converter_mul(q,k,v)
            
            if args.hardware:
                cri_output = cri_convert.run_CRI_hw_testing(cri_input,softwareNetwork)
            else:
                cri_output = cri_convert.run_CRI_sw_testing(cri_input,softwareNetwork)
            
            spiking_mul = net_mul.forward_mul(encoded_img)
            
            #reconstruct the output matrix from spike idices
            outputs = np.zeros(spiking_mul.shape)
            for b, output_spikes in enumerate(cri_output):
                for spike_idx in output_spikes:
                    i = spike_idx//(outputs.shape[-1])
                    j = spike_idx%(outputs.shape[-1])
                    outputs[b,:,i,j] = 1
        
            outputs = torch.tensor(outputs)
            
            #compare the multiplication outputs from cri with spkingjelly
            correct = (outputs == spiking_mul).float().sum().item()
            
            #feed the mul output from cri into the net
            # breakpoint()
            out_cri += net_test.forward_output(outputs.float().to(device))
            
        functional.reset_net(net)
        functional.reset_net(net_test)
        functional.reset_net(net_mul)
        
        out_fr = out_fr/args.num_steps
        out_cri = out_cri/args.num_steps
        
        loss = loss_fun(out_fr, label_onehot)
        test_samples += label.numel()
        test_loss += loss.item() * label.numel()
        test_acc += (out_fr.argmax(1) == label).float().sum().item()
        
        loss_cri = loss_fun(out_cri, label_onehot)
        test_loss_cri += loss_cri.item() * label.numel()
        test_acc_cri += (out_cri.argmax(1) == label).float().sum().item()
        
        print(f'test_loss ={test_loss/test_samples: .4f}, test_acc ={test_acc/test_samples: .4f}')
        print(f'test_loss_cri ={test_loss_cri/test_samples: .4f}, test_acc_cri ={test_acc_cri/test_samples: .4f}')
        breakpoint()
        
    test_loss /= test_samples
    test_acc /= test_samples
    test_loss_cri /= test_samples
    test_acc_cri /= test_samples
    
    print(f'test_loss_cri ={test_loss_cri: .4f}, test_acc_cri ={test_acc_cri: .4f}')

    
if __name__ == '__main__':
    main()