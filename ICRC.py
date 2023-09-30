import numpy as np
from quant.quant_layer import *
import torch 
import argparse
from hs_api.api import CRI_network

# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import time
from hs_api.api import CRI_network
import hs_bridge
from tqdm import tqdm


# import hs_bridge

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='.', type=str, help='path to dataset')
parser.add_argument('-b','--batch_size', default=1, type=int)

net_Wc = torch.tensor(np.loadtxt("net_Wc.csv", delimiter=",", dtype=float))
net_Wh = torch.tensor(np.loadtxt("net_Wh.csv", delimiter=",", dtype=float))
net_b_vch = torch.tensor(np.loadtxt("net_b_vch.csv", delimiter=",", dtype=float))


def main():
    args = parser.parse_args()
    #TODO: quantitize the weight and bias
    w_alpha=1
    w_bits=16
    threshold = 1.0
    weight_quant = weight_quantize_fn(w_bit= w_bits, wgt_alpha = w_alpha)
    w_delta = w_alpha/(2**(w_bits-1)-1)

    quant_net_Wc = (weight_quant(net_Wc)/w_delta).reshape(40,500)
    quant_net_Wh = (weight_quant(net_Wh)/w_delta).reshape(784,500)
    quant_net_b_vch = weight_quant(net_b_vch)/w_delta
    quant_threshold = threshold/w_delta


    Nv = 784
    Nc = 40
    qunat_b_v = quant_net_b_vch[:Nv]
    qunat_b_c = quant_net_b_vch[Nv:Nv+Nc]
    qunat_b_h = quant_net_b_vch[Nv+Nc:]

    axon_dict = {}
    neuron_dict = {}
    output_list = []

    breakpoint()

    axon_offset = 0

    for axon in range(quant_net_Wh.shape[0]):
        axon_dict['a'+str(axon)] = [(str(neuron), int(w)) for neuron, w in enumerate(quant_net_Wh[axon])]
        
    for neuron in range(quant_net_Wh.transpose(0,1).shape[0]):
        neuron_dict[str(neuron)] = [('a'+str(axon), int(w)) for axon, w in enumerate(quant_net_Wh.transpose(0,1)[neuron])]   
        
    axon_offset = len(axon_dict)
        
    for axon in range(quant_net_Wc.shape[0]):
        axon_dict['a'+str(axon+axon_offset)] = [(str(neuron), int(w)) for neuron, w in enumerate(quant_net_Wc[axon])]
        output_list.append('a'+str(axon+axon_offset))
        
    for neuron in range(quant_net_Wc.transpose(0,1).shape[0]):
        neuron_dict[str(neuron)].extend([('a'+str(axon+axon_offset), int(w)) for axon, w in enumerate(quant_net_Wc.transpose(0,1)[neuron])])
        
    axon_offset = len(axon_dict)

    #TODO: bias axons
    for axon in range(qunat_b_v.shape[0]):
        axon_dict['a'+str(axon+axon_offset)] = [(str(idx),int(qunat_b_v[axon])) for idx in range(len(neuron_dict)) ]

    axon_offset = len(axon_dict)

    for axon in range(qunat_b_c.shape[0]):
        axon_dict['a'+str(axon+axon_offset)] = [(str(idx),int(qunat_b_c[axon])) for idx in range(len(neuron_dict)) ]
        
    axon_offset = len(axon_dict)

    for axon in range(qunat_b_h.shape[0]):
        axon_dict['a'+str(axon+axon_offset)] = [(str(idx),int(qunat_b_h[axon])) for idx in range(len(neuron_dict)) ]
        
        
    #TODO: figure out the parameters (threshold, perturbMag, leak for the network
    config= {}
    config['neuron_type'] = "ANN" #memoryless neurons
    config['global_neuron_params'] = {}
    config['global_neuron_params']['v_thr'] = int(quant_threshold)


    hardwareNetwork = CRI_network(axon_dict,
                                connections=neuron_dict,
                                config=config,
                                target='CRI', 
                                outputs = output_list,
                                coreID=1, 
                                perturbMag=16,#Highest randomness  
                                leak=2**6)#IF

    #Prepare the dataset
    mnist_train = datasets.MNIST(args.data_path, train=True, download=True, transform=transforms.Compose(
        [transforms.Resize((7,7)),transforms.ToTensor()]))
    mnist_test = datasets.MNIST(args.data_path, train=False, download=True, transform=transforms.Compose(
        [transforms.Resize((7,7)),transforms.ToTensor()]))
        
    # Create DataLoaders
    train_loader = DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_correct = 0
    test_samples = 0
    for batch, label in tqdm(test_loader):
        predictions = []
        cri_inputs = [['a'+str(idx) for idx, pixel in enumerate(img.flatten()) if pixel > 0.5] for img in batch]
        for currInput in cri_inputs:
            # initiate the hardware for each image
            hs_bridge.FPGA_Execution.fpga_controller.clear(
                len(output_list), False, 0
            )  ##Num_neurons, simDump, coreOverride
            spikeRate = [0] * 10
            hwSpike, latency, hbmAcc = hardwareNetwork.step(currInput, membranePotential=False)
            spikeIdx = [int(spike)-784 for spike in hwSpike]
            for idx in spikeIdx:
                spikeRate[idx%10] += 1
            predictions.append(spikeRate.index(max(spikeRate)))
        
        predictions = torch.tensor(predictions)
        test_correct += (predictions == label).float().sum().item()
        test_samples += label.numel()
        
    test_acc = test_correct/test_samples
    print(test_acc)  
        
        
        
main()
                            