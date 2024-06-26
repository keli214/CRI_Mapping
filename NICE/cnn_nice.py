import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from spikingjelly.activation_based.model import parametric_lif_net
from torch.cuda import amp
from spikingjelly.datasets.n_mnist import NMNIST
from spikingjelly.activation_based import encoding, surrogate, neuron, layer
from hs_api.converter import CRI_Converter, Quantize_Network, BN_Folder
from hs_api.api import CRI_network
from models import NMNISTNet, NMNIST_CNN
import time
import pickle
# import hs_bridge

parser = argparse.ArgumentParser()
parser.add_argument('-resume_path', default='', type=str, help='checkpoint file')
parser.add_argument('-load_path', default='', type=str, help='checkpoint loading path')
parser.add_argument('-b', default=32, type=int, help='batch size')
parser.add_argument('-data-dir', default='/Volumes/export/isn/keli/code/data/NMNIST', type=str, help='path to dataset')
parser.add_argument('-out-dir', default='/Volumes/export/isn/keli/code/HS/CRI_Mapping/runs/nmnist', type=str, help='dir path that stores the trained model checkpoint')
parser.add_argument('-T', default=16, type=int)
parser.add_argument('-channels', default=80, type=int)
parser.add_argument('-num_batches', default=4, type=int)
parser.add_argument('-alpha',  default=4, type=int, help='Range of value for quantization')

def main():
    
    # Verify on Hardware with DVS data
    # python cnn_nice.py -data-dir /Users/keli/code/CRI/data/NMNIST -out-dir /Users/keli/code/CRI/CRI_Mapping/runs/nmnist -resume_path runs/nmnist/checkpoint_max_T_16_C_80_lr_0.001.pth
    
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
        
    #Prepare the test dataset
    test_set = NMNIST(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')
     
    # Create DataLoaders
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=False,
        pin_memory=True
    )
    
    # Initialize SnnTorch/SpikingJelly model
    net = NMNIST_CNN(channels=args.channels, 
                    spiking_neuron=neuron.LIFNode, 
                    surrogate_function=surrogate.ATan(), 
                    detach_reset=True)
    
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
    
    # Resume the network from checkpoint
    checkpoint = torch.load(args.resume_path, map_location=device)
    net.load_state_dict(checkpoint['net'])
    
    # Set the network to evaluation mode    
    net.eval()
        
    # Fold the BN layer 
    bn = BN_Folder() 
    net_bn = bn.fold(net)
            
    # Weight, Bias Quantization 
    qn = Quantize_Network(w_alpha=args.alpha) 
    net_quan = qn.quantize(net_bn)
    print(net_quan)
    
    # Set the network parameter for converter
    input_layer = 10 #first linear layer 
    output_layer = 13 #second linear layer
    input_shape = (2, 34, 34)
    backend = 'spikingjelly'
    v_threshold = qn.v_threshold

    cn = CRI_Converter(num_steps = args.T, 
                    input_layer = input_layer, 
                    output_layer = output_layer, 
                    input_shape = input_shape,
                    backend=backend,
                    v_threshold = v_threshold,
                    embed_dim=0,
                    dvs = True)
    
    # cn.layer_converter(net_quan)
    
    config = {}
    config['neuron_type'] = "I&F"
    config['global_neuron_params'] = {}
    config['global_neuron_params']['v_thr'] = int(qn.v_threshold)
            
    #TODO: Get the number during conversion
    CONV1_OUTPUT_SHAPE = (args.channels, 34, 34)
    CONV2_OUTPUT_SHAPE = (args.channels, 16, 16)
    LINEAR_INPUT_SHAPE = (args.channels, 7, 7)
    L1_OUTPUT_SHAPE = 2048
    OUTPUT_SHAPE = (10)

    # Initialize One CRI network.
    # net_cri = CRI_network(dict(cn.axon_dict),
    #                     connections=dict(cn.neuron_dict),
    #                     config=config,target='CRI', 
    #                     outputs = cn.output_neurons,
    #                     coreID=1)
    net_cri = None
    
    encoder = encoding.PoissonEncoder()
    loss_fun = nn.MSELoss()
    
    start_time = time.time()
    
    test_loss = 0
    test_acc = 0
    test_samples = 0
    
    b_idx = 0
    
    for img, label in test_loader:
        img = img.to(device)
        breakpoint()
        label = label.to(device)
        label_onehot = F.one_hot(label, 10).float()
        out_fr = 0.   
        cri_output = []
        latency_hist = []
        hbmacc_hist = []
        b_idx += 1
        for b in range(args.b):
            spike_rate = np.zeros(OUTPUT_SHAPE)
            encoded_img = encoder(img[b])
            # hs_bridge.FPGA_Execution.fpga_controller.clear(
            #     len(cn.neuron_dict), False, 0
            # )
            for t in range(args.T):
                axon_offset = 0
                neuron_offset = 0
                
                # conv1
                curr_input = ['a' + str(idx) for idx, axon in enumerate(encoded_img[t].flatten()) if axon != 0]
                axon_offset += np.prod(input_shape)
                
                conv1_out, latency, hbmAcc = net_cri.step(curr_input)
                latency_hist.append(latency)
                hbmacc_hist.append(hbmAcc)
                hwSpike, latency, hbmAcc = net_cri.step([])
                latency_hist.append(latency)
                hbmacc_hist.append(hbmAcc)
                conv1_out = conv1_out + hwSpike
                if t == args.T - 1:
                    hwSpike, latency, hbmAcc = net_cri.step([])
                    conv1_out = conv1_out + hwSpike
                    latency_hist.append(latency)
                    hbmacc_hist.append(hbmAcc)
                
                conv1_out_idx = []
                for idx in conv1_out:
                    if int(idx)-neuron_offset < np.prod(CONV1_OUTPUT_SHAPE) and int(idx)-neuron_offset >= 0:
                        conv1_out_idx.append(int(idx)-neuron_offset)
                outputs = np.zeros(CONV1_OUTPUT_SHAPE).flatten()
                outputs[conv1_out_idx] = 1
                neuron_offset += np.prod(CONV1_OUTPUT_SHAPE)
                curr_input = torch.tensor(outputs.reshape(CONV1_OUTPUT_SHAPE)).to(device)
                
                # maxPool1
                curr_input = F.max_pool2d(curr_input, 2).cpu().detach().numpy()
                
                # conv2
                curr_input = np.where(curr_input.flatten() == 1)[0]
                curr_input = ['a' + str(idx + axon_offset) for idx in curr_input]
                axon_offset += np.prod(CONV2_OUTPUT_SHAPE)
                conv2_out, latency, hbmAcc = net_cri.step(curr_input)
                latency_hist.append(latency)
                hbmacc_hist.append(hbmAcc)
                hwSpike, latency, hbmAcc = net_cri.step([])
                latency_hist.append(latency)
                hbmacc_hist.append(hbmAcc)
                conv2_out = conv2_out + hwSpike
                if t == args.T - 1:
                    hwSpike, latency, hbmAcc = net_cri.step([])
                    latency_hist.append(latency)
                    hbmacc_hist.append(hbmAcc)
                    conv2_out = conv2_out + hwSpike
                
                conv2_out_idx = []
                for idx in conv2_out: 
                    if int(idx)-neuron_offset < np.prod(CONV2_OUTPUT_SHAPE) and int(idx)-neuron_offset >= 0:
                        conv2_out_idx.append(int(idx)-neuron_offset )
                neuron_offset += np.prod(CONV2_OUTPUT_SHAPE)
                outputs = np.zeros(CONV2_OUTPUT_SHAPE).flatten()
                outputs[conv2_out_idx] = 1
                curr_input = torch.tensor(outputs.reshape(CONV2_OUTPUT_SHAPE)).to(device)
                
                # maxPool2
                curr_input = F.max_pool2d(curr_input, 2).cpu().detach().numpy()
                
                # linear
                curr_input = np.where(curr_input.flatten() == 1)[0]
                curr_input = ['a' + str(idx + axon_offset) for idx in curr_input]
                axon_offset += np.prod(LINEAR_INPUT_SHAPE)
                bias_input = ['a' + str(idx) for idx in range(axon_offset, len(cn.axon_dict))]  
                
                linear_out, latency, hbmAcc = net_cri.step(curr_input+bias_input)
                latency_hist.append(latency)
                hbmacc_hist.append(hbmAcc)
                hwSpike, latency, hbmAcc = net_cri.step([])
                latency_hist.append(latency)
                hbmacc_hist.append(hbmAcc)
                linear_out = linear_out + hwSpike
                if t == args.T - 1:
                    hwSpike, latency, hbmAcc = net_cri.step([])
                    linear_out = linear_out + hwSpike
                    latency_hist.append(latency)
                    hbmacc_hist.append(hbmAcc)
                
                neuron_offset += np.prod(L1_OUTPUT_SHAPE)
                linear_out_idx = []
                for idx in linear_out:
                    if int(idx)-neuron_offset < 10 and int(idx)-neuron_offset >= 0:
                        linear_out_idx.append(int(idx)-neuron_offset)
                outputs = np.zeros(OUTPUT_SHAPE)
                outputs[linear_out_idx] = 1
                
                # Add spikes from each step
                spike_rate += outputs
            
            # Stack all the img from a batch
            cri_output.append(torch.tensor(spike_rate))
        
        cri_output = torch.stack(cri_output).to(device)
        out_fr = cri_output/args.T
        
        loss = loss_fun(out_fr, label_onehot)
        test_samples += label.numel()
        test_loss += loss.item() * label.numel()
        test_acc += (out_fr.argmax(1) == label).float().sum().item()   
        
        with open(f'history/C_{args.channels}_B_{b_idx}_latency_hist.pkl', 'wb') as f:
            pickle.dump(latency_hist, f) 
        with open(f'history/C_{args.channels}_B_{b_idx}_hbmacc_hist.pkl', 'wb') as f:
            pickle.dump(hbmacc_hist, f) 
        with open(f'history/C_{args.channels}_accuracy.pkl', 'wb') as f:
            pickle.dump(test_acc/test_samples, f) 
        
    test_time = time.time()
    test_speed = test_samples / (test_time - start_time)
    test_loss /= test_samples
    test_acc /= test_samples
    
    print(f'test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}')
    print(f'test speed ={test_speed: .4f} images/s')
    with open(f'history/C_{args.channels}_accuracy.pkl', 'wb') as f:
        pickle.dump(test_acc, f) 
    with open(f'history/C_{args.channels}_loss.pkl', 'wb') as f:
        pickle.dump(test_loss, f) 
    
    
            
            
            
if __name__ == '__main__':
    main()