import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from hs_api.converter import CRI_Converter, Quantize_Network, BN_Folder
from models import DVSGestureNet
from spikingjelly.activation_based import neuron, surrogate, encoding
from utils import validate_DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.utils.data import DataLoader
from hs_api.api import CRI_network
import time
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('-s', default=1, type=int, help='stride size')
parser.add_argument('-k', default=3, type=int, help='kernel size')
parser.add_argument('-p', default=0, type=int, help='padding size')
parser.add_argument('-c', default=4, type=int, help='channel size')
parser.add_argument('-data-dir', default='/Users/keli/code/CRI/data', type=str, help='path to dataset')
parser.add_argument('-alpha',  default=4, type=int, help='Range of value for quantization')
parser.add_argument('-b', default=32, type=int, help='batch size')
parser.add_argument('-data-dir', default='/Volumes/export/isn/keli/code/data/DVS128Gesture', type=str, help='path to dataset')

def main():
    
    args = parser.parse_args()
    print(args)
    
    #Prepare the dataset
    # DVS128
    train_set = DVS128Gesture(root=args.data_dir, train=True, data_type='frame', frames_number=args.T, split_by='number')
    test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')
    
    # Create DataLoaders
    test_loader = DataLoader(
        test_set, batch_size=args.b, shuffle=True, drop_last=True, pin_memory = True
    )
    
    net = DVSGestureNet(channels=20, spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    
    device = torch.device("cpu")
    
    checkpoint = torch.load(args.resume_path, map_location=device)
    net.load_state_dict(checkpoint['net'])
    
    net.eval()
    
    bn = BN_Folder()
    net_bn = bn.fold(net)
    
    qn = Quantize_Network(w_alpha=args.alpha)
    net_quan = qn.quantize(net_bn)
    
    #Set the parameters for conversion
    input_layer = 0 #first pytorch layer that acts as synapses, indexing begins at 0 
    output_layer = 14 #last pytorch layer that acts as synapses
    input_shape = (2, 128, 128)
    v_threshold = qn.v_threshold

    cn = CRI_Converter(num_steps = 4,
                    input_layer = input_layer, 
                    output_layer = output_layer, 
                    input_shape = input_shape,
                    v_threshold = v_threshold,
                    embed_dim=0)
    
    cn.layer_converter(net_quan)
    
    breakpoint()
    
    config = {}
    config['neuron_type'] = "I&F"
    config['global_neuron_params'] = {}
    config['global_neuron_params']['v_thr'] = int(qn.v_threshold)
    
    
    hardwareNetwork = CRI_network(dict(cn.axon_dict),
                    connections=dict(cn.neuron_dict),
                    config=config,target='CRI', 
                    outputs = cn.output_neurons,
                    simDump=False,
                    coreID=1)

    start_time = time.time()
    
    test_loss = 0
    test_acc = 0
    test_samples = 0
    
    writer = SummaryWriter(log_dir='./log_hardware')
    encoder = encoding.PoissonEncoder()
    
    loss_fun = nn.MSELoss()

    for img, label in test_loader:
        img = img.transpose(0, 1) # [B, T, C, H, W] -> [T, B, C, H, W]
        label_onehot = F.one_hot(label, 11).float()
        out_fr = 0.
        
        cri_input = []
        
        for t in img:
            encoded_img = encoder(t)
            cri_input.append(encoded_img)
        
        cri_input = torch.stack(cri_input)
        cri_input = cn.input_converter(cri_input)
        
        out_fr = torch.tensor(cn.run_CRI_hw(cri_input,hardwareNetwork), dtype=float).to(device)    
            
        loss = loss_fun(out_fr, label_onehot)
        test_samples += label.numel()
        test_loss += loss.item() * label.numel()

        test_acc += (out_fr.argmax(1) == label).float().sum().item()      
    
    test_time = time.time()
    test_speed = test_samples / (test_time - start_time)
    test_loss /= test_samples
    test_acc /= test_samples
    
    writer.add_scalar('test_loss', test_loss)
    writer.add_scalar('test_acc', test_acc)            
    
    print(f'test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}')
    print(f'test speed ={test_speed: .4f} images/s')

    
if __name__ == '__main__':
    main()
    