import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torchvision import datasets, transforms
from spikingjelly.activation_based import functional, encoding
from torch.utils.data import DataLoader
import time
import argparse
from cri_converter import CRI_Converter
from quantization import Quantizer
from bn_folder import BN_Folder
from hs_api.api import CRI_network
from utils import train, validate
from tqdm import tqdm
import numpy as np
from spikeformer import Spikeformer, SSA


parser = argparse.ArgumentParser()
parser.add_argument('--resume_path', default='', type=str, help='checkpoint file')
parser.add_argument('--load_path', default='', type=str, help='checkpoint loading path')
parser.add_argument('--load_ssa_path', default='', type=str, help='ssa checkpoint loading path')
parser.add_argument('--train', action='store_true', default=False, help='Train the network from stratch')
parser.add_argument('-b','--batch_size', default=32, type=int)
parser.add_argument('--data_path', default='/Volumes/export/isn/keli/code/data', type=str, help='path to dataset')
parser.add_argument('--out_dir', default='/Volumes/export/isn/keli/code/HS/CRI_Mapping/runs/ssa', type=str, help='dir path that stores the trained model checkpoint')
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--lr', default=1e-1, type=float)
parser.add_argument('-m','--momentum', default=0.9, type=float)
parser.add_argument('-T','--num_steps', default=20, type=int)
parser.add_argument('-c','--channels', default=1, type=int)
parser.add_argument('--writer', action='store_true', default=False, help='Use torch summary')
parser.add_argument('--encoder',action='store_true',default=False, help='Using spike rate encoder to process the input')
parser.add_argument('--amp', action='store_true', default=False, help='Use mixed percision training')
parser.add_argument('--hardware',action='store_true', default=False, help='Run the network on FPGA')
parser.add_argument('--num_batches', default=4, type=int)

def main():
    
    args = parser.parse_args()

    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    
    scaler = None
    if args.amp:
        scaler = amp.GradScaler()
        
    #Prepare the dataset
    mnist_train = datasets.MNIST(args.data_path, train=True, download=True, transform=transforms.Compose(
        [transforms.Resize((16,16)),transforms.ToTensor()]))
    mnist_test = datasets.MNIST(args.data_path, train=False, download=True, transform=transforms.Compose(
        [transforms.Resize((16,16)),transforms.ToTensor()]))
     
    # Create DataLoaders
    train_loader = DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # Initialize SnnTorch/SpikingJelly model
    N = 16
    net = Spikeformer(
        img_size_h=16, img_size_w=16,
        patch_size=4, embed_dims=N, num_heads=1, mlp_ratios=4,
        in_channels=1, num_classes=10, qkv_bias=False,
        depths=1, sr_ratios=1,
        T=1
    )
    
    net_test = Spikeformer(
        img_size_h=16, img_size_w=16,
        patch_size=4, embed_dims=N, num_heads=1, mlp_ratios=4,
        in_channels=1, num_classes=10, qkv_bias=False,
        depths=1, sr_ratios=1,
        T=1
    )
    
    net_mul = Spikeformer(
        img_size_h=16, img_size_w=16,
        patch_size=4, embed_dims=N, num_heads=1, mlp_ratios=4,
        in_channels=1, num_classes=10, qkv_bias=False,
        depths=1, sr_ratios=1,
        T=1
    )
    
    ssa = SSA(dim=N, num_heads=1)
    
    net.to(device)
    net_test.to(device)
    net_mul.to(device)
    ssa.to(device)
    
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
#     # print(net)
    
    if args.resume_path != "" or args.train:
        print('Start Training')
        train(args, net, train_loader, test_loader, device, scaler)
        return
        
    elif args.load_path != "":
        checkpoint = torch.load(args.load_path, map_location=device)
        checkpoint_ssa = torch.load(args.load_ssa_path, map_location=device)
        
        net.load_state_dict(checkpoint['net'])
        
        # Testing for conversion: 
        net_test.load_state_dict(checkpoint['net'])
        net_mul.load_state_dict(checkpoint['net'])
        ssa.load_state_dict(checkpoint_ssa['ssa'])
        
        validate(args, net, test_loader, device)

  
    # Fold the BN layer 
    bn = BN_Folder()  
    ssa_bn = bn.fold(ssa.eval())
    # validate(args, net_bn, test_loader, device)
    
    # weight_quantization
    quan_fun = Quantizer(w_alpha = 3,dynamic_alpha = False) 
    ssa_quan = quan_fun.quantize(ssa_bn)
    # validate(args, net_bn, test_loader, device)
    print(ssa_quan)

    # CRI only spikes when the input > threshold
    # add offset to ensure the postneuron will spike 
    threshold_offset = 1000
    
    cri_convert = CRI_Converter(args.num_steps, # num_steps
                                0, # input_layer
                                8, # output_layer
                                (1, 16, 16), # input_size
                                'spikingjelly', # backend
                                int(quan_fun.v_threshold) + threshold_offset , # used for the weight of the synapses
                                N) # embed_dim
    
    cri_convert._attention_converter(ssa_quan)
    
    breakpoint()
    
    config= {}
    config['neuron_type'] = "L&F" 
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
            
            out_fr += net(img)
            
            q,k,v = net_test.forward_qkv(img)
            embed = net_mul.forward_embed(img)
                
            cri_input = cri_convert._input_converter_mul(q,k,v)
            
            
            if args.hardware:
                first_out, cri_output = cri_convert._run_CRI_hw_ssa_testing(cri_input,hardwareNetwork)
            else:
                first_out, cri_output = cri_convert._run_CRI_sw_ssa_testing(cri_input,softwareNetwork)
            
            
            #reconstruct the output matrix from spike idices
            outputs = np.zeros(embed.shape)
            for b, output_spikes in enumerate(cri_output):
                for spike_idx in output_spikes:
                    i = spike_idx//(outputs.shape[-1])
                    j = spike_idx%(outputs.shape[-1])
                    outputs[b,:,i,j] = 1
            
            outputs = torch.tensor(outputs).float().to(device)
            
            #feed the mul output from cri into the net
            # breakpoint()
            out_cri += net_mul.forward_test(embed, outputs)
            
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
        # breakpoint()
        
    test_loss /= test_samples
    test_acc /= test_samples
    test_loss_cri /= test_samples
    test_acc_cri /= test_samples
    
    print(f'test_loss_cri ={test_loss_cri: .4f}, test_acc_cri ={test_acc_cri: .4f}')

    
if __name__ == '__main__':
    main()