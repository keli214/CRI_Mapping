import argparse
import torch
from torch.utils.data import DataLoader
from torch.cuda import amp
from torchvision import datasets, transforms
from spikingjelly.activation_based import surrogate, neuron
from utils import train, validate
from hs_api.converter import CRI_Converter, Quantize_Network, BN_Folder
from hs_api.api import CRI_network
from models import Mnist

parser = argparse.ArgumentParser()
parser.add_argument('-resume_path', default='', type=str, help='checkpoint file')
parser.add_argument('-train', action='store_true', default=False, help='Train the network from stratch')
parser.add_argument('-b', default=32, type=int, help='batch size')
parser.add_argument('-data-dir', default='/Volumes/export/isn/keli/code/data', type=str, help='path to dataset')
parser.add_argument('-out-dir', default='/Volumes/export/isn/keli/code/HS/CRI_Mapping/runs/mnist', type=str, help='dir path that stores the trained model checkpoint')
parser.add_argument('-epochs', default=5, type=int)
parser.add_argument('-lr', default=1e-3, type=float)
parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('-T', default=4, type=int)
parser.add_argument('-features', default=1000, type=int)
parser.add_argument('-writer', action='store_true', default=False, help='Use torch summary')
parser.add_argument('-encoder',action='store_true',default=True, help='Using spike rate encoder to process the input')
parser.add_argument('-amp', action='store_true', default=False, help='Use mixed percision training')
parser.add_argument('-opt', default="adam", type=str, help='use which optimizer. SDG or Adam')
parser.add_argument('-transformer', action='store_true', default=False, help='Training transformer model')
parser.add_argument('-convert', action='store_true', default=True, help='Convert the network for CRI')
parser.add_argument('-test', action='store_true', help='Test the PyTorch network')
parser.add_argument('-dvs', action='store_true', default=False, help='Using the DVS datasets')
parser.add_argument('-quant', action='store_true', help='Test the quantized network ')
parser.add_argument('-alpha',  default=4, type=int, help='Range of value for quantization')
parser.add_argument('-cri',  action='store_true', default=True, help='Test the converted network')
parser.add_argument('-save',  action='store_true', help='Save converted network')
parser.add_argument('-hardware',action='store_true', default=False, help='Run the network on FPGA')

def main():
    
    # Train
    # python test_cnn.py -data-dir /Users/keli/Desktop/CRI/data -out-dir /Users/keli/Desktop/CRI/CRI_Mapping/runs/cnn
    
    # Verify on Hardware
    # python test_cnn.py -data-dir /Users/keli/Desktop/CRI/data -out-dir /Users/keli/Desktop/CRI/CRI_Mapping/runs/cnn -resume_path runs/cnn/checkpoint_max_T_4_lr_0.001.pth -hardware
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    
    scaler = None
    if args.amp:
        scaler = amp.GradScaler()
        
    #Prepare the dataset
    train_set = datasets.MNIST(args.data_dir, train=True, download=True, transform=transforms.Compose(
        [transforms.ToTensor()]))
    test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=transforms.Compose(
        [transforms.ToTensor()]))
     
    # Create DataLoaders
    train_loader = DataLoader(
        train_set, batch_size=args.b, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=args.b, shuffle=True, drop_last=True
    )
    
    # Initialize SnnTorch/SpikingJelly model
    net = Mnist(features=args.features, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    
    net.to(device)
    
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
    
    if args.train:
        print('Start Training')
        train(args, net, train_loader, test_loader, device, scaler)
    
    if args.convert:
        
        if args.resume_path != "":
            checkpoint = torch.load(args.resume_path, map_location=device)
            net.load_state_dict(checkpoint['net'])
            
        net.eval()
        
        #Fold the BN layer 
        bn = BN_Folder() 
        net_bn = bn.fold(net)
            
        #Weight, Bias Quantization 
        qn = Quantize_Network(w_alpha=args.alpha) 
        net_quan = qn.quantize(net_bn)
        
        #Set the parameters for conversion
        input_layer = 0 #first pytorch layer that acts as synapses, indexing begins at 0 
        output_layer = 5 #last pytorch layer that acts as synapses
        input_shape = (1, 28, 28)
        backend = 'spikingjelly'
        v_threshold = qn.v_threshold

        cn = CRI_Converter(num_steps = args.T, 
                        input_layer = input_layer, 
                        output_layer = output_layer, 
                        input_shape = input_shape,
                        backend=backend,
                        v_threshold = v_threshold,
                        embed_dim=0,
                        dvs = args.dvs)
        
        cn.layer_converter(net_quan)
        
        breakpoint()
        
        config = {}
        config['neuron_type'] = "I&F"
        config['global_neuron_params'] = {}
        config['global_neuron_params']['v_thr'] = int(qn.v_threshold)
        
        #TODO: Get the number during conversion
        cn.bias_start_idx = int(1*28*28)
    
        if args.hardware:
            hardwareNetwork = CRI_network(dict(cn.axon_dict),
                            connections=dict(cn.neuron_dict),
                            config=config,target='CRI', 
                            outputs = cn.output_neurons,
                            simDump=False,
                            coreID=1)

            validate(args, hardwareNetwork, test_loader, device, cn=cn)
            
        else:    
            softwareNetwork = CRI_network(dict(cn.axon_dict),
                            connections=dict(cn.neuron_dict),
                            config=config,target='simpleSim', 
                            outputs = cn.output_neurons,
                            coreID=1)
            validate(args, softwareNetwork, test_loader, device, cn=cn)
        
        
if __name__ == '__main__':
    main()