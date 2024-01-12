import argparse
import torch
from torch.utils.data import DataLoader
from spikingjelly.activation_based.model import parametric_lif_net
from torch.cuda import amp
from spikingjelly.datasets.n_mnist import NMNIST
from spikingjelly.activation_based import functional, surrogate, neuron, layer
from utils import train, validate
from hs_api.converter import CRI_Converter, Quantize_Network

parser = argparse.ArgumentParser()
parser.add_argument('-resume_path', default='', type=str, help='checkpoint file')
parser.add_argument('-load_path', default='', type=str, help='checkpoint loading path')
parser.add_argument('-load_ssa_path', default='', type=str, help='ssa checkpoint loading path')
parser.add_argument('-train', action='store_true', default=False, help='Train the network from stratch')
parser.add_argument('-b', default=32, type=int, help='batch size')
parser.add_argument('-data-dir', default='/Volumes/export/isn/keli/code/data/NMNIST', type=str, help='path to dataset')
parser.add_argument('-out-dir', default='/Volumes/export/isn/keli/code/HS/CRI_Mapping/runs/nmnist', type=str, help='dir path that stores the trained model checkpoint')
parser.add_argument('-epochs', default=10, type=int)
parser.add_argument('-lr', default=1e-1, type=float)
parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('-T', default=16, type=int)
parser.add_argument('-channels', default=128, type=int)
parser.add_argument('-writer', action='store_true', default=False, help='Use torch summary')
parser.add_argument('-encoder',action='store_true',default=False, help='Using spike rate encoder to process the input')
parser.add_argument('-amp', action='store_true', default=False, help='Use mixed percision training')
parser.add_argument('-hardware',action='store_true', default=False, help='Run the network on FPGA')
parser.add_argument('-num_batches', default=4, type=int)
parser.add_argument('-transformer', action='store_true', default=False, help='Training transformer model')
parser.add_argument('-dvs', action='store_true', default=False, help='Training with DVS datasets')
parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
parser.add_argument('-opt', type=str, help='use which optimizer. SDG or Adam')
parser.add_argument('-convert', action='store_true', help='Convert the network for CRI')
parser.add_argument('-test', action='store_true', help='Test the quantized network for CRI')

def main():
    
    # Train
    # python main_encoder.py -train -data-dir /Users/keli/Desktop/CRI/data/NMNIST -out-dir /Users/keli/Desktop/CRI/CRI_Mapping/runs/nmnist -T 16 -channels 102 -encoder -dvs -opt adam -lr 0.001 -j 8
    
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    
    scaler = None
    if args.amp:
        scaler = amp.GradScaler()
        
    #Prepare the dataset
    train_set = NMNIST(root=args.data_dir, train=True, data_type='frame', frames_number=args.T, split_by='number')
    test_set = NMNIST(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')
     
    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )
    
    # Initialize SnnTorch/SpikingJelly model
    net = parametric_lif_net.NMNISTNet(channels=args.channels, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    
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
            
        #Weight, Bias Quantization 
        qn = Quantize_Network(w_alpha=4) 
        net_quan = qn.quantize(net)
        
        #Set the parameters for conversion
        input_layer = 0 #first pytorch layer that acts as synapses, indexing begins at 0 
        output_layer = 13 #last pytorch layer that acts as synapses
        input_shape = (2, 28, 28)
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
        cn.save_model()
    
    if args.test:
        #Weight, Bias Quantization 
        qn = Quantize_Network(w_alpha=4) 
        net_quan = qn.quantize(net)
        net_quan.to(device)
        
        validate(args, net_quan, test_loader, device)
        
if __name__ == '__main__':
    main()