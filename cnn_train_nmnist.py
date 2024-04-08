import argparse
import torch
from torch.utils.data import DataLoader
from torch.cuda import amp
from spikingjelly.datasets.n_mnist import NMNIST
from spikingjelly.activation_based import surrogate, neuron
from models import NMNIST_SD
from utils import train_DVS

parser = argparse.ArgumentParser()
parser.add_argument('-resume_path', default='', type=str, help='checkpoint file')
parser.add_argument('-load_path', default='', type=str, help='checkpoint loading path')
parser.add_argument('-load_ssa_path', default='', type=str, help='ssa checkpoint loading path')
parser.add_argument('-train', action='store_true', default=False, help='Train the network from stratch')
parser.add_argument('-b', default=32, type=int, help='batch size')
parser.add_argument('-data-dir', default='/Volumes/export/isn/keli/code/data/NMNIST', type=str, help='path to dataset')
parser.add_argument('-out-dir', default='/Volumes/export/isn/keli/code/HS/CRI_Mapping/output/nmnist', type=str, help='dir path that stores the trained model checkpoint')
parser.add_argument('-epochs', default=20, type=int)
parser.add_argument('-lr', default=1e-3, type=float)
parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('-weight_decay', default=0.01, type=float, help='weight decay for Adam')
parser.add_argument('-T', default=16, type=int)
parser.add_argument('-channels', default=20, type=int)
parser.add_argument('-writer', action='store_true', default=False, help='Use torch summary')
parser.add_argument('-encoder',action='store_true',default=True, help='Using spike rate encoder to process the input')
parser.add_argument('-amp', action='store_true', default=True, help='Use mixed percision training')
parser.add_argument('-num_batches', default=4, type=int)
parser.add_argument('-transformer', action='store_true', default=False, help='Training transformer model')
parser.add_argument('-j', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
parser.add_argument('-opt', default="adam", type=str, help='use which optimizer. SDG or Adam')
parser.add_argument('-dvs', action='store_true', default=True, help='Using the DVS datasets')
parser.add_argument('-targets', default=10, type=int, help='target label size')

def main():
    
    # Train
    # python cnn_train.py -data-dir /Users/keli/Code/CRI/data/DVS128Gesture -out-dir /Users/keli/Code/CRI/CRI_Mapping/runs/dvs_gesture
    
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    
    scaler = amp.GradScaler()
        
    #Prepare the dataset
    # DVS128
    train_set = NMNIST(root=args.data_dir, train=True, data_type='frame', frames_number=args.T, split_by='number')
    test_set = NMNIST(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_set, batch_size=args.b, shuffle=True, drop_last=True, pin_memory = True
    )
    test_loader = DataLoader(
        test_set, batch_size=args.b, shuffle=True, drop_last=True, pin_memory = True
    )
    
    # Initialize SnnTorch/SpikingJelly model
    net = NMNIST_SD(spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    
    net.to(device)
    
    # functional.set_step_mode(net, 'm')
    
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
    
    print('Start Training')
    train_DVS(args, net, train_loader, test_loader, device, scaler)
    # train_DVS_Mul(args, net, train_loader, test_loader, device, scaler)    
        
if __name__ == '__main__':
    main()