import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.cuda import amp
from models import QuantMnist
from utils import train, validate

parser = argparse.ArgumentParser()
parser.add_argument('--resume_path', default='', type=str, help='checkpoint file')
parser.add_argument('--load_path', default='', type=str, help='checkpoint loading path')
parser.add_argument('--load_ssa_path', default='', type=str, help='ssa checkpoint loading path')
parser.add_argument('-train', action='store_true', default=False, help='Train the network from stratch')
parser.add_argument('-b','--batch_size', default=32, type=int)
parser.add_argument('--data_path', default='/Volumes/export/isn/keli/code/data', type=str, help='path to dataset')
parser.add_argument('--out_dir', default='/Volumes/export/isn/keli/code/HS/CRI_Mapping/runs/mnist', type=str, help='dir path that stores the trained model checkpoint')
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
parser.add_argument('-transformer', action='store_true', default=False, help='Training transformer model')

def main():
    
    # Train
    # python test_quant.py -train --data_path /Users/keli/Desktop/CRI/data --out_dir /Users/keli/Desktop/CRI/CRI_Mapping/runs/mnist -T 4 --encoder 
    
    args = parser.parse_args()
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    
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
    net = QuantMnist(features=2000)
    
    net.to(device)
    
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
    
    if args.resume_path != "" or args.train:
        print('Start Training')
        train(args, net, train_loader, test_loader, device, scaler)
        return
    
if __name__ == '__main__':
    main()