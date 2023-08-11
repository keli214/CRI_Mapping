import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
import argparse
from torch.utils.data import DataLoader
from spikingjelly.activation_based import encoding, functional
import time
import os
import datetime
# import hs_bridge
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.activation_based.neuron import IFNode, LIFNode

def isSNNLayer(layer):
    # Test = isinstance(layer, MultiStepLIFNode) or isinstance(layer, LIFNode) or isinstance(layer, IFNode)
    # print(layer, Test)
    return isinstance(layer, MultiStepLIFNode) or isinstance(layer, LIFNode) or isinstance(layer, IFNode)


""" Given a net and train_loader, this helper function trains the network for the given epochs 
    It can also resume from checkpoint 

Args:
    args: command line arguments
    net: the network to be trained
    train_loader: pytorch train DataLoader object
    test_loader: pytorch test DataLoader object
    device: cpu or cuda
    scaler: used for amp mixed percision training

"""
def train(args, net, train_loader, test_loader, device, scaler):  
    start_epoch = 0
    max_test_acc = -1
   
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    # optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    loss_fun = nn.MSELoss()
    #loss_fun = nn.CrossEntropyLoss()
    
    encoder, writer = None, None
    if args.encoder:
        encoder = encoding.PoissonEncoder()
        # encoder = encoding.LatencyEncoder(args.num_steps)
            
    if args.resume_path != "":
        checkpoint = torch.load(args.resume_path, map_location=device)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        max_test_acc = checkpoint['max_test_acc']
    
    if args.writer:
        writer = SummaryWriter(args.out_dir)
    
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_loader:
            optimizer.zero_grad()
            img = img.to(device)
            label = label.to(device)
            label_onehot = F.one_hot(label, 10).float()
            out_fr = 0.
            if args.encoder:
                if args.amp:
                    with amp.autocast():
                        for t in range(args.num_steps):
                            encoded_img = encoder(img)
                            out_fr += net(encoded_img)
                        out_fr = out_fr/args.num_steps   
                        loss = loss_fun(out_fr, label_onehot)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        
                else:
                    for t in range(args.num_steps):
                        encoded_img = encoder(img)
                        out_fr += net(encoded_img)
                    out_fr = out_fr/args.num_steps  
                    loss = loss_fun(out_fr, label_onehot)
                    loss.backward()
                    optimizer.step()
            else:
                out_fr = net(img)
                loss = loss_fun(out_fr, label_onehot)
                if args.amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            
            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples
        
        if args.writer:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0

        with torch.no_grad():
            for img, label in test_loader:
                img = img.to(device)
                label = label.to(device)
                label_onehot = F.one_hot(label, 10).float()
                out_fr = 0.
                
                if args.encoder:
                    for t in range(args.num_steps):
                        encoded_img = encoder(img)
                        out_fr += net(encoded_img)
                    out_fr = out_fr/args.num_steps   
                else:
                    out_fr = net(img)
                    
                loss = loss_fun(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
            
            test_time = time.time()
            test_speed = test_samples / (test_time - train_time)
            test_loss /= test_samples
            test_acc /= test_samples
            if args.writer:
                writer.add_scalar('test_loss', test_loss, epoch)
                writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(args.out_dir, f'checkpoint_max_{args.num_steps}.pth'))

        torch.save(checkpoint, os.path.join(args.out_dir, f'checkpoint_latest_{args.num_steps}.pth'))

        print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')


def validate(args, net, test_loader, device):
    start_time = time.time()
    net.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    
    writer, encoder = None, None
    if args.writer:
        writer = SummaryWriter(args.out_dir)
    encoder = None
    if args.encoder:
        encoder = encoding.PoissonEncoder()
    
    loss_fun = nn.MSELoss()
    #loss_fun = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(device)
            label = label.to(device)
            label_onehot = F.one_hot(label, 10).float()
            out_fr = 0.
            if args.encoder:
                for t in range(args.num_steps):
                    encoded_img = encoder(img)
                    out_fr += net(encoded_img)
                out_fr = out_fr/args.num_steps
            else:
                out_fr = net(img)
                
            loss = loss_fun(out_fr, label_onehot)
            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (out_fr.argmax(1) == label).float().sum().item()
            functional.reset_net(net) #reset the membrane potential after each img
        test_time = time.time()
        test_speed = test_samples / (test_time - start_time)
        test_loss /= test_samples
        test_acc /= test_samples
        
        if args.writer:
            writer.add_scalar('test_loss', test_loss)
            writer.add_scalar('test_acc', test_acc)
    
    print(f'test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}')
    print(f'test speed ={test_speed: .4f} images/s')

    
def run_CRI_hw(inputList,hardwareNetwork,cri_convert):
    predictions = []
    #each image
    total_time_cri = 0
    for currInput in inputList:
        #initiate the hardware for each image
        cri_simulations.FPGA_Execution.fpga_controller.clear(len(cri_convert.neuron_dict), False, 0)  ##Num_neurons, simDump, coreOverride
        spikeRate = [0]*10
        #each time step
        for slice in currInput:
            start_time = time.time()
            hwSpike = hardwareNetwork.step(slice, membranePotential=False)
            # print("Mem:",mem)
            end_time = time.time()
            total_time_cri = total_time_cri + end_time-start_time
            print(hwSpike)
            for spike in hwSpike:
                print(int(spike))
                spikeIdx = int(spike) - self.bias_start_idx 
                if spikeIdx >= 0: 
                    spikeRate[spikeIdx] += 1 
        predictions.append(spikeRate.index(max(spikeRate))) 
    print(f"Total execution time CRIFPGA: {total_time_cri:.5f} s")
    return(predictions)
    
# def run_CRI_sw(inputList,softwareNetwork):
#     predictions = []
#     total_time_cri = 0
#     #each image
#     for currInput in inputList:
#         #reset the membrane potential to zero
#         softwareNetwork.simpleSim.initialize_sim_vars(len(neuronsDict))
#         spikeRate = [0]*10
#         #each time step
#         for slice in currInput:
#             start_time = time.time()
#             swSpike = softwareNetwork.step(slice, membranePotential=False)
#             end_time = time.time()
#             total_time_cri = total_time_cri + end_time-start_time
#             for spike in swSpike:
#                 spikeIdx = int(spike) - self.bias_start_idx 
#                 try: 
#                     if spikeIdx >= 0: 
#                         spikeRate[spikeIdx] += 1 
#                 except:
#                     print("SpikeIdx: ", spikeIdx,"\n SpikeRate:",spikeRate )
#         predictions.append(spikeRate.index(max(spikeRate)))
#     print(f"Total simulation execution time: {total_time_cri:.5f} s")
#     return(predictions)
        
# def run(self, network, test_loader, loss_fun, backend='hardware'):
#     self.bias_start_idx = int(self.output_neurons[0])#add this to the end of conversion
#     start_time = time.time()
#     test_loss = 0
#     test_acc = 0
#     test_samples = 0
#     for img, label in test_loader:
#         cri_input = self.input_converter(img)
#         # print(f'cri_input: {cri_input}')
#         # img = img.to(device)
#         # label = label.to(device)

#         label_onehot = F.one_hot(label, 10).float()
#         # out_fr = net(img)

#         if backend == 'hardware':
#             output = torch.tensor(self.run_CRI_hw(cri_input,network))

#         elif backend == 'software':
#             output = torch.tensor(self.run_CRI_sw(cri_input,network))

#         else:
#             print(f'Not supported {backend}')
#             return 

#         loss = loss_fun(output, label_onehot)

#         test_samples += label.numel()
#         test_loss += loss.item() * label.numel()
#         test_acc += (output.argmax(1) == label).float().sum().item()

#     test_time = time.time()
#     test_speed = test_samples / (test_time - start_time)
#     test_loss /= test_samples
#     test_acc /= test_samples

#     print(f'test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}')
#     print(f'test speed ={test_speed: .4f} images/s')