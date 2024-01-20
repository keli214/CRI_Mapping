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
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.activation_based.neuron import IFNode, LIFNode
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import profiler
import numpy as np

def isSNNLayer(layer):
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
    
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    loss_fun = nn.MSELoss()
    #loss_fun = nn.CrossEntropyLoss()
    
    encoder, writer = None, None
    if args.encoder:
        encoder = encoding.PoissonEncoder()
        # encoder = encoding.LatencyEncoder(args.T)
            
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
                        if args.transformer:
                            encoded_img = encoder(img)
                            out_fr += net(encoded_img)
                        if args.dvs:
                            # [N, T, C, H, W] -> [T, N, C, H, W]
                            img = img.transpose(0, 1) 
                            for t in range(args.T):
                                encoded_img = encoder(img[t])
                                out_fr += net(encoded_img)
                        else:
                            for t in range(args.T):
                                encoded_img = encoder(img)
                                out_fr += net(encoded_img)
                else:
                    if args.transformer:
                        encoded_img = encoder(img)
                        out_fr += net(encoded_img)
                    if args.dvs:
                        # [N, T, C, H, W] -> [T, N, C, H, W]
                        img = img.transpose(0, 1) 
                        for t in range(args.T):
                            encoded_img = encoder(img[t])
                            out_fr += net(encoded_img)
                    else:
                        for t in range(args.T):
                            encoded_img = encoder(img)
                            out_fr += net(encoded_img)

            else:
                if args.transformer:
                    out_fr += net(img)
                if args.dvs:
                    # [N, T, C, H, W] -> [T, N, C, H, W]
                    img = img.transpose(0, 1)
                    for t in range(args.T):
                        out_fr += net(img[t])
                else:
                    for t in range(args.T):
                        out_fr += net(img)
            
            out_fr = out_fr/args.T   
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
                    if args.transformer:
                        encoded_img = encoder(img)
                        out_fr += net(encoded_img)
                    if args.dvs:
                        img = img.transpose(0, 1) 
                        for t in range(args.T):
                            encoded_img = encoder(img[t])
                            out_fr += net(encoded_img)
                    else:
                        for t in range(args.T):
                            encoded_img = encoder(img)
                            out_fr += net(encoded_img)  
                else:
                    if args.dvs:
                        img = img.transpose(0, 1) 
                        for t in range(args.T):
                            out_fr += net(img[t])
                    else:
                        for t in range(args.T):
                            out_fr += net(img)
                
                out_fr = out_fr/args.T 
                    
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
            torch.save(checkpoint, os.path.join(args.out_dir, f'checkpoint_max_T_{args.T}_C_{args.channels}_lr_{args.lr}.pth'))
            if args.transformer:
                checkpoint_ssa = {'ssa': net.block[0].attn.state_dict()}
                torch.save(checkpoint_ssa, os.path.join(args.out_dir, f'checkpoint_max_ssa_T_{args.T}_C_{args.channels}_lr_{args.lr}.pth'))

        torch.save(checkpoint, os.path.join(args.out_dir, f'checkpoint_latest_T_{args.T}_C_{args.channels}_lr_{args.lr}.pth'))

        print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')


def validate(args, net, test_loader, device, cn=None):
    start_time = time.time()
    
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
    
    if args.cri:
        # dvs: [B, T, C, H, W] regualr img: [B, C, H, W]
        for img, label in test_loader:
            label_onehot = F.one_hot(label, 10).float()
            out_fr = 0.
            
            cri_input = None
            
            if args.dvs:
                if args.encoder:
                    encoded_img = encoder(img)
                    cri_input = cn.input_converter(encoded_img)
                else:
                    cri_input = cn.input_converter(img)
            else:
                if args.encoder: 
                    img_repeats = img.repeat(args.T, 1, 1, 1, 1)
                    cri_input = []
                    for t in range(args.T): 
                        encoded_img = encoder(img_repeats[t])
                        cri_input.append(encoded_img)
                    cri_input = cn.input_converter(np.array(cri_input).transpose(1,0,2,3,4))
                else:
                    cri_input = cn.input_converter(img.repeat(args.T, 1, 1, 1, 1))
            
            if args.hardware:
                out_fr = torch.tensor(cn.run_CRI_hw(cri_input,net), dtype=float).to(device)
            else:
                out_fr = torch.tensor(cn.run_CRI_sw(cri_input,net), dtype=float).to(device)
                
            loss = loss_fun(out_fr, label_onehot)
            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            
            test_acc += (out_fr.argmax(1) == label).float().sum().item()      
        
        test_time = time.time()
        test_speed = test_samples / (test_time - start_time)
        test_loss /= test_samples
        test_acc /= test_samples
        
        if args.writer:
            writer.add_scalar('test_loss', test_loss)
            writer.add_scalar('test_acc', test_acc)            
                    
    
    else:
        
        net.eval()
        
        with torch.no_grad():
            for img, label in test_loader:
                img = img.to(device) 
                label = label.to(device)
                label_onehot = F.one_hot(label, 10).float()
                out_fr = 0.
                
                if args.dvs:
                    img = img.transpose(0, 1) 
                    if args.encoder:
                        for t in range(args.T):
                            encoded_img = encoder(img[t])
                            out_fr += net(encoded_img)
                    else:
                        for t in range(args.T):
                            out_fr += net(img[t])
                else:
                    if args.encoder:
                        encoded_img = encoder(img)
                        out_fr += net(img)
                    else:
                        out_fr += net(img)
                
                out_fr = out_fr/args.T

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
