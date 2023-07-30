# imports
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os
import time
import argparse
from torch.cuda import amp
import sys
import datetime
from spikingjelly import visualizing
import shutil
from quant_layer import *
from cri_converter import BN_Folder, Quantize_Network, CRI_Converter
# from torchsummary import summary
import snntorch as snn
import sparselinear as sl
from l2s.api import CRI_network
import cri_simulations
import pickle

axonNum = 1000
neuronNum = 10000

#startHbm = 0
#startLatency = 0
class CSNN(nn.Module):
	def __init__(self, T: int, beta:int, sparsity):
		super().__init__()
		self.T = T
		self.fc1 = sl.SparseLinear(in_features=axonNum,out_features=neuronNum,sparsity=sparsity)
		self.lif1 = snn.Leaky(beta=beta)
		self.sparsity = sparsity


	def forward(self, x: torch.Tensor):
		# x.shape = [N, C, H, W]
		mem1 = self.lif1.init_leaky()

		# Record the final layer
		spk1_rec = []
		mem1_rec = []
		
		start_time = time.perf_counter()
		for step in range(self.T):
		    cur1 = self.fc1(x)
		    spk1, mem1 = self.lif1(cur1, mem1)
		    spk1_rec.append(spk1)
		    mem1_rec.append(mem1)
		end_time = time.perf_counter()
		duration = end_time - start_time
		return torch.stack(spk1_rec, dim=0), torch.stack(mem1_rec, dim=0), duration


def main(cri, sparsity, startHbm, startLatency, T):
	T = T
	beta = 1
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	net = CSNN(T = T, beta = beta, sparsity = sparsity)
	#print(net.fc1.weight)
	d = net.fc1.weight.to_dense()
	#print(d.shape)
	#print(net.fc1.bias.shape)
	net.to(device)
	
	
	input = torch.ones(axonNum)

	_, _, duration = net(input)
	print(f'Snntorch Duration: {duration}')
	
	latency, hbmAcc = None, None

	if cri:
		cri_convert = CRI_Converter(T, 0, 1, axonNum, 'snntorch', 1)
		cri_convert.layer_converter(net)
		
		config = {}
		config['neuron_type'] = "I&F"
		config['global_neuron_params'] = {}
		config['global_neuron_params']['v_thr'] = 1
		hardwareNetwork = CRI_network(axons=dict(cri_convert.axon_dict),connections=dict(cri_convert.neuron_dict),config=config,target='CRI', outputs = cri_convert.output_neurons,coreID=1)
		
		for t in range(T):
			input = list(cri_convert.axon_dict.keys())
			_, latency, hbmAcc = hardwareNetwork.step(input, membranePotential=False)
			#if t == T-1:
			#	startLatency = latency
			#	startHbm = hbmAcc
			ChbmAcc = hbmAcc-startHbm
			Clatency = latency - startLatency
			if t == T-1:
				startLatency = latency	
				startHbm = hbmAcc
			print("timestep: "+str(t)+":")
			print("Latency: "+str(latency))
			print("hbmAcc: "+str(hbmAcc))
	
	return duration, Clatency, ChbmAcc

if __name__ == '__main__':
	sparsity = [0.95, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]	
	snnL, criL, hbmE = [], [], []
	startHbm, startLatency = 0, 0
	T = 30
	for s in sparsity:
		snnl, cril, hbm = main(True, s, startHbm, startLatency, T)
		snnL.append(snnl/T)
		criL.append(cril/T)
		hbmE.append(hbm/T)

	result = (snnL, criL, hbmE)
	files = open('sparsity_result.pkl','wb')
	pickle.dump(result, files)
	files.close()
	

	

	
	

