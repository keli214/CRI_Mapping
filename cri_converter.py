import torch
import torch.nn as nn
import torch.nn.functional as F
# from spikingjelly.clock_driven.neuron import MultiStepLIFNode
# from spikingjelly.activation_based.neuron import IFNode, LIFNode
from snntorch import spikegen 
from spikingjelly.activation_based import encoding
import csv 
import time
from tqdm import tqdm
from collections import defaultdict
from hs_api.api import CRI_network
# import hs_bridge
import snntorch as snn
import multiprocessing as mp
import numpy as np
from utils import isSNNLayer
#import sparselinear as sl

# Helper function for converting the Conv2d layer
def _pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

class CRI_Converter():
    
    HIGH_SYNAPSE_WEIGHT = 1e6
    
    def __init__(self, num_steps = 4, input_layer = 0, output_layer = 11, input_shape = (1,28,28), backend='spikingjelly', v_threshold = 1e6, embed_dim = 4):
        self.axon_dict = defaultdict(list)
        self.neuron_dict = defaultdict(list)
        self.output_neurons = []
        self.input_shape = np.array(input_shape)
        self.num_steps = num_steps
        self.axon_offset = 0
        self.neuron_offset = 0
        self.backend = backend
        self.save_input = False
        self.bias_start_idx = 0
        self.output_start_idx = 0
        self.curr_input = None
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.layer_index = 0
        self.total_axonSyn = 0
        self.total_neuronSyn = 0
        self.max_fan = 0
        self.v_threshold = v_threshold
        
        #For spikformer only
        self.q = None
        self.v = None
        self.k = None
        self.embed_dim = embed_dim
        
        #For maxpool testing only
        self.maxPool_axon = defaultdict(list)
        self.maxPool_neuron = defaultdict(list)

        #For matrix multiplication testing only
        self.mul_axon1 = defaultdict(list)
        self.mul_neuron1 = defaultdict(list)
        self.mul_output1 = []
        
        self.mul_axon2 = defaultdict(list)
        self.mul_neuron2 = defaultdict(list)
        self.mul_output2 = []
        
        self.mul_axon = defaultdict(list)
        self.mul_neuron = defaultdict(list)
        self.mul_output = []
        
        
    
    '''Given an img, encode it into spikes and then convert it to axons lists
    '''
    def input_converter(self, input_data):
        self.input_shape = input_data.shape
        print('input batch data shape: ', input_data.shape)
        return self._input_converter(input_data)
        

    def _input_converter(self, input_data):
        encoder = encoding.PoissonEncoder()
        current_input = input_data.view(input_data.size(0), -1)
        batch = []
        for img in current_input:
            spikes = []
            for step in range(self.num_steps):
                encoded_img = encoder(img)
                input_spike = ['a'+str(idx) for idx, axon in enumerate(encoded_img) if axon != 0]
                bias_spike = ['a'+str(idx) for idx in range(self.bias_start_idx,len(self.axon_dict))] #firing bias neurons at each step
                spikes.append(input_spike + bias_spike)
            batch.append(spikes)
        #TODO: if we don't do rate encoding?
        if self.save_input:
            with open('/Volumes/export/isn/keli/code/CRI/data/cri_mnist.csv', 'w') as f:
                write = csv.writer(f)
                # write.writerow(fields)
                write.writerows(batch)
        return batch
    
    '''Input converter for maxpool testing only
        Given an converted img ouput, return a list of input axons ignore 
    '''
    def _intput_converter_maxPool(self, input_data):
        current_input = input_data.view(input_data.size(0), -1)
        batch = []
        for img in current_input:
            input_spike = ['a' + str(idx) for idx, axon in enumerate(img) if axon != 0]
            batch.append(input_spike)
        return batch
    
    '''Convert the q,k,v output from spikingjelly network 
    into spikes input for multiplication testing'''
    def _input_converter_mul(self, q, k, v):
        inputs = [q[0], k[0], v[0].transpose(-1,-2)]
        batch = []
        
        for b in range(q.shape[0]):
            first_input, second_input, input_spikes = [], [], []
            offset = 0
            for i in range(len(inputs)):
                spikes = ['a' + str(idx + offset) for idx, axon in enumerate(inputs[i][b].flatten()) if axon != 0]
                if i < 2:
                    first_input.extend(spikes)  
                else:
                    second_input.extend(spikes)
                offset += np.prod(inputs[i][b].shape)
            
            input_spikes = [first_input, second_input]

            batch.append(input_spikes)

        return batch
                
    
    def layer_converter(self, model):
        
        module_names = list(model._modules)
        
        #TODO: Remove incrementing the axon_offset in the layer converter functions
        axons = np.array(['a' + str(i) for i in range(np.prod(self.input_shape))]).reshape(self.input_shape)
        self.curr_input = axons
        self.axon_offset += np.prod(self.curr_input.shape)
        
        for k, name in enumerate(module_names):
            if len(list(model._modules[name]._modules)) > 0 and not isSNNLayer(model._modules[name]):
                if name == 'attn':
                    self._attention_converter(model._modules[name])
                else:
                    self.layer_converter(model._modules[name])
            else:
                self._layer_converter(model._modules[name], name)
    
    def _layer_converter(self, layer, name):
        # print(name, self.layer_index)
        if isinstance(layer, nn.Linear):
            self._linear_converter(layer)
        
        elif isinstance(layer, nn.Conv2d):
            self._conv_converter(layer)
        
        elif isinstance(layer, nn.AvgPool2d):
            self._avgPool_converter(layer)
            
        elif isinstance(layer, nn.MaxPool2d):
            self._maxPool_converter(layer)
            
        # elif isinstance(layer, sl.SparseLinear): #disabled for crisp since I couldn't install sl on crisp
        #     self._sparse_converter(layer)
    
        elif isSNNLayer(layer):
            if name == 'attn_lif':
                self._attention_converter(layer)
        else:
            pass
            # print("Unconvertered layer: ", layer)
        self.layer_index += 1
    
    """ Given a SSA block, the converter loop through all layers in the block.
        It converts SSA block into HiAER-Spike format by calling the corresponding 
        layer converter based on the layer itself.
    
    """
    def _attention_converter(self, model):
        print(f"Convert attention layer")
        # #Flatten the current_input matrix to N*D (D = self.embed_dim, N = H*W)
        # self.curr_input = np.transpose(self.curr_input.reshape(self.curr_input.shape[-2]*self.curr_input.shape[-1], self.embed_dim))#Hardcode for now 
        
        # For SSA testing only
        axons = np.array(['a' + str(i) for i in range(np.prod(self.input_shape))])
        self.curr_input = axons
        self.axon_offset += np.prod(self.curr_input.shape)

        module_names = list(model._modules)
        for k, name in enumerate(module_names):
            if not isSNNLayer(model._modules[name]):
                if name == 'q_linear':
                    self.q = self._attention_linear_converter(model._modules[name])
                elif name == 'k_linear':
                    self.k = self._attention_linear_converter(model._modules[name])
                elif name == 'v_linear':
                    self.v = self._attention_linear_converter(model._modules[name])
                elif name == 'proj_linear':
                    pass
                    # self.curr_input = self._attention_linear_converter(model._modules[name])
            elif name == 'attn_lif':
                self._matrix_mul_cri_testing(self.q, np.transpose(self.k, (0, 2, 1)), 1)
                self._matrix_mul_cri_testing(self.curr_input, self.v, 2)
            self.layer_index += 1
        # Do we need transpose here
        # self.curr_input = np.transpose(self.curr_input)
                    
    """ Given an attention layer, the function creates neurons and axons 
        by calling _attention_linear_weight. 
        It returns the output neurons to be stored for the proceeding 
        matrix multiplication layers in SSA models.
    """
    def _attention_linear_converter(self, layer):
        
        print(f'Input layer shape(infeature, outfeature):\
               {self.curr_input.shape} {layer.out_features}')
        # breakpoint()
        output_shape = (self.input_shape[0], self.embed_dim, self.embed_dim) #C,N,N
        
        #flatten the layer 
        output = np.array([str(i) for i in range(self.neuron_offset, self.neuron_offset + np.prod(output_shape))])
        
        weights = layer.weight.detach().cpu().numpy()
        
        # for n in range(self.curr_input.shape[0]):
        #     # print(self.curr_input[d], weights)
        #     for neuron_idx, neuron in enumerate(self.curr_input[n,:]):
                # self.neuron_dict[neuron].extend([(output[n, neuron_idx], int(weight)) for idx, weight in enumerate(weights[n])])
        
        # For SSA testing only
        for neuron in self.curr_input.flatten():       
            self.axon_dict[neuron].append([(output[idx], weight) for idx, weight in enumerate(weights)])
        
        # self.neuron_offset += np.prod(output_shape)
        # print(f'curr_neuron_offset: {self.neuron_offset}')
        
        if layer.bias is not None and self.layer_index != self.output_layer:
            print(f'Constructing {layer.bias.shape[0]} bias axons for hidden linear layer')
            self._cri_bias(layer, output, atten_flag=True)
            self.axon_offset = len(self.axon_dict)
        
        output = output.reshape(output_shape)
        print(output.shape)
        self.neuron_offset += np.prod(output.shape)
        return output
    
    """ Testing method for matrix multiplication in HiAER Spike
    
    Args:
        a: a np array of strings with T*N*D neuron names
        b: a np array of strings with T*N*D neuron names
        test: flag for testing mode, 0: not in testing, 1: qk, 2: kv 
    """
    def _matrix_mul_cri_testing(self, x, y, test):

        print(f"x shape: {x.shape}, y shape: {y.shape}")
        c, h, w = x.shape
        c, _, d = y.shape
        
        
        # breakpoint()

        # x_flatten = x.flatten() # (h*w)
        # y_flatten = y.transpose().flatten() #(d*w)
        
        #Creating the first layer of dummy neurons of the shape (h, w, d)
        first_layer = np.array([str(i) for i in range(self.neuron_offset, self.neuron_offset + c*h*w*d)])
        first_layer = first_layer.reshape(c,h,w,d)
        self.neuron_offset += c*h*w*d
        
        #Creating the output layer of the matrix multiplication
        second_layer = np.array([str(i) for i in range(self.neuron_offset, self.neuron_offset + c*h*d)])
        second_layer = second_layer.reshape(c,h,d)
        self.neuron_offset += c*h*d
        

        #Generates the synapses between input x and the first layer of dummy neurons
        for chanIdx, channel in enumerate(x):  
            for rowIdx, row in enumerate(channel):
                for idx, neuron in enumerate(row):
                    for i in range(d):
                    # breakpoint()
                        self.neuron_dict[neuron].append((first_layer[chanIdx, rowIdx, idx, i], \
                                                        self.v_threshold/2))
                        if test == 1:
                            self.mul_axon['a'+neuron].append((first_layer[chanIdx, rowIdx, idx, i], \
                                                        self.v_threshold/2))
                        if test == 2:
                            self.mul_neuron[neuron].append((first_layer[chanIdx, rowIdx, idx, i], \
                                                        self.v_threshold/2))

        #Generates the synapses between input y and the first layer of dummy neurons
        for chanIdx, channel in enumerate(y):
            for rowIdx, row in enumerate(channel):
                for idx, neuron in enumerate(row):
                    for i in range(h):
                        self.neuron_dict[neuron].append((first_layer[chanIdx, i, rowIdx, idx], \
                                                        self.v_threshold/2))
                        if test == 1:
                            self.mul_axon['a'+neuron].append((first_layer[chanIdx, i, rowIdx, idx], \
                                                        self.v_threshold/2))
                        if test == 2:
                            self.mul_axon['a'+neuron].append((first_layer[chanIdx, i, rowIdx, idx], \
                                                        self.v_threshold/2))
        
        
        #Generates the synapses between first layer of dummy neurons and output neurons
        for chanIdx, channel in enumerate(first_layer):
            for mIdx, matrix in enumerate(channel):
                for rowIdx, row in enumerate(matrix):
                    for colIdx, neuron in enumerate(row):
                        # breakpoint()
                        self.neuron_dict[neuron].append((second_layer[chanIdx, mIdx, colIdx], \
                                                    self.v_threshold))
                        if test == 1:
                            self.mul_neuron[neuron].append((second_layer[chanIdx, mIdx, colIdx], \
                                                        self.v_threshold))
                        if test == 2:
                            self.mul_neuron[neuron].append((second_layer[chanIdx, mIdx, colIdx], \
                                                        self.v_threshold))
        
        # breakpoint()
        # print(f'outputshape: {self.curr_input.shape}')
        self.curr_input = second_layer
        # if test == 1:
        #     self.output_start_idx = int(second_layer.flatten()[0])
        #     for output_neuron in second_layer.flatten():
        #         self.mul_neuron1[output_neuron] = []
        #     self.mul_output1 = second_layer.flatten().tolist()
        if test == 2:
            self.output_start_idx = int(second_layer.flatten()[0])
            for output_neuron in second_layer.flatten():
                self.mul_neuron[output_neuron] = []
            self.mul_output = second_layer.flatten().tolist()


    def _sparse_converter(self, layer):
        input_shape = layer.in_features
        output_shape = layer.out_features
        print(f'Input layer shape(infeature, outfeature): {input_shape} {output_shape}')
        axons = np.array([str(i) for i in range(0, input_shape)])
        output = np.array([str(i) for i in range(0, output_shape)])
        weight = layer.weight.detach().cpu().to_dense().numpy()
        print(f'Weight shape:{weight.shape}')
        curr_neuron_offset, next_neuron_offset = 0, input_shape
        # print(f'curr_neuron_offset, next_neuron_offset: {curr_neuron_offset, next_neuron_offset}')
        for neuron_idx, neuron in enumerate(weight.T):
            neuron_id = str(neuron_idx)
            neuron_entry = [(str(base_postsyn_id + next_neuron_offset), int(syn_weight)) for base_postsyn_id, syn_weight in enumerate(neuron) if syn_weight != 0]
            self.axon_dict[neuron_id] = neuron_entry
        print('Instantiate output neurons')
        for output_neuron in range(next_neuron_offset, next_neuron_offset + layer.out_features):
            self.neuron_dict[str(output_neuron)] = []
            self.output_neurons.append(neuron_id)
        print(f'Numer of neurons: {len(self.neuron_dict)}, number of axons: {len(self.axon_dict)}')
        
    def _linear_converter(self, layer): 
        output_shape = layer.out_features
        if self.layer_index == self.input_layer:
            print('Constructing axons from linear Layer')
            print(f'Input layer shape(infeature, outfeature): {self.input_shape} {output_shape}')
            self.axon_offset += np.prod(self.curr_input.shape)
        else:
            print('Constructing neurons from linear Layer')
            print("Hidden layer shape(infeature, outfeature): ", self.curr_input.shape, layer.out_features)
            self.neuron_offset += np.prod(self.curr_input.shape)
        output = np.array([str(i) for i in range(self.neuron_offset, self.neuron_offset + output_shape)])
        # print(f'Last output: {output[-1]}')
        self._linear_weight(self.curr_input,output,layer)
        self.curr_input = output
        print(f'Numer of neurons: {len(self.neuron_dict)}, number of axons: {len(self.axon_dict)}')
    
    def _linear_weight(self, input, outputs, layer):
        inputs = input.flatten()
        weight = layer.weight.detach().cpu().numpy().transpose()
        for neuron_idx, neuron in enumerate(weight):
            if self.layer_index == self.input_layer:  
                neuron_entry = [(str(base_postsyn_id), int(syn_weight)) for base_postsyn_id, syn_weight in enumerate(neuron) if syn_weight != 0]
                self.axon_dict[inputs[neuron_idx]] = neuron_entry
            else:
                curr_neuron_offset, next_neuron_offset = self.neuron_offset-inputs.shape[0], self.neuron_offset   
                # print(f'curr_neuron_offset, next_neuron_offset: {curr_neuron_offset, next_neuron_offset}')
                neuron_entry = [(str(base_postsyn_id + next_neuron_offset), int(syn_weight)) for base_postsyn_id, syn_weight in enumerate(neuron) if syn_weight != 0]
                neuron_id = str(neuron_idx + curr_neuron_offset)
                self.neuron_dict[neuron_id] = neuron_entry
        if self.layer_index == self.output_layer:
            print('Instantiate output neurons')
            for output_neuron in range(layer.out_features):
                neuron_id = str(output_neuron + self.neuron_offset)
                self.neuron_dict[neuron_id] = []
                self.output_neurons.append(neuron_id)
        elif layer.bias is not None:
            print(f'Constructing {layer.bias.shape[0]} bias axons for linear layer')
            self._cri_bias(layer,outputs)
            self.axon_offset = len(self.axon_dict)
        
    def _conv_converter(self, layer):
        # print(f'Converting layer: {layer}')
        input_shape, output_shape, axons, output = None, None, None, None
        start_time = time.time()
        if self.layer_index == 0:
            print('Constructing Axons from Conv2d Layer')
            output_shape = self._conv_shape(layer, self.input_shape)
            print(f'Input layer shape(infeature, outfeature): {self.input_shape} {output_shape}')
            # axons = np.array(['a' + str(i) for i in range(np.prod(self.input_shape))]).reshape(self.input_shape)
            axons = np.array([i for i in range(np.prod(self.input_shape))]).reshape(self.input_shape)
            # output = np.array([str(i) for i in range(np.prod(output_shape))]).reshape(output_shape)
            output = np.array([i for i in range(np.prod(output_shape))]).reshape(output_shape)
            
            self._conv_weight(axons,output,layer)
            self.axon_offset = len(self.axon_dict)
        else:
            print('Constructing Neurons from Conv2d Layer')
            output_shape = self._conv_shape(layer, self.curr_input.shape)  
            print(f'Hidden layer shape(infeature, outfeature): {self.curr_input.shape} {output_shape}')             
            self.neuron_offset += np.prod(self.curr_input.shape)
            # print(f'Neuron_offset: {self.neuron_offset}')
            output = np.array([str(i) for i in range(self.neuron_offset, self.neuron_offset + np.prod(output_shape))]).reshape(output_shape)
            # print(f'Last output: {output[-1][-1]}')
            self._conv_weight(self.curr_input,output,layer)

        if layer.bias is not None:
            print(f'Constructing {layer.bias.shape[0]} bias axons from conv layer.')
            self._cri_bias(layer,output)
            self.axon_offset = len(self.axon_dict)
        
        self.curr_input = output
        print(f'Numer of neurons: {len(self.neuron_dict)}, number of axons: {len(self.axon_dict)}')
        print(f'Converting {layer} takes {time.time()-start_time}')
        
    def _conv_weight(self, input, output, layer):
        
        h_i, w_i = input.shape[-2], input.shape[-1]
        h_o, w_o = output.shape[-2],output.shape[-1]
        h_k, w_k = layer.kernel_size
        h_s, w_s = layer.stride
        pad_top, pad_left = layer.padding
        filters = layer.weight.detach().cpu().numpy()

        # input_layer = input.reshape(input.shape[-2], input.shape[-1])
        #TODO: add padding of 0s
        print(f'Input.shape: {input.shape}')
        input_padded = np.pad(input, ((0, 0), layer.padding, layer.padding), _pad_with, padder=-1)
        # input_padded = input_padded.reshape((input.shape[0], input_padded.shape[0], input_padded.shape[1]))
        print(f'input_padded: {input_padded.shape}')
        start_time = time.time()
        # for n in tqdm(range(input.shape[0])):
        for c in range(input.shape[0]):
            for row in range(pad_top,h_i-h_k,h_s):
                padded_row = row-pad_top
                for col in range(pad_left,w_i-w_k,w_s):
                    #Input axons/neurons
                    padded_col = col-pad_left
                    patch = input_padded[c, padded_row:padded_row+h_k, padded_col:padded_col+w_k]
                    for fil_idx, filter in enumerate(filters):
                        # print(filter.shape)
                        post_r_idx = (padded_row)//h_s
                        post_c_idx = (padded_col)//w_s
                        post_syn = output[fil_idx, post_r_idx, post_c_idx]
                        for i,neurons in enumerate(patch):
                            for j,neuron in enumerate(neurons):
                                if self.layer_index == 0:
                                    if filter[c,i,j] != 0 and neuron != -1:
                                        self.axon_dict['a' + str(neuron)].append((str(post_syn),int(filter[c,i,j])))
                                else: 
                                    if filter[c,i,j] != 0:
                                        self.neuron_dict[str(neuron)].append((str(post_syn),int(filter[c,i,j])))
        
        
                                    
    def _avgPool_converter(self, layer):
        # print(f'Converting layer: {layer}')
        print('Constructing hidden avgpool layer')
        output_shape = self._avgPool_shape(layer,self.curr_input.shape)
        print(f'Hidden layer shape(infeature, outfeature): {self.curr_input.shape} {output_shape}')
        self.neuron_offset += np.prod(self.curr_input.shape)
        # print(f'Neuron_offset: {self.neuron_offset}')
        output = np.array([str(i) for i in range(self.neuron_offset, self.neuron_offset + np.prod(output_shape))]).reshape(output_shape)
        # print(f'Last output: {output.flatten()[-1]}')
        self._avgPool_weight(self.curr_input,output,layer)
        self.curr_input = output
        print(f'Numer of neurons: {len(self.neuron_dict)}, number of axons: {len(self.axon_dict)}')
    
    def _avgPool_weight(self, input, output, layer):
        h_k, w_k = layer.kernel_size,layer.kernel_size
        h_o, w_o = output.shape[-2],output.shape[-1]
        h_i, w_i = input.shape[-2], input.shape[-1] 
        pad_top, pad_left = h_k//2,w_k//2
        scaler = self.v_threshold #TODO: finetuning maybe?
        # print(h_i, w_i,input,output)
        for c in tqdm(range(input.shape[0])):
            for row in range(0,h_i,2):
                for col in range(0,w_i,2):
                    patch = input[c,row:row+pad_top+1, col:col+pad_left+1]
                    post_syn = str(output[c,row//2,col//2])
                    for i, neurons in enumerate(patch):
                        for j, neuron in enumerate(neurons):
                            self.neuron_dict[str(neuron)].append((post_syn,scaler))
                            
    def _maxPool_converter(self, layer):
        print('Constructing hidden maxPool layer')
        output_shape = self._maxPool_shape(layer,self.curr_input.shape)
        print(f'Hidden layer shape(infeature, outfeature): {self.curr_input.shape} {output_shape}')
        self.neuron_offset += np.prod(self.curr_input.shape)
        output = np.array([str(i) for i in range(self.neuron_offset, self.neuron_offset + np.prod(output_shape))]).reshape(output_shape)
        for output_neuron in range(np.prod(output_shape)):
            neuron_id = str(output_neuron)
            self.maxPool_neuron[neuron_id] = []
        self._maxPool_weight(self.curr_input,output,layer)
        self.curr_input = output
        print(f'Numer of neurons: {len(self.neuron_dict)}, number of axons: {len(self.axon_dict)}')
        
    def _maxPool_weight(self, input, output, layer):
        h_k, w_k = layer.kernel_size,layer.kernel_size
        h_o, w_o = output.shape[-2],output.shape[-1]
        h_i, w_i = input.shape[-2], input.shape[-1] 
        pad_top, pad_left = h_k//2,w_k//2
        scaler = self.v_threshold
        
        # print(h_i, w_i,input,output)
        for c in tqdm(range(input.shape[0])):
            for row in range(0,h_i,h_k):
                for col in range(0,w_i,w_k):
                    patch = input[c,row:row+h_k, col:col+w_k]
                    post_syn = str(output[c,row//2,col//2])
                    for i, neurons in enumerate(patch):
                        for j, neuron in enumerate(neurons):
                            self.neuron_dict[str(neuron)].append((post_syn,scaler))
                            #Testing
                            self.maxPool_axon['a' + str(neuron)].append((str(int(post_syn)-self.neuron_offset),scaler))
    
    def _cri_bias(self, layer, outputs, atten_flag = False):
        biases = layer.bias.detach().cpu().numpy()
        for bias_idx, bias in enumerate(biases):
            bias_id = 'a'+str(bias_idx+self.axon_offset)
            if isinstance(layer, nn.Conv2d):
                self.axon_dict[bias_id] = [(str(neuron_idx),int(bias)) for neuron_idx in outputs[bias_idx].flatten()]
            elif isinstance(layer, nn.Linear):
                self.axon_dict[bias_id] = [(str(outputs[bias_idx]),int(bias))]
            else:
                print(f'Unspported layer: {layer}')
    
    def _conv_shape(self, layer, input_shape):
        h_out = (input_shape[-2] + 2*layer.padding[0]-layer.dilation[0]*(layer.kernel_size[0]-1)-1)/layer.stride[0] +1
        w_out = (input_shape[-1] + 2*layer.padding[1]-layer.dilation[1]*(layer.kernel_size[1]-1)-1)/layer.stride[1] +1
        if len(input_shape) == 4:
            return np.array((input_shape[0],layer.out_channels,int(h_out),int(w_out)))
        else:
            return np.array((layer.out_channels,int(h_out),int(w_out)))
    
    def _avgPool_shape(self, layer ,input_shape):
        h_out = (input_shape[-2] + layer.padding*2 - (layer.kernel_size))/layer.stride +1
        w_out = (input_shape[-1] + layer.padding*2 - (layer.kernel_size))/layer.stride +1
        if len(input_shape) == 4:
            return np.array((input_shape[0],input_shape[1],int(h_out),int(w_out)))
        else:
            return np.array((input_shape[0],int(h_out),int(w_out)))
        
    def _maxPool_shape(self, layer, input_shape):
        
        h_out = (input_shape[-2] + layer.padding*2 - layer.dilation * (layer.kernel_size - 1) - 1)/layer.stride +1
        w_out = (input_shape[-1] + layer.padding*2 - layer.dilation * (layer.kernel_size - 1) - 1)/layer.stride +1
        if len(input_shape) == 4:
            return np.array((input_shape[0],input_shape[1],int(h_out),int(w_out)))
        else:
            return np.array((input_shape[0],int(h_out),int(w_out)))
    
    def _cri_fanout(self):
        for key in self.axon_dict.keys():
            self.total_axonSyn += len(self.axon_dict[key])
            if len(self.axon_dict[key]) > self.max_fan:
                self.max_fan = len(self.axon_dict[key])
        print("Total number of connections between axon and neuron: ", self.total_axonSyn)
        print("Max fan out of axon: ", self.max_fan)
        print('---')
        print("Number of neurons: ", len(self.neuron_dict))
        self.max_fan = 0
        for key in self.neuron_dict.keys():
            self.total_neuronSyn += len(self.neuron_dict[key])
            if len(self.neuron_dict[key]) > self.max_fan:
                self.max_fan = len(self.neuron_dict[key])
        print("Total number of connections between hidden and output layers: ", self.total_neuronSyn)
        print("Max fan out of neuron: ", self.max_fan)
        
    
    # def run_CRI_hw(self,inputList,hardwareNetwork):
    #     predictions = []
    #     # each image
    #     total_time_cri = 0
    #     for currInput in inputList:
    #         # initiate the hardware for each image
    #         hs_bridge.FPGA_Execution.fpga_controller.clear(
    #             len(self.output_neurons), False, 0
    #         )  ##Num_neurons, simDump, coreOverride
    #         spikeRate = [0] * len(self.output_neurons)
    #         # each time step
    #         for slice in currInput:
    #             start_time = time.time()
    #             hwSpike, latency, hbmAcc = hardwareNetwork.step(
    #                 slice, membranePotential=False
    #             )
    #             end_time = time.time()
    #             total_time_cri = total_time_cri + end_time - start_time
    #             spikeIdx = [int(spike) - self.bias_start_idx for spike in hwSpike]
    #             for idx in spikeIdx:
    #                 spikeRate[idx] += 1
    #         #Since CRI only get spikes after the time step it has occurred
    #         if self.num_steps == 1:
    #             hwSpike, _, _ = hardwareNetwork.step([], membranePotential=False)
    #             spikeIdx = [int(spike) - self.bias_start_idx for spike in hwSpike]
    #             for idx in spikeIdx:
    #                 spikeRate[idx] += 1
    #         #Empty input to flush out the spike
    #         hwSpike, _, _ = hardwareNetwork.step([], membranePotential=False)
    #         spikeIdx = [int(spike) - self.bias_start_idx for spike in hwSpike]
    #         for idx in spikeIdx:
    #             spikeRate[idx] += 1
    #         predictions.append(spikeRate.index(max(spikeRate)))
    #     return predictions
    
    ''' Given a input list of spikes and a softwareNetwork
        The function calls the step function of HiAER-Spike network 
        And returns the prediction in the idx format
        It also measures the total execution time of the software simulation
    '''
    def run_CRI_sw(self,inputList,softwareNetwork):
        predictions = []
        total_time_cri = 0
        # each image
        for currInput in tqdm(inputList):
            # reset the membrane potential to zero
            # Testing: change from output_neurons to maxPool_neuron
            softwareNetwork.simpleSim.initialize_sim_vars(len(self.output_neurons))
            spikeRate = [0] * len(self.output_neurons)
            # each time step
            for slice in currInput:
                # breakpoint()
                start_time = time.time()
                swSpike = softwareNetwork.step(slice, membranePotential=False)
                end_time = time.time()
                total_time_cri = total_time_cri + end_time - start_time
                spikeIdx = [int(spike) - self.bias_start_idx for spike in swSpike]
                for idx in spikeIdx:
                    spikeRate[idx] += 1
            
            if self.num_steps == 1:
                #Empty input for output delay since HiAER spike only get spikes after the spikes have occurred
                swSpike = softwareNetwork.step([], membranePotential=False)
                spikeIdx = [int(spike) - self.bias_start_idx for spike in swSpike]
                for idx in spikeIdx:
                    spikeRate[idx] += 1
            #Empty input for output delay 
            swSpike = softwareNetwork.step([], membranePotential=False)
            spikeIdx = [int(spike) - self.bias_start_idx for spike in swSpike]
            for idx in spikeIdx:
                spikeRate[idx] += 1
            predictions.append(spikeRate.index(max(spikeRate)))
        return predictions
    
    # def _run_CRI_hw_ssa_testing(self, inputList,hardwareNetwork):
    #     output = []
    #     for currInput in tqdm(inputList):
    #         # initiate the hardware for each image
    #         hs_bridge.FPGA_Execution.fpga_controller.clear(
    #             len(self.mul_neuron), False, 0
    #         )  ##Num_neurons, simDump, coreOverride
            
    #         # Given the network is 5 layers, feed in the two inputs at different time steps          
    #         spikeOut1, latency, hbmAcc = hardwareNetwork.step(currInput[0], membranePotential=False)
            
    #         spikeOut2, _, _ = hardwareNetwork.step([], membranePotential=False)
            
    #         spikeOut3, _, _ = hardwareNetwork.step([], membranePotential=False)
            
    #         spikeOut = spikeOut1 + spikeOut2 + spikeOut3
    #         spikeIn = []
    #         for i in range(len(spikeOut)):
    #             #Offset = Total # of neurons and axon in the first model
    #             offset = self.embed_dim * self.embed_dim * 3 + self.embed_dim * self.embed_dim * self.embed_dim 
    #             spikeIn.append('a'+str(int(spikeOut[i])-offset))
                
    #         for i in range(len(currInput[1])):
    #             #Offset = N * N (starts from the second input)
    #             offset = self.embed_dim * self.embed_dim
    #             currInput[1][i] = 'a' + str(int(currInput[1][i][1:])-offset)
            
    #         hs_bridge.FPGA_Execution.fpga_controller.clear(
    #             len(self.mul_neuron), False, 0
    #         )  ##Num_neurons, simDump, coreOverride
            
    #         hwSpike1, _, _ = hardwareNetwork.step(currInput[1]+spikeIn, membranePotential=False)
    #         spikeIdx1 = [int(spike) - self.output_start_idx for spike in hwSpike1] 
            
    #         hwSpike2 = hardwareNetwork.step([], membranePotential=False)
    #         spikeIdx2 = [int(spike) - self.output_start_idx for spike in hwSpike2] 
            
    #         hwSpike3 = hardwareNetwork.step([], membranePotential=False)
    #         spikeIdx3 = [int(spike) - self.output_start_idx for spike in hwSpike3] 
            
    #         output.append(spikeIdx1+ spikeIdx2 + spikeIdx3)    
            
    #     return spikeOut, output 
    
    #Function used for SSA testing only 
    #Only process a batch of input for a single time step 
    def _run_CRI_sw_ssa_testing(self,inputList,softwareNetwork):
        # each image
        output = []
        for currInput in tqdm(inputList):
            softwareNetwork.simpleSim.initialize_sim_vars(len(self.mul_neuron))
            
            # Given the network is 5 layers, feed in the two inputs at different time steps
            swSpike0 = softwareNetwork.step(currInput[0], membranePotential=False)
            spikeIdx0 = [int(spike) - self.output_start_idx for spike in swSpike0] 
            
            swSpike1 = softwareNetwork.step([], membranePotential=False)
            spikeIdx1 = [int(spike) - self.output_start_idx for spike in swSpike1] 
            
            swSpike2 = softwareNetwork.step([], membranePotential=False)
            spikeIdx2 = [int(spike) - self.output_start_idx for spike in swSpike2] 
            
            spikeOut = spikeIdx0 + spikeIdx1 + spikeIdx2
        
            # spikeIn = []
            # for i in range(len(spikeOut)):
            #     #Offset = Total # of neurons and axon in the first model
            #     offset = self.embed_dim * self.embed_dim * 3 + self.embed_dim * self.embed_dim * self.embed_dim 
            #     spikeIn.append('a'+str(int(spikeOut[i])-offset))
                # breakpoint()
                
            # if len(currInput[1]) > 0:
            #     breakpoint()
                   
            # for i in range(len(currInput[1])):
            #     #Offset = N * N (starts from the second input)
            #     offset = self.embed_dim * self.embed_dim
            #     currInput[1][i] = 'a' + str(int(currInput[1][i][1:])-offset)
            
                
            # softwareNetwork.simpleSim.initialize_sim_vars(len(self.mul_neuron1))

            swSpike3 = softwareNetwork.step(currInput[1], membranePotential=False)
            spikeIdx3 = [int(spike) - self.output_start_idx for spike in swSpike3] 
            
            swSpike4 = softwareNetwork.step([], membranePotential=False)
            spikeIdx4 = [int(spike) - self.output_start_idx for spike in swSpike4] 
            
            swSpike5 = softwareNetwork.step([], membranePotential=False)
            spikeIdx5 = [int(spike) - self.output_start_idx for spike in swSpike5] 
            
            output.append(spikeIdx3+ spikeIdx4 + spikeIdx5)    
            
#             if len(spikeIdx1) > 0 or len(spikeIdx2) > 0 or len(spikeIdx3) > 0:
#                 breakpoint()
            
        return spikeOut, output 
    
    #Function used for maxpooling testing only 
    #Only process a batch of input for a single time step 
    def _run_CRI_sw_testing(self,inputList,softwareNetwork):
        # each image
        output = []
        for currInput in tqdm(inputList):
            
           
            swSpike = softwareNetwork.step(currInput, membranePotential=False)
            spikeIdx1 = [int(spike) - self.output_start_idx for spike in swSpike]
            # Empty input for output delay
            swSpike = softwareNetwork.step([], membranePotential=False)
            spikeIdx2 = [int(spike) - self.output_start_idx for spike in swSpike]
            # breakpoint()
            # Additional empty input for phase delay since the network is only 2 layers
            swSpike = softwareNetwork.step([], membranePotential=False)
            spikeIdx3 = [int(spike) - self.output_start_idx for spike in swSpike] 
            
            output.append(spikeIdx1+spikeIdx2+spikeIdx3)    
            
        return output 
    
    def _run_CRI_hw_testing(self,inputList,hardwareNetwork):
        # each image
        output = []
        for currInput in tqdm(inputList):
           
            hwSpike, latency, hbmAcc = hardwareNetwork.step(
                    slice, membranePotential=False
                )
            spikeIdx1 = [int(spike) - self.bias_start_idx for spike in hwSpike]
            # Empty input for output delay
            hwSpike, latency, hbmAcc = hardwareNetwork.step(
                    [], membranePotential=False
                )
            spikeIdx2 = [int(spike) - self.bias_start_idx for spike in hwSpike]
            # breakpoint()
            # Additional empty input for phase delay since the network is only 2 layers
            hwSpike, latency, hbmAcc = hardwareNetwork.step(
                    [], membranePotential=False
                )
            spikeIdx3 = [int(spike) - self.bias_start_idx for spike in hwSpike] 
            output.append(spikeIdx1+spikeIdx2+spikeIdx3)    
        return output 
    

                
# TODO: customer dataset class
# class CRIMnistDataset(Dataset):
#     def __init__():
#         pass 
        
    
