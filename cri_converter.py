import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
# from spikingjelly.clock_driven.neuron import MultiStepLIFNode
# from spikingjelly.activation_based.neuron import IFNode, LIFNode
# from quant.quant_layer import *
from snntorch import spikegen 
from spikingjelly.activation_based import encoding
import csv 
import time
from tqdm import tqdm
from collections import defaultdict
# from hs_api.api import CRI_network
# import hs_bridge
import snntorch as snn
import multiprocessing as mp
import numpy as np
from utils import isSNNLayer
#import sparselinear as sl

def weight_quantization(b):

    def uniform_quant(x, b):
        xdiv = x.mul((2 ** b - 1))
        xhard = xdiv.round().div(2 ** b - 1)
        #print('uniform quant bit: ', b)
        return xhard

    class _pq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input.div_(alpha)                          # weights are first divided by alpha
            input_c = input.clamp(min=-1, max=1)       # then clipped to [-1,1]
            sign = input_c.sign()
            input_abs = input_c.abs()
            input_q = uniform_quant(input_abs, b).mul(sign)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)               # rescale to the original range
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()             # grad for weights will not be clipped
            input, input_q = ctx.saved_tensors
            i = (input.abs()>1.).float()     # >1 means clipped. # output matrix is a form of [True, False, True, ...]
            sign = input.sign()              # output matrix is a form of [+1, -1, -1, +1, ...]
            # grad_alpha = (grad_output*(sign*i + (input_q-input)*(1-i))).sum()
            grad_alpha = (grad_output*(sign*i + (0.0)*(1-i))).sum()
            # above line, if i = True,  and sign = +1, "grad_alpha = grad_output * 1"
            #             if i = False, "grad_alpha = grad_output * (input_q-input)"
            grad_input = grad_input*(1-i)
            return grad_input, grad_alpha

    return _pq().apply

class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit, w_alpha):
        super(weight_quantize_fn, self).__init__()
        self.w_bit = w_bit-1
        self.weight_q = weight_quantization(b=self.w_bit)
        self.wgt_alpha = w_alpha
    def forward(self, weight):
        weight_q = self.weight_q(weight, self.wgt_alpha)
        return weight_q

class Quantize_Network():
    def __init__(self, w_alpha = 1, dynamic_alpha = False):
        self.w_alpha = w_alpha #Range of the parameter (CSNN:4, Transformer: 5)
        self.dynamic_alpha = dynamic_alpha
        self.v_threshold = None
        self.w_bits = 16
        self.w_delta = self.w_alpha/(2**(self.w_bits-1)-1)
        self.weight_quant = weight_quantize_fn(self.w_bits, self.w_alpha)
    
    def quantize(self, model):
        new_model = copy.deepcopy(model)
        start_time = time.time()
        module_names = list(new_model._modules)
        
        for k, name in enumerate(module_names):
            if len(list(new_model._modules[name]._modules)) > 0 and not isSNNLayer(new_model._modules[name]):
                print('Quantized: ',name)
                if name == 'block':
                    new_model._modules[name] = self.quantize_block(new_model._modules[name])
                else:
                    # if name == 'attn':
                    #     continue
                    new_model._modules[name] = self.quantize(new_model._modules[name])
            else:
                print('Quantized: ',name)
                if name == 'attn_lif':
                    continue
                quantized_layer = self._quantize(new_model._modules[name])
                new_model._modules[name] = quantized_layer
        
        end_time = time.time()
        print(f'Quantization time: {end_time - start_time}')
        return new_model
    
    def quantize_block(self, model):
        new_model = copy.deepcopy(model)
        module_names = list(new_model._modules)
        
        for k, name in enumerate(module_names):
            
            if len(list(new_model._modules[name]._modules)) > 0 and not isSNNLayer(new_model._modules[name]):
                if name.isnumeric() or name == 'attn' or name == 'mlp':
                    print('Block Quantized: ',name)
                    new_model._modules[name] = self.quantize_block(new_model._modules[name])
                else:
                    print('Block Unquantized: ', name)
            else:
                if name == 'attn_lif':
                    continue
                else:
                    new_model._modules[name] = self._quantize(new_model._modules[name])
        return new_model
    
    def _quantize(self, layer):
        if isSNNLayer(layer):
            return self._quantize_LIF(layer)

        elif isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            return self._quantize_layer(layer)
        
        else:
            return layer
        
    def _quantize_layer(self, layer):
        quantized_layer = copy.deepcopy(layer)
        
        if self.dynamic_alpha:
            # weight_range = abs(max(layer.weight.flatten()) - min(layer.weight.flatten()))
            self.w_alpha = abs(max(layer.weight.flatten()) - min(layer.weight.flatten()))
            self.w_delta = self.w_alpha/(2**(self.w_bits-1)-1)
            self.weight_quant = weight_quantize_fn(self.w_bits) #reinitialize the weight_quan
            self.weight_quant.wgt_alpha = self.w_alpha
        
        layer.weight = nn.Parameter(self.weight_quant(layer.weight))
        quantized_layer.weight = nn.Parameter(layer.weight/self.w_delta)
        
        if layer.bias is not None: #check if the layer has bias
            layer.bias = nn.Parameter(self.weight_quant(layer.bias))
            quantized_layer.bias = nn.Parameter(layer.bias/self.w_delta)
        
        
        return quantized_layer

    
    def _quantize_LIF(self,layer):
        
        layer.v_threshold = layer.v_threshold/self.w_delta
        self.v_threshold = layer.v_threshold
        
        return layer

class BN_Folder():
    def __init__(self):
        super().__init__()
        
    def fold(self, model):

        new_model = copy.deepcopy(model)

        module_names = list(new_model._modules)

        for k, name in enumerate(module_names):

            if len(list(new_model._modules[name]._modules)) > 0:
                
                new_model._modules[name] = self.fold(new_model._modules[name])

            else:
                if isinstance(new_model._modules[name], nn.BatchNorm2d) or isinstance(new_model._modules[name], nn.BatchNorm1d):
                    if isinstance(new_model._modules[module_names[k-1]], nn.Conv2d) or isinstance(new_model._modules[module_names[k-1]], nn.Linear):

                        # Folded BN
                        folded_conv = self._fold_conv_bn_eval(new_model._modules[module_names[k-1]], new_model._modules[name])

                        # Replace old weight values
                        # new_model._modules.pop(name) # Remove the BN layer
                        new_model._modules[module_names[k]] = nn.Identity()
                        new_model._modules[module_names[k-1]] = folded_conv # Replace the Convolutional Layer by the folded version

        return new_model


    def _bn_folding(self, prev_w, prev_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b, model_2d):
        if prev_b is None:
            prev_b = bn_rm.new_zeros(bn_rm.shape)
            
        bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)
          
        if model_2d:
            w_fold = prev_w * (bn_w * bn_var_rsqrt).view(-1, 1, 1, 1)
        else:
            w_fold = prev_w * (bn_w * bn_var_rsqrt).view(-1, 1)
        b_fold = (prev_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

        return torch.nn.Parameter(w_fold), torch.nn.Parameter(b_fold)
        
    def _fold_conv_bn_eval(self, prev, bn):
        assert(not (prev.training or bn.training)), "Fusion only for eval!"
        fused_prev = copy.deepcopy(prev)
        
        if isinstance(bn, nn.BatchNorm2d):
            fused_prev.weight, fused_prev.bias = self._bn_folding(fused_prev.weight, fused_prev.bias,
                                 bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias, True)
        else:
            fused_prev.weight, fused_prev.bias = self._bn_folding(fused_prev.weight, fused_prev.bias,
                                 bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias, False)

        return fused_prev


def pad_with(vector, pad_width, iaxis, kwargs):
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
        self.bias_start_idx = None
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
    def intput_converter_maxPool(self, input_data):
        current_input = input_data.view(input_data.size(0), -1)
        batch = []
        for img in current_input:
            input_spike = ['a' + str(idx) for idx, axon in enumerate(img) if axon != 0]
            batch.append(input_spike)
        return batch
    
    def input_converter_mul(self, q, k, v):
        q_flat = q.view(q.size(0), -1)
        k_flat = k.view(k.size(0), -1)
        v_flat = v.view(v.size(0), -1)
        inputs = [q_flat, k_flat, v_flat]
        batch = []
        offset = 0
        
        for i in range(3):
            # breakpoint()
            input_spike = np.array([['a' + str(idx + offset) for idx, axon in enumerate(b) if axon != 0] for b in inputs[i]])
            batch.append(input_spike.flatten())
            offset += len(input_spike)

        return batch
                
    
    def layer_converter(self, model):
        
        module_names = list(model._modules)
        
        #TODO: Remove incrementing the axon_offset in the layer converter functions
        axons = np.array(['a' + str(i) for i in range(np.prod(self.input_shape))]).reshape(self.input_shape)
        self.curr_input = axons
        self.axon_offset += np.prod(self.curr_input.shape)
        
        breakpoint()
        
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
                    self.curr_input = self._attention_linear_converter(model._modules[name])
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
        output_shape = (1, 1, 1024) #hardcoded for testing
        
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

    
    # """ Given two matrix, maps the matrix multiplication a@b into CRI neurons connections
    
    # Args:
    #     a: a np array of strings with T*N*D neuron names
    #     b: a np array of strings with T*N*D neuron names
        
    # """
    # def _matrix_mul_cri(self, x, y):
    #     #TODO: parallelize each time step 
    #     print(f"x.shape: {x.shape}")
    #     h, w = x.shape
        
    #     _, d = y.shape
    #     x_flatten = x.flatten() # (h*w)
    #     y_flatten = y.transpose().flatten() #(d*w)
        
    #     first_layer = np.array([str(i) for i in range(self.neuron_offset, self.neuron_offset + h*w*d)])
    #     # first_layer = first_layer.reshape(h*w*d)
    #     self.neuron_offset += h*w*d
        
    #     second_layer = np.array([str(i) for i in range(self.neuron_offset, self.neuron_offset + h*d)])
    #     # second_layer = second_layer.reshape(b, h*d)
    #     self.neuron_offset += h*d
        
    #     for idx, neuron in enumerate(x_flatten):
    #         for i in range(d):
    #             # print(f"idx%w + w*i + w*d*(idx//w): {idx%w + w*i + w*d*(idx//w)}")
    #             self.neuron_dict[neuron].extend([(first_layer[idx%w + w*i + w*d*(idx//w)], self.v_threshold)])
        
    #     for idx, neuron in enumerate(y_flatten):
    #         for i in range(h):
    #             # print(f"idx%(w*d): {idx%(w*d)}")
    #             self.neuron_dict[neuron].append([(first_layer[idx%(w*d)], self.v_threshold)])
        
    #     # for r in tqdm(range(b)):
    #     for idx, neuron in enumerate(first_layer):
    #         # print(f"idx//w: {idx//w}")
    #         self.neuron_dict[neuron].extend((second_layer[idx//w], self.v_threshold))
        
    #     second_layer = second_layer.reshape(h,d)
    #     # print(f'outputshape: {self.curr_input.shape}')
    #     self.curr_input = second_layer
    
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
        for chanIdx, channel in enumerate(x): #Note: add c later 
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
                        if test != 0:
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
                        if test != 0:
                            self.mul_neuron[neuron].append((second_layer[chanIdx, mIdx, colIdx], \
                                                        self.v_threshold))
        
        # breakpoint()
        # print(f'outputshape: {self.curr_input.shape}')
        self.curr_input = second_layer
        if test == 2:
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
            self._cri_bias(layer,output)
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
        
        h_k, w_k = layer.kernel_size
        h_o, w_o = output.shape[-2],output.shape[-1]
        pad_top, pad_left = h_k//2,w_k//2
        filters = layer.weight.detach().cpu().numpy()
        h_i, w_i = input.shape[-2], input.shape[-1] 
        input_layer = input.reshape(input.shape[-2], input.shape[-1])
        #TODO: add padding of 0s
        print(f'Input.shape: {input_layer.shape}')
        input_padded = np.pad(input_layer, 1, pad_with, padder=-1)
        input_padded = input_padded.reshape((input.shape[0], input_padded.shape[0], input_padded.shape[1]))
        print(f'input_padded: {input_padded.shape}')
        start_time = time.time()
        # for n in tqdm(range(input.shape[0])):
        for c in tqdm(range(input.shape[0])):
            for row in range(pad_top,h_i):
                for col in range(pad_left,w_i):
                    #Input axons/neurons
                    patch = input_padded[c,row-pad_top:row+pad_top+1,col-pad_left:col+pad_left+1]
                    for fil_idx, fil in enumerate(filters):
                        # print(fil.shape)
                        post_syn = output[fil_idx,row-pad_top,col-pad_left]
                        for i,neurons in enumerate(patch):
                            for j,neuron in enumerate(neurons):
                                if self.layer_index == 0:
                                    if fil[c,i,j] != 0 and neuron != -1:
                                        self.axon_dict['a' + str(neuron)].append((str(post_syn),int(fil[c,i,j])))
                                else: 
                                    if fil[c,i,j] != 0:
                                        self.neuron_dict[str(neuron)].append((str(post_syn),int(fil[c,i,j])))
        
        
                                    
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
        
    def _cri_fanout_test(self):
        for key in self.mul_axon.keys():
            self.total_axonSyn += len(self.mul_axon[key])
            if len(self.mul_axon[key]) > self.max_fan:
                self.max_fan = len(self.mul_axon[key])
        print("Total number of connections between axon and neuron: ", self.total_axonSyn)
        print("Max fan out of axon: ", self.max_fan)
        print('---')
        print("Number of neurons: ", len(self.mul_neuron))
        self.max_fan = 0
        for key in self.mul_neuron.keys():
            self.total_neuronSyn += len(self.mul_neuron[key])
            if len(self.mul_neuron[key]) > self.max_fan:
                self.max_fan = len(self.mul_neuron[key])
        print("Total number of connections between hidden and output layers: ", self.total_neuronSyn)
        print("Max fan out of neuron: ", self.max_fan)
    
    def run_CRI_hw(self,inputList,hardwareNetwork):
        predictions = []
        # each image
        total_time_cri = 0
        for currInput in inputList:
            # initiate the hardware for each image
            cri_simulations.FPGA_Execution.fpga_controller.clear(
                len(self.output_neurons), False, 0
            )  ##Num_neurons, simDump, coreOverride
            spikeRate = [0] * len(self.output_neurons)
            # each time step
            for slice in currInput:
                start_time = time.time()
                hwSpike, latency, hbmAcc = hardwareNetwork.step(
                    slice, membranePotential=False
                )
                end_time = time.time()
                total_time_cri = total_time_cri + end_time - start_time
                spikeIdx = [int(spike) - self.bias_start_idx for spike in hwSpike]
                for idx in spikeIdx:
                    spikeRate[idx] += 1
            #Since CRI only get spikes after the time step it has occurred
            if self.num_steps == 1:
                hwSpike, _, _ = hardwareNetwork.step([], membranePotential=False)
                spikeIdx = [int(spike) - self.bias_start_idx for spike in hwSpike]
                for idx in spikeIdx:
                    spikeRate[idx] += 1
            #Empty input to flush out the spike
            hwSpike, _, _ = hardwareNetwork.step([], membranePotential=False)
            spikeIdx = [int(spike) - self.bias_start_idx for spike in hwSpike]
            for idx in spikeIdx:
                spikeRate[idx] += 1
            predictions.append(spikeRate.index(max(spikeRate)))
        return predictions
    
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
    
    #Function used for maxpooling testing only 
    #Only process a batch of input for a single time step 
    def run_CRI_sw_testing(self,inputList,hardwareNetwork):
        # each image
        output = []
        for currInput in tqdm(inputList):
           
            swSpike = hardwareNetwork.step(currInput, membranePotential=False)
            spikeIdx1 = [int(spike) - self.bias_start_idx for spike in swSpike]
            # Empty input for output delay
            swSpike = hardwareNetwork.step([], membranePotential=False)
            spikeIdx2 = [int(spike) - self.bias_start_idx for spike in swSpike]
            # breakpoint()
            # Additional empty input for phase delay since the network is only 2 layers
            swSpike = hardwareNetwork.step([], membranePotential=False)
            spikeIdx3 = [int(spike) - self.bias_start_idx for spike in swSpike] 
            output.append(spikeIdx1+spikeIdx2+spikeIdx3)    
        return output 
    
    def run_CRI_hw_testing(self,inputList,softwareNetwork):
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
        
    
