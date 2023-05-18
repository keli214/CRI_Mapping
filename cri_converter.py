#Imports

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.activation_based.neuron import IFNode, LIFNode
from quant_layer import *
from snntorch import spikegen
from spikingjelly.clock_driven import encoding
import csv 
import time
from tqdm import tqdm
from collections import defaultdict
from l2s.api import CRI_network
import cri_simulations
import snntorch as snn
import multiprocessing as mp
#import sparselinear as sl


def isSNNLayer(layer):
    # Test = isinstance(layer, MultiStepLIFNode) or isinstance(layer, LIFNode) or isinstance(layer, IFNode)
    # print(layer, Test)
    return isinstance(layer, MultiStepLIFNode) or isinstance(layer, LIFNode) or isinstance(layer, IFNode)
    
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

class Quantize_Network():
    def __init__(self, w_alpha = 1, dynamic_alpha = False):
        self.w_alpha = w_alpha #Range of the parameter (CSNN:4, Transformer: 5)
        self.dynamic_alpha = dynamic_alpha
        self.v_threshold = None
        self.w_bits = 16
        self.w_delta = self.w_alpha/(2**(self.w_bits-1)-1)
        self.weight_quant = weight_quantize_fn(self.w_bits)
        self.weight_quant.wgt_alpha = self.w_alpha
    
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
        
    def input_converter(self, input_data):
        self.input_shape = input_data.shape
        print('input batch data shape: ', input_data.shape)
        return self._input_converter(input_data)
        

    def _input_converter(self, input_data):
        current_input = input_data.view(input_data.size(0), -1)
        batch = []
        if self.backend == 'snntorch':
            for img in current_input:
                spikes = []
                rate_enc = spikegen.rate(img, self.num_steps)
                rate_enc = rate_enc.numpy()
                for step in rate_enc:
                    input_spike = ['a'+str(idx) for idx, axon in enumerate(step) if axon != 0]
                    bias_spike = ['a'+str(idx) for idx in range(self.bias_start_idx,len(self.axon_dict))] #firing bias neurons at each step
                    spikes.append(input_spike + bias_spike)
                    spikes.append(input_spike)
                batch.append(spikes)
            #TODO: write out the data into csv files
            if self.save_input:
                with open('/Volumes/export/isn/keli/code/CRI/data/cri_mnist.csv', 'w') as f:
                    write = csv.writer(f)
                    # write.writerow(fields)
                    write.writerows(batch)
            return batch
                    
        if self.backend == 'spikingjelly':
            # TODO: implement latency encoding with spikingjelly 
            for img in current_input:
                spikes = []
                for step in range(self.num_steps):
                    
                    input_spike = ['a'+str(idx) for idx, axon in enumerate(img)]
                    bias_spike = ['a'+str(idx) for idx in range(self.bias_start_idx,len(self.axon_dict))] #firing bias neurons at each step
                    spikes.append(input_spike + bias_spike)
                    # spikes.append(input_spike)
                batch.append(spikes)
            return batch
    
    def layer_converter(self, model):
        
        module_names = list(model._modules)
        
        # self.input_shape = (model.img_size_h, model.img_size_h)
        # print("Number of layers in net: ", len(module_names))
        
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
            self._avgPool_converter(layer)
            
        # elif isinstance(layer, sl.SparseLinear): #disabled for crisp since I couldn't install sl on crisp
        #     self._sparse_converter(layer)
    
        # elif isinstance(layer,  MultiStepLIFNode):
        #     if name == 'attn_lif':
        #         self._attention_converter(layer)
        else:
            pass
            # print("Unconvertered layer: ", layer)
        self.layer_index += 1
    
    def _attention_converter(self, model):
        print(f"Convert attention layer")
        #Flatten the current_input matrix to N*D (D = self.embed_dim, N = H*W)
        self.curr_input = np.transpose(self.curr_input.reshape(self.curr_input.shape[-2]*self.curr_input.shape[-1], self.embed_dim))#Hardcode for now 
        
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
                self._matrix_mul_cri(self.q, self.v)
                self._matrix_mul_cri(self.curr_input, self.k)
            self.layer_index += 1
        self.curr_input = np.transpose(self.curr_input)
                    
    def _attention_linear_converter(self, layer):
        print(f'Input layer shape(infeature, outfeature): {self.curr_input.shape} {self.curr_input.shape}')
        output_shape = self.curr_input.shape
        output = np.array([str(i) for i in range(self.neuron_offset, self.neuron_offset + np.prod(output_shape))]).reshape(output_shape)
        weights = layer.weight.detach().cpu().numpy()
        for n in range(self.curr_input.shape[0]):
            # print(self.curr_input[d], weights)
            for neuron_idx, neuron in enumerate(self.curr_input[n,:]):
                self.neuron_dict[neuron].extend([(output[n, neuron_idx], int(weight)) for idx, weight in enumerate(weights[n])])
        self.neuron_offset += np.prod(output_shape)
        print(f'curr_neuron_offset: {self.neuron_offset}')
        if layer.bias is not None and self.layer_index != self.output_layer:
            print(f'Constructing {layer.bias.shape[0]} bias axons for hidden linear layer')
            self._cri_bias(layer, output, atten_flag=True)
            self.axon_offset = len(self.axon_dict)
        return output.transpose(-2,-1)
    
    """ Given two matrix, maps the matrix multiplication a@b into CRI neurons connections
    
    Args:
        a: a np array of strings with T*N*D neuron names
        b: a np array of strings with T*N*D neuron names
        
    """
    def _matrix_mul_cri(self, x, y):
        #TODO: parallelize each time step 
        print(f"x.shape: {x.shape}")
        h, w = x.shape
        
        _, d = y.shape
        x_flatten = x.flatten() # (h*w)
        y_flatten = y.transpose().flatten() #(d*w)
        
        first_layer = np.array([str(i) for i in range(self.neuron_offset, self.neuron_offset + h*w*d)])
        # first_layer = first_layer.reshape(h*w*d)
        self.neuron_offset += h*w*d
        
        second_layer = np.array([str(i) for i in range(self.neuron_offset, self.neuron_offset + h*d)])
        # second_layer = second_layer.reshape(b, h*d)
        self.neuron_offset += h*d
        
        for idx, neuron in enumerate(x_flatten):
            for i in range(d):
                # print(f"idx%w + w*i + w*d*(idx//w): {idx%w + w*i + w*d*(idx//w)}")
                self.neuron_dict[neuron].extend([(first_layer[idx%w + w*i + w*d*(idx//w)], self.v_threshold)])
        for idx, neuron in enumerate(y_flatten):
            for i in range(h):
                # print(f"idx%(w*d): {idx%(w*d)}")
                self.neuron_dict[neuron].append([(first_layer[idx%(w*d)], self.v_threshold)])
        
        # for r in tqdm(range(b)):
        for idx, neuron in enumerate(first_layer):
            # print(f"idx//w: {idx//w}")
            self.neuron_dict[neuron].extend((second_layer[idx//w], self.v_threshold))
        
        second_layer = second_layer.reshape(h,d)
        # print(f'outputshape: {self.curr_input.shape}')
        self.curr_input = second_layer
        
    def _sparse_converter(self, layer):
        input_shape = layer.in_features
        output_shape = layer.out_features
        print(f'Input layer shape(infeature, outfeature): {input_shape} {output_shape}')
        axons = np.array([str(i) for i in range(0, input_shape)])
        output = np.array([str(i) for i in range(0, output_shape)])
        weight = layer.weight.detach().cpu().to_dense().numpy()
        print(f'Weight shape:{weight.shape}')
        curr_neuron_offset, next_neuron_offset = 0, input_shape
        print(f'curr_neuron_offset, next_neuron_offset: {curr_neuron_offset, next_neuron_offset}')
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
        # print(f'Converting layer: {layer}')
        if self.layer_index == self.input_layer:
            print('Constructing Axons from Linear Layer')
            #TODO: implement the linear layer conversion when it's the input layer  
        else:
            print('Constructing Neurons from Linear Layer')
            print("Hidden layer shape(infeature, outfeature): ", self.curr_input.shape, layer.out_features)
        output_shape = layer.out_features
        self.neuron_offset += np.prod(self.curr_input.shape)
        print(f'Neuron_offset: {self.neuron_offset}')
        output = np.array([str(i) for i in range(self.neuron_offset, self.neuron_offset + output_shape)])
        print(f'Last output: {output[-1]}')
        self._linear_weight(self.curr_input,output,layer)
        if layer.bias is not None and self.layer_index != self.output_layer:
            print(f'Constructing {layer.bias.shape[0]} bias axons for hidden linear layer')
            self._cri_bias(layer,output)
            self.axon_offset = len(self.axon_dict)
        self.curr_input = output
        print(f'Numer of neurons: {len(self.neuron_dict)}, number of axons: {len(self.axon_dict)}')
    
    def _linear_weight(self, input, outputs, layer):
        inputs = input.flatten()
        weight = layer.weight.detach().cpu().numpy()
        curr_neuron_offset, next_neuron_offset = self.neuron_offset-inputs.shape[0], self.neuron_offset   
        print(f'curr_neuron_offset, next_neuron_offset: {curr_neuron_offset, next_neuron_offset}')
        for neuron_idx, neuron in enumerate(weight.T):
            neuron_id = str(neuron_idx + curr_neuron_offset)
            neuron_entry = [(str(base_postsyn_id + next_neuron_offset), int(syn_weight)) for base_postsyn_id, syn_weight in enumerate(neuron) if syn_weight != 0]
            self.neuron_dict[neuron_id] = neuron_entry
        if self.layer_index == self.output_layer:
            print('Instantiate output neurons')
            for output_neuron in range(layer.out_features):
                neuron_id = str(output_neuron + next_neuron_offset)
                self.neuron_dict[neuron_id] = []
                self.output_neurons.append(neuron_id)
        
    def _conv_converter(self, layer):
        # print(f'Converting layer: {layer}')
        input_shape, output_shape, axons, output = None, None, None, None
        start_time = time.time()
        if self.layer_index == 0:
            print('Constructing Axons from Conv2d Layer')
            input_shape = self.input_shape
            output_shape = self._conv_shape(layer, self.input_shape)
            print(f'Input layer shape(infeature, outfeature): {input_shape} {output_shape}')
            axons = np.array(['a' + str(i) for i in range(np.prod(input_shape))]).reshape(self.input_shape)
            output = np.array([str(i) for i in range(np.prod(output_shape))]).reshape(output_shape)
            self._conv_weight(axons,output,layer)
            self.axon_offset = len(self.axon_dict)
        else:
            print('Constructing Neurons from Conv2d Layer')
            output_shape = self._conv_shape(layer, self.curr_input.shape)  
            print(f'Hidden layer shape(infeature, outfeature): {self.curr_input.shape} {output_shape}')             
            self.neuron_offset += np.prod(self.curr_input.shape)
            print(f'Neuron_offset: {self.neuron_offset}')
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
        start_time = time.time()
        #TODO: optimization
        # for n in tqdm(range(input.shape[0])):
        for c in tqdm(range(input.shape[0])):
            for row in range(pad_top,h_i-pad_top):
                for col in range(pad_left,w_i-pad_left):
                    patch = input[c,row-pad_top:row+pad_top+1,col-pad_left:col+pad_left+1]
                    for fil_idx, fil in enumerate(filters):
                        # print(fil.shape)
                        post_syn = output[fil_idx,row-pad_top,col-pad_left]
                        for i,neurons in enumerate(patch):
                            for j,neuron in enumerate(neurons):
                                if self.layer_index == 0:
                                    self.axon_dict[neuron].append((post_syn,int(fil[c,i,j])))
                                else: 
                                    self.neuron_dict[neuron].append((post_syn,int(fil[c,i,j])))
                                    # print(neuron, self.neuron_dict[neuron])
                                    # break
                                    
    def _avgPool_converter(self, layer):
        # print(f'Converting layer: {layer}')
        print('Constructing hidden avgpool layer')
        output_shape = self._avgPool_shape(layer,self.curr_input.shape)
        print(f'Hidden layer shape(infeature, outfeature): {self.curr_input.shape} {output_shape}')
        self.neuron_offset += np.prod(self.curr_input.shape)
        print(f'Neuron_offset: {self.neuron_offset}')
        output = np.array([str(i) for i in range(self.neuron_offset, self.neuron_offset + np.prod(output_shape))]).reshape(output_shape)
        print(f'Last output: {output.flatten()[-1]}')
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
    
    def _cri_bias(self, layer, outputs, atten_flag = False):
        biases = layer.bias.detach().cpu().numpy()
        for bias_idx, bias in enumerate(biases):
            bias_id = 'a'+str(bias_idx+self.axon_offset)
            if isinstance(layer, nn.Conv2d):
                self.axon_dict[bias_id] = [(str(neuron_idx),int(bias)) for neuron_idx in outputs[bias_idx].flatten()]
            elif isinstance(layer, nn.Linear):
                if atten_flag:
                    self.axon_dict[bias_id] = [(str(neuron_idx),int(bias)) for neuron_idx in outputs[bias_idx,:].flatten()]
                else:   
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
    
    def run_CRI_hw(self,inputList,hardwareNetwork):
        predictions = []
        #each image
        total_time_cri = 0
        for currInput in inputList:
            #initiate the softwareNetwork for each image
            cri_simulations.FPGA_Execution.fpga_controller.clear(len(self.neuron_dict), False, 0)  ##Num_neurons, simDump, coreOverride
            spikeRate = [0]*10
            #each time step
            for slice in currInput:
                start_time = time.time()
                hwSpike = hardwareNetwork.step(slice, membranePotential=False)
    #             print("Mem:",mem)
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
    
    def run_CRI_sw(self,inputList,softwareNetwork):
        predictions = []
        total_time_cri = 0
        #each image
        for currInput in inputList:
            #reset the membrane potential to zero
            softwareNetwork.simpleSim.initialize_sim_vars(len(neuronsDict))
            spikeRate = [0]*10
            #each time step
            for slice in currInput:
                start_time = time.time()
                swSpike = softwareNetwork.step(slice, membranePotential=False)
                end_time = time.time()
                total_time_cri = total_time_cri + end_time-start_time
                for spike in swSpike:
                    spikeIdx = int(spike) - self.bias_start_idx 
                    try: 
                        if spikeIdx >= 0: 
                            spikeRate[spikeIdx] += 1 
                    except:
                        print("SpikeIdx: ", spikeIdx,"\n SpikeRate:",spikeRate )
            predictions.append(spikeRate.index(max(spikeRate)))
        print(f"Total simulation execution time: {total_time_cri:.5f} s")
        return(predictions)
        
    def run(self, network, test_loader, loss_fun, backend='hardware'):
    
        start_time = time.time()
        # net.eval() #TODO: add Net as paramter
        test_loss = 0
        test_acc = 0
        test_samples = 0
        # test_loss_cri = 0
        # test_acc_cri = 0
        # loss = nn.MSELoss()
        # with torch.no_grad():
        for img, label in test_loader:
            cri_input = self.input_converter(img)
            # print(f'cri_input: {cri_input}')
            # img = img.to(device)
            # label = label.to(device)
            
            label_onehot = F.one_hot(label, 10).float()
            # out_fr = net(img)
            
            if backend == 'hardware':
                output = torch.tensor(self.run_CRI_hw(cri_input,network))
                
            elif backend == 'software':
                output = torch.tensor(self.run_CRI_sw(cri_input,network))
                
            else:
                print(f'Not supported {backend}')
                return 
            
            loss = loss_fun(output, label_onehot)
            # loss = F.mse_loss(out_fr, label_onehot)
            # loss_cri = loss_fun(cri_out, label_onehot)

            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (output.argmax(1) == label).float().sum().item()
            # functional.reset_net(net)
            
#             test_loss_cri += loss_cri.item() * label.numel()

#             test_acc_cri += (cri_out.argmax(1) == label).float().sum().item()
            

        test_time = time.time()
        test_speed = test_samples / (test_time - start_time)
        test_loss /= test_samples
        test_acc /= test_samples
        # writer.add_scalar('test_loss', test_loss)
        # writer.add_scalar('test_acc', test_acc)
        # writer.add_scalar('test_loss_cri', test_loss)
        # writer.add_scalar('test_acc_cri', test_acc)

        print(f'test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}')
        print(f'test speed ={test_speed: .4f} images/s')
    
    
                
# TODO: customer dataset class
# class CRIMnistDataset(Dataset):
#     def __init__():
#         pass 
        
    
