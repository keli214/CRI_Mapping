#Imports

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.activation_based.neuron import IFNode
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
import sparselinear as sl


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
    w_alpha = 4 #Range of the parameter (CSNN:4)
    w_bits = 16
    w_delta = w_alpha/(2**(w_bits-1)-1)
    weight_quant = weight_quantize_fn(w_bits)
    weight_quant.wgt_alpha = w_alpha
    
    def __init__(self, dynamic_alpha = False):
        self.dynamic_alpha = dynamic_alpha
        self.v_threshold = None
    
    def quantize(self, model):
        new_model = copy.deepcopy(model)
        start_time = time.time()
        module_names = list(new_model._modules)
        
        for k, name in enumerate(module_names):
            if len(list(new_model._modules[name]._modules)) > 0 and not isinstance(new_model._modules[name], MultiStepLIFNode) and not isinstance(new_model._modules[name], IFNode):
                print('Quantized: ',name)
                if name == 'block':
                    new_model._modules[name] = self.quantize_block(new_model._modules[name])
                else:
                    new_model._modules[name] = self.quantize(new_model._modules[name])
            else:
                print('Quantized: ',name)
                quantized_layer = self._quantize(new_model._modules[name])
                new_model._modules[name] = quantized_layer
        
        end_time = time.time()
        print(f'Quantization time: {end_time - start_time}')
        return new_model
    
    def quantize_block(self, model):
        new_model = copy.deepcopy(model)
        module_names = list(new_model._modules)
        
        for k, name in enumerate(module_names):
            
            if len(list(new_model._modules[name]._modules)) > 0 and not isinstance(new_model._modules[name], MultiStepLIFNode):
                if name.isnumeric() or name == 'attn' or name == 'mlp':
                    print('Quantized: ',name)
                    new_model._modules[name] = self.quantize_block(new_model._modules[name])
                else:
                    print('Unquantized: ', name)
            else:
                if name == 'attn_lif':
                    continue
                else:
                    new_model._modules[name] = self._quantize(new_model._modules[name])
        return new_model
    
    def _quantize(self, layer):
        if isinstance(layer, MultiStepLIFNode) or isinstance(layer, IFNode):
            return self._quantize_LIF(layer)

        elif isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            return self._quantize_layer(layer)
        
        else:
            return layer
        
    def _quantize_layer(self, layer):
        quantized_layer = copy.deepcopy(layer)
        
        if self.dynamic_alpha:
            # weight_range = abs(max(layer.weight.flatten()) - min(layer.weight.flatten()))
            Quantize_Network.w_alpha = abs(max(layer.weight.flatten()) - min(layer.weight.flatten()))
            Quantize_Network.w_delta = Quantize_Network.w_alpha/(2**(Quantize_Network.w_bits-1)-1)
            Quantize_Network.weight_quant = weight_quantize_fn(Quantize_Network.w_bits) #reinitialize the weight_quan
            Quantize_Network.weight_quant.wgt_alpha = Quantize_Network.w_alpha
        
        layer.weight = nn.Parameter(Quantize_Network.weight_quant(layer.weight))
        quantized_layer.weight = nn.Parameter(layer.weight/Quantize_Network.w_delta)
        
        if layer.bias is not None: #check if the layer has bias
            layer.bias = nn.Parameter(Quantize_Network.weight_quant(layer.bias))
            quantized_layer.bias = nn.Parameter(layer.bias/Quantize_Network.w_delta)
        
        
        return quantized_layer

    
    def _quantize_LIF(self,layer):
        
        layer.v_threshold = layer.v_threshold/Quantize_Network.w_delta
        self.v_threshold = layer.v_threshold
        
        return layer

    
class CRI_Converter():
    
    HIGH_SYNAPSE_WEIGHT = 1e6
    
    def __init__(self, num_steps = 4, input_layer = 0, output_layer = 11, input_shape = np.array((1,28,28)), backend='spikingjelly', v_threshold = 1e6):
        self.axon_dict = defaultdict(list)
        self.neuron_dict = defaultdict(list)
        self.output_neurons = []
        self.input_shape = input_shape
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
        print("Number of layers in net: ", len(module_names))
        
        for k, name in enumerate(module_names):
            if len(list(model._modules[name]._modules)) > 0 and not isinstance(model._modules[name], MultiStepLIFNode) and not isinstance(model._modules[name], IFNode):
                self.layer_converter(model._modules[name])
            else:
                self._layer_converter(model._modules[name])
    
    def _layer_converter(self, layer):
        if isinstance(layer, nn.Linear):
            self._linear_converter(layer)
        
        elif isinstance(layer, nn.Conv2d):
            self._conv_converter(layer)
        
        elif isinstance(layer, nn.AvgPool2d):
            self._avgPool_converter(layer)
            
        elif isinstance(layer, sl.SparseLinear):
            self._sparse_converter(layer)
            
        else:
            pass
            # print("Unconvertered layer: ", layer)
        self.layer_index += 1
    
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
            print("Hidden layer shape(infeature, outfeature): ", layer.in_features, layer.out_features)
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
            print(f'Last output: {output[-1][-1]}')
            self._conv_weight(self.curr_input,output,layer)

        if layer.bias is not None:
            print(f'Constructing {layer.bias.shape[0]} bias axons for input layer.')
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
    
    def _cri_bias(self, layer, outputs):
        biases = layer.bias.detach().cpu().numpy()
        for bias_idx, bias in enumerate(biases):
            bias_id = 'a'+str(bias_idx+self.axon_offset)
            if isinstance(layer, nn.Conv2d):
                self.axon_dict[bias_id] = [(str(neuron_idx),int(bias)) for neuron_idx in outputs[bias_idx].flatten()]
            elif isinstance(layer, nn.Linear):
                self.axon_dict[bias_id] = [(str(outputs[bias_id]),int(bias))]
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
        h_out = (input_shape[-2] + layer.padding*2 - (layer.kernel_size-1))/layer.stride +1
        w_out = (input_shape[-1] + layer.padding*2 - (layer.kernel_size-1))/layer.stride +1
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
        
    def _run_cri(self):
        #TODO: refractor code
        pass
    
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
    
    def validate_cri(self, net, hardwareNetwork,test_loader, out_dir, device):
        start_time = time.time()
        net.eval() #TODO: add Net as paramter
        test_loss = 0
        test_acc = 0
        test_samples = 0
        test_loss_cri = 0
        test_acc_cri = 0

        with torch.no_grad():
            for img, label in test_loader:

                cri_input = self.input_converter(img)
                print(f'cri_input: {cri_input}')
                img = img.to(device)
                label = label.to(device)
                label_onehot = F.one_hot(label, 10).float()
                out_fr = net(img)
                cri_out = torch.tensor(self.run_CRI_hw(cri_input,hardwareNetwork)).to(device)
                loss = F.mse_loss(out_fr, label_onehot)
                loss_cri = F.mse_loss(cri_out, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_loss_cri += loss_cri.item() * label.numel()

                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                test_acc_cri += (cri_out.argmax(1) == label).float().sum().item()
                functional.reset_net(net)

            test_time = time.time()
            test_speed = test_samples / (test_time - start_time)
            test_loss /= test_samples
            test_acc /= test_samples
            writer.add_scalar('test_loss', test_loss)
            writer.add_scalar('test_acc', test_acc)
            writer.add_scalar('test_loss_cri', test_loss)
            writer.add_scalar('test_acc_cri', test_acc)

        print(f'test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}')
        print(f'test speed ={test_speed: .4f} images/s')
# TODO: customer dataset class
# class CRIMnistDataset(Dataset):
#     def __init__():
#         pass 
        
    