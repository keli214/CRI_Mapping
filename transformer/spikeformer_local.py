import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron
# from timm.models.layers import to_2tuple, trunc_normal_
# from timm.models.registry import register_model
# from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from functools import partial
import numpy as np 
__all__ = ['spikeformer']
from torch.utils.tensorboard import SummaryWriter
from quantization import act_quantization
from local_attention.local_attention import LocalAttention

def activation_visual(x, layer):
    writer = SummaryWriter('runs/transformer/activations')
    img_batch = np.array(x.flatten(0, 1).cpu())
    img_batch = np.expand_dims(img_batch, axis=1)
    weight_shape = img_batch.shape
    num_kernels = weight_shape[0]
    for k in range(num_kernels):
        # print(img_batch.shape, weight_shape)
        writer.add_image(f'activation_{layer} kernel_{k}', img_batch[k], 0, dataformats='CHW')
    writer.close()

def activation_histogram(x):
    writer = SummaryWriter('runs/transformer/activations')
    flattened_weights = x.flatten().cpu()
    # weight_range = abs(max(flattened_weights) - min(flattened_weights))
    tag = f"Range_{x.shape}"
    writer.add_histogram(tag, flattened_weights, global_step=0, bins='tensorflow')
    writer.close()

    
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_linear = nn.Linear(in_features, hidden_features)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = neuron.LIFNode(tau=2.0)

        self.fc2_linear = nn.Linear(hidden_features, out_features)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = neuron.LIFNode(tau=2.0)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T,B,N,C = x.shape
        x_ = x.flatten(0, 1)
        x = self.fc1_linear(x_)
        x = self.fc1_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.c_hidden).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_linear(x.flatten(0,1))
        x = self.fc2_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        x = self.fc2_lif(x)
        return x


class SSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = neuron.LIFNode(tau=2.0)

        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = neuron.LIFNode(tau=2.0)

        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = neuron.LIFNode(tau=2.0)
        #NOTE: Threshld should not be quantized for attn_lif
        self.attn_lif = neuron.LIFNode(tau=2.0, v_threshold = 0.5)

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = neuron.LIFNode(tau=2.0)
        #NOTE: testing the accuracy w/ single bit quantization
        self.act_alq = act_quantization(1) #activation quantization 
        self.act_alpha = 8.0 #scaling factor

    def forward(self, x):
        T,B,N,C = x.shape # C: embed dim
                        

        x_for_qkv = x.flatten(0, 1)  # TB, N, C
        q_linear_out = self.q_linear(x_for_qkv)  # [TB, N, C]
        q_linear_out = self.q_bn(q_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        q_linear_out = self.q_lif(q_linear_out)
        q = q_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous() # T, B, num_heads, num_patches, patch_size

        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T,B, N, C).contiguous()
        k_linear_out = self.k_lif(k_linear_out)
        k = k_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T,B, N, C).contiguous()
        v_linear_out = self.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        
        # attn = (q @ k.transpose(-2, -1)) 
        
        # attn = self.act_alq(attn, self.act_alpha).clone().detach().requires_grad_(True)
        
        # x = attn @ v
        
        # x = self.act_alq(x, self.act_alpha).clone().detach().requires_grad_(True)
        # breakpoint()
        
        attn = LocalAttention(
          window_size = C//self.num_heads//12,       # window size. 512 is optimal, but 256 or 128 yields good enough results
          look_backward = 1,       # each window looks at the window before
          look_forward = 0,        # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
          dropout = 0,           # post-attention dropout
          exact_windowsize = False # if this is set to true, in the causal setting, each query will see at maximum the number of keys equal to the window size
        )
        
        x = attn(q,k,v)
        
        # breakpoint()
        
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        
        x = self.attn_lif(x)
    
        x = x.flatten(0, 1)
        
        x = self.proj_lif(self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C))
        
        return x
    
    def forward_qkv(self, x):
        T,B,N,C = x.shape # C: embed dim
                        

        x_for_qkv = x.flatten(0, 1)  # TB, N, C
        q_linear_out = self.q_linear(x_for_qkv)  # [TB, N, C]
        q_linear_out = self.q_bn(q_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        q_linear_out = self.q_lif(q_linear_out)
        q = q_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T,B, N, C).contiguous()
        k_linear_out = self.k_lif(k_linear_out)
        k = k_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T,B, N, C).contiguous()
        v_linear_out = self.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        
        return q,k,v
    
    def forward_test(self, x):
        T,B,N,C = x.shape
        x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C))
        return x
        

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        # Note: layer norm wasn't used in the actual implementation
        # self.norm1 = norm_layer(dim) 
        # self.norm2 = norm_layer(dim)
        
        self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        # print(self.norm1, self.norm2)
        x_attn = self.attn(x)
        x = x + x_attn
        x = x + self.mlp(x)

        return x
    
    def forward_test(self, x, x_attn):
        x_attn = self.attn.forward_test(x_attn)
        x = x + x_attn
        x = x + self.mlp(x)

        return x
    
    def forward_qkv(self, x):
        q,k,v = self.attn.forward_qkv(x)
        return q,k,v


class SPS(nn.Module):
    def __init__(self, img_size_h=28, img_size_w=28, patch_size=7, in_channels=1, embed_dims=16):
        super().__init__()
        self.image_size = [img_size_h, img_size_w] #32*32
        # patch_size = []
        self.patch_size = [patch_size, patch_size]
        self.C = in_channels #3
        
        self.H, self.W = self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]
        self.num_patches = self.H * self.W #1024
        self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims//8)
        self.proj_lif = neuron.LIFNode(tau=2.0)

        self.proj_conv1 = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims//4)
        self.proj_lif1 = neuron.LIFNode(tau=2.0)

        self.proj_conv2 = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims//2)
        self.proj_lif2 = neuron.LIFNode(tau=2.0)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = neuron.LIFNode(tau=2.0)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = neuron.LIFNode(tau=2.0)

    def forward(self, x):
        
        T, B, C, H, W = x.shape #time_step, batch_num, color_channel, h, w
        x = self.proj_conv(x.flatten(0, 1)) # have some fire value
        # breakpoint()
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif(x).flatten(0, 1).contiguous()
        
        # print("T0:", x.reshape(T,B, -1, H, W)[0,0,:,0,0] )
        # print("T1:", x.reshape(T,B, -1, H, W)[1,0,:,0,0] 
        
        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif1(x).flatten(0, 1).contiguous()

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif2(x).flatten(0, 1).contiguous()
        x = self.maxpool2(x)

        x = self.proj_conv3(x)
        x = self.proj_bn3(x).reshape(T, B, -1, H//2, W//2).contiguous()
        x = self.proj_lif3(x).flatten(0, 1).contiguous()
        x = self.maxpool3(x)

        x_feat = x.reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.rpe_conv(x)
        x = self.rpe_bn(x).reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.rpe_lif(x)
        x = x + x_feat

        x = x.flatten(-2).transpose(-1, -2)  # T,B,N,D
        return x


class Spikeformer(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=7, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T = 4
                 ):
        super().__init__()
        self.T = T  # time step
        self.num_classes = num_classes
        self.depths = depths

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        
        patch_embed = SPS(img_size_h=img_size_h,
                                 img_size_w=img_size_w,
                                 patch_size=patch_size,
                                 in_channels=in_channels,
                                 embed_dims=embed_dims)

        block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
                      
                      
    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x = patch_embed(x)
        for blk in block:
            x = blk(x)
        return x.mean(2)
    
    def forward_qkv(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x = patch_embed(x)
        
        q, k, v = block[0].forward_qkv(x)
        return q,k,v
    
    def forward_embed(self,x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x = patch_embed(x)
        return x
    
    def forward_test(self, x, x_attn):
        block = getattr(self, f"block")
        x = block[0].forward_test(x, x_attn)
        x = x.mean(2)
        x = self.head(x.mean(0)) 
        return x    
        
    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = self.head(x.mean(0)) 
        return x


# @register_model
def spikeformer(pretrained=False, **kwargs):
    model = Spikeformer(
        # img_size_h=224, img_size_h=224,
        # patch_size=16, embed_dims=768, num_heads=12, mlp_ratios=4,
        # in_channels=3, num_classes=1000, qkv_bias=False,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=12, sr_ratios=1,
        **kwargs
    )
    # model.default_cfg = _cfg()
    return model
