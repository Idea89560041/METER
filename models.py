import torch as torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import math
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

c = 6 # c is the number of distortion type

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'rest_small': _cfg(),
}

class AQPNet(nn.Module):
    def __init__(self, lda_out_channels, aqp_in_channels, target_in_size, target_fc1_size, target_fc2_size, target_fc3_size, target_fc4_size, feature_size):
        super(AQPNet, self).__init__()

        self.aqpInChn = aqp_in_channels
        self.target_in_size = target_in_size
        self.f1 = target_fc1_size
        self.f2 = target_fc2_size
        self.f3 = target_fc3_size
        self.f4 = target_fc4_size
        self.feature_size = feature_size

        self.res = rest_base(lda_out_channels, target_in_size, pretrained=True)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Conv layers for resnet output features
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            # nn.Conv2d(1024, 512, 1, padding=(0, 0)),
            # nn.ReLU(inplace=True),
            nn.Conv2d(256, self.aqpInChn, 1, padding=(0, 0)), #112
            nn.ReLU(inplace=True)
        )

        # Hyper network part, conv for generating target fc weights, fc for generating target fc biases
        self.fc1w_conv = nn.Conv2d(self.aqpInChn, int(self.target_in_size * self.f1 / feature_size ** 2), 3,
                                   padding=(1, 1))
        self.fc1b_fc = nn.Linear(self.aqpInChn, self.f1)

        self.fc2w_conv = nn.Conv2d(self.aqpInChn, int(self.f1 * self.f2 / feature_size ** 2), 3, padding=(1, 1))
        self.fc2b_fc = nn.Linear(self.aqpInChn, self.f2)

        self.fc3w_conv = nn.Conv2d(self.aqpInChn, int(self.f2 * self.f3 / feature_size ** 2), 3, padding=(1, 1))
        self.fc3b_fc = nn.Linear(self.aqpInChn, self.f3)

        self.fc4w_conv = nn.Conv2d(self.aqpInChn, int(self.f3 * self.f4 / feature_size ** 2), 3, padding=(1, 1))
        self.fc4b_fc = nn.Linear(self.aqpInChn, self.f4)

        self.fc5w_fc = nn.Linear(self.aqpInChn, self.f4*c)
        self.fc5b_fc = nn.Linear(self.aqpInChn, c)

        # initialize
        for i, m_name in enumerate(self._modules):
            if i > 2:
                nn.init.kaiming_normal_(self._modules[m_name].weight.data)

    def forward(self, img):
        feature_size = self.feature_size
        res_out = self.res(img)
        target_in_vec = res_out['target_in_vec'].view(-1, self.target_in_size, 1, 1) #224*1*1

        # input features for aqpnet
        aqp_in_feat = self.conv1(res_out['aqp_in_feat']).view(-1, self.aqpInChn, feature_size, feature_size) #112*7*7
        # generating AFC weights & biases
        target_fc1w = self.fc1w_conv(aqp_in_feat).contiguous().view(-1, self.f1, self.target_in_size, 1, 1)
        target_fc1b = self.fc1b_fc(self.pool(aqp_in_feat).squeeze()).view(-1, self.f1)

        target_fc2w = self.fc2w_conv(aqp_in_feat).contiguous().view(-1, self.f2, self.f1, 1, 1)
        target_fc2b = self.fc2b_fc(self.pool(aqp_in_feat).squeeze()).view(-1, self.f2)

        target_fc3w = self.fc3w_conv(aqp_in_feat).contiguous().view(-1, self.f3, self.f2, 1, 1)
        target_fc3b = self.fc3b_fc(self.pool(aqp_in_feat).squeeze()).view(-1, self.f3)

        target_fc4w = self.fc4w_conv(aqp_in_feat).contiguous().view(-1, self.f4, self.f3, 1, 1)
        target_fc4b = self.fc4b_fc(self.pool(aqp_in_feat).squeeze()).view(-1, self.f4)

        target_fc5w = self.fc5w_fc(self.pool(aqp_in_feat).squeeze()).view(-1, c, self.f4, 1, 1)
        target_fc5b = self.fc5b_fc(self.pool(aqp_in_feat).squeeze()).view(-1, c)

        out = {}
        out['target_in_vec'] = target_in_vec
        out['target_fc1w'] = target_fc1w
        out['target_fc1b'] = target_fc1b
        out['target_fc2w'] = target_fc2w
        out['target_fc2b'] = target_fc2b
        out['target_fc3w'] = target_fc3w
        out['target_fc3b'] = target_fc3b
        out['target_fc4w'] = target_fc4w
        out['target_fc4b'] = target_fc4b
        out['target_fc5w'] = target_fc5w
        out['target_fc5b'] = target_fc5b

        return out

class DTINet(nn.Module):
    """
    Distortion Type Identification Network for distortion type classification.
    """

    def __init__(self, n_distortions, n1_nodes, n2_nodes, n3_nodes):
        super(DTINet, self).__init__()
        self.fc1 = nn.Linear(n1_nodes, n2_nodes)
        self.fc2 = nn.Linear(n2_nodes, n3_nodes)
        self.fc3 = nn.Linear(n3_nodes, n_distortions)
        self.dropout = nn.Dropout()
        nn.init.kaiming_normal_(self.fc1.weight.data)
        nn.init.kaiming_normal_(self.fc2.weight.data)
        nn.init.kaiming_normal_(self.fc3.weight.data)

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))  #
        x = x.squeeze(3).squeeze(2)
        x1 = F.relu(self.fc1(x))
        x1 = self.dropout(x1)
        x2 = F.relu(self.fc2(x1))
        x2 = self.dropout(x2)
        x3 = self.fc3(x2)
        p = F.softmax(x3, dim=1)

        return p

class AFCNet(nn.Module):
    """
    AFCNet for quality prediction.
    """
    def __init__(self, paras):
        super(AFCNet, self).__init__()
        self.l1 = nn.Sequential(
            AFC(paras['target_fc1w'], paras['target_fc1b']),
            nn.Sigmoid(),
        )
        self.l2 = nn.Sequential(
            AFC(paras['target_fc2w'], paras['target_fc2b']),
            nn.Sigmoid(),
        )

        self.l3 = nn.Sequential(
            AFC(paras['target_fc3w'], paras['target_fc3b']),
            nn.Sigmoid(),
        )

        self.l4 = nn.Sequential(
            AFC(paras['target_fc4w'], paras['target_fc4b']),
            nn.Sigmoid(),
            AFC(paras['target_fc5w'], paras['target_fc5b']),
        )

    def forward(self, x):
        q = self.l1(x)
        q = F.dropout(q)
        q = self.l2(q)
        q = F.dropout(q)
        q = self.l3(q)
        q = self.l4(q).squeeze()
        return q

class AFC(nn.Module):

    def __init__(self, weight, bias):
        super(AFC, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input_):

        input_re = input_.view(-1, input_.shape[0] * input_.shape[1], input_.shape[2], input_.shape[3])
        weight_re = self.weight.view(self.weight.shape[0] * self.weight.shape[1], self.weight.shape[2], self.weight.shape[3], self.weight.shape[4])
        bias_re = self.bias.view(self.bias.shape[0] * self.bias.shape[1])
        out = F.conv2d(input=input_re, weight=weight_re, bias=bias_re, groups=self.weight.shape[0])

        return out.view(input_.shape[0], self.weight.shape[1], input_.shape[2], input_.shape[3])


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1,
                 apply_transform=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio+1, stride=sr_ratio, padding=sr_ratio // 2, groups=dim)
            self.sr_norm = nn.LayerNorm(dim)

        self.apply_transform = apply_transform and num_heads > 1
        if self.apply_transform:
            self.transform_conv = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1)
            self.transform_norm = nn.InstanceNorm2d(self.num_heads)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.sr_norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.apply_transform:
            attn = self.transform_conv(attn)
            attn = attn.softmax(dim=-1)
            attn = self.transform_norm(attn)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.2, act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, apply_transform=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, apply_transform=apply_transform)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PA(nn.Module):
    def __init__(self, dim): #64
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))

class GL(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gl_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        return x + self.gl_conv(x)

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding"""
    def __init__(self, patch_size=16, in_ch=3, out_ch=768, with_pos=False):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=patch_size+1, stride=patch_size, padding=patch_size // 2)
        self.norm = nn.BatchNorm2d(out_ch)

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)
        x = self.norm(x)
        if self.with_pos:
            x = self.pos(x)
        x = x.flatten(2).transpose(1, 2)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)

class BasicStem(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, with_pos=False):
        super(BasicStem, self).__init__()
        hidden_ch = out_ch // 2
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(hidden_ch)
        self.conv3 = nn.Conv2d(hidden_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)

        self.act = nn.ReLU(inplace=True)
        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch) #64

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv3(x)
        if self.with_pos:
            x = self.pos(x)
        return x

class Stem(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, with_pos=False):
        super(Stem, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.max_pool(x)

        if self.with_pos:
            x = self.pos(x)
        return x

class ResT(nn.Module):
    def __init__(self, lda_out_channels, in_chn, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
                 norm_layer=nn.LayerNorm, apply_transform=False):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.apply_transform = apply_transform
        self.stem = BasicStem(in_ch=in_chans, out_ch=embed_dims[0], with_pos=True)

        self.patch_embed_2 = PatchEmbed(patch_size=2, in_ch=embed_dims[0], out_ch=embed_dims[1], with_pos=True)
        self.patch_embed_3 = PatchEmbed(patch_size=2, in_ch=embed_dims[1], out_ch=embed_dims[2], with_pos=True)
        self.patch_embed_4 = PatchEmbed(patch_size=2, in_ch=embed_dims[2], out_ch=embed_dims[3], with_pos=True)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.stage1 = nn.ModuleList([
            Block(embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur+i], norm_layer=norm_layer, sr_ratio=sr_ratios[0], apply_transform=apply_transform)
            for i in range(self.depths[0])])

        cur += depths[0]
        self.stage2 = nn.ModuleList([
            Block(embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur+i], norm_layer=norm_layer, sr_ratio=sr_ratios[1], apply_transform=apply_transform)
            for i in range(self.depths[1])])

        cur += depths[1]
        self.stage3 = nn.ModuleList([
            Block(embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur+i], norm_layer=norm_layer, sr_ratio=sr_ratios[2], apply_transform=apply_transform)
            for i in range(self.depths[2])])

        cur += depths[2]
        self.stage4 = nn.ModuleList([
            Block(embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur+i], norm_layer=norm_layer, sr_ratio=sr_ratios[3], apply_transform=apply_transform)
            for i in range(self.depths[3])])

        self.norm = norm_layer(embed_dims[3])

        # local distortion aware module
        self.lda1_pool = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda1_fc = nn.Linear(16 * 64, lda_out_channels)

        self.lda2_pool = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda2_fc = nn.Linear(32 * 16, lda_out_channels)

        self.lda3_pool = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda3_fc = nn.Linear(64 * 4, lda_out_channels)

        self.lda4_pool = nn.AvgPool2d(7, stride=7)
        self.lda4_fc = nn.Linear(512, in_chn - lda_out_channels * 3)

        # ResNet初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # initialize
        nn.init.kaiming_normal_(self.lda1_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda2_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda3_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda1_fc.weight.data)
        nn.init.kaiming_normal_(self.lda2_fc.weight.data)
        nn.init.kaiming_normal_(self.lda3_fc.weight.data)
        nn.init.kaiming_normal_(self.lda4_fc.weight.data)

        # init weights
        self.apply(self._init_weights)

    # ResT
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        B, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)

        # stage 1
        for blk in self.stage1:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        lda_1 = self.lda1_fc(self.lda1_pool(x).view(x.size(0), -1))

        # stage 2
        x, (H, W) = self.patch_embed_2(x)
        for blk in self.stage2:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        lda_2 = self.lda2_fc(self.lda2_pool(x).view(x.size(0), -1))

        # stage 3
        x, (H, W) = self.patch_embed_3(x)
        for blk in self.stage3:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        lda_3 = self.lda3_fc(self.lda3_pool(x).view(x.size(0), -1))

        # stage 4
        x, (H, W) = self.patch_embed_4(x)
        for blk in self.stage4:
            x = blk(x, H, W)
        x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        lda_4 = self.lda4_fc(self.lda4_pool(x).view(x.size(0), -1))
        vec = torch.cat((lda_1, lda_2, lda_3, lda_4), 1)
        out = {}
        out['aqp_in_feat'] = x
        out['target_in_vec'] = vec

        return out

def rest_base(lda_out_channels, in_chn, pretrained=True, **kwargs):

    model = ResT(lda_out_channels, in_chn, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                 depths=[2, 2, 6, 2], sr_ratios=[8, 4, 2, 1], apply_transform=True, **kwargs)
    if pretrained:
        save_model = torch.load('./pretrained/rest_small.pth', map_location='cpu')
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    return model



