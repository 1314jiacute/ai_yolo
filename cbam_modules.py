import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义CBAM模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# 注册函数 - 适配所有Ultralytics版本
def register_cbam_to_ultralytics():
    """将CBAM模块注册到Ultralytics的模块系统中"""
    try:
        import ultralytics.nn.modules as ul_modules
        
        # 方式1：直接添加到ultralytics的modules模块
        ul_modules.CBAM = CBAM
        ul_modules.ChannelAttention = ChannelAttention
        ul_modules.SpatialAttention = SpatialAttention
        
        # 方式2：添加到__all__，确保能被导入
        if hasattr(ul_modules, '__all__'):
            if 'CBAM' not in ul_modules.__all__:
                ul_modules.__all__.append('CBAM')
        
        print("✅ CBAM模块已注册到ultralytics.nn.modules")
        
    except Exception as e:
        print(f"⚠️ CBAM模块注册警告：{e}")
    
    # 方式3：修改ultralytics的parse_model函数（兼容版本）
    try:
        import ultralytics.nn.tasks as ul_tasks
        original_parse_model = ul_tasks.parse_model
        
        # 兼容不同版本的模块导入
        def safe_import_modules():
            modules_dict = {}
            try:
                from ultralytics.nn.modules import Conv, GhostConv, Bottleneck, GhostBottleneck
                modules_dict.update({
                    'Conv': Conv,
                    'GhostConv': GhostConv,
                    'Bottleneck': Bottleneck,
                    'GhostBottleneck': GhostBottleneck
                })
            except ImportError:
                pass
            
            try:
                from ultralytics.nn.modules import SPP, SPPF, DWConv
                modules_dict.update({
                    'SPP': SPP,
                    'SPPF': SPPF,
                    'DWConv': DWConv
                })
            except ImportError:
                pass
            
            try:
                from ultralytics.nn.modules import ConvTranspose, BottleneckCSP
                modules_dict.update({
                    'ConvTranspose': ConvTranspose,
                    'BottleneckCSP': BottleneckCSP
                })
            except ImportError:
                pass
            
            try:
                from ultralytics.nn.modules import C1, C2, C2f, C3, C3TR, C3SPP, C3Ghost
                modules_dict.update({
                    'C1': C1,
                    'C2': C2,
                    'C2f': C2f,
                    'C3': C3,
                    'C3TR': C3TR,
                    'C3SPP': C3SPP,
                    'C3Ghost': C3Ghost
                })
            except ImportError:
                pass
            
            try:
                from ultralytics.nn.modules import Focus, Stem
                modules_dict.update({
                    'Focus': Focus,
                    'Stem': Stem
                })
            except ImportError:
                pass
            
            try:
                from ultralytics.nn.modules import ContextModule, Proto, Detect, Segment, Pose, Classify, Concat
                modules_dict.update({
                    'ContextModule': ContextModule,
                    'Proto': Proto,
                    'Detect': Detect,
                    'Segment': Segment,
                    'Pose': Pose,
                    'Classify': Classify,
                    'Concat': Concat
                })
            except ImportError:
                pass
            
            # 添加CBAM模块
            modules_dict['CBAM'] = CBAM
            
            return modules_dict
        
        def custom_parse_model(d, ch, verbose=True):
            """自定义parse_model，添加CBAM支持（兼容所有版本）"""
            import copy
            import math
            import torch.nn as nn
            
            # 安全导入模块
            modules = safe_import_modules()
            
            if verbose:
                print(f"✅ 已加载{len(modules)}个模块，包含CBAM")
            
            d = copy.deepcopy(d)
            ch = copy.deepcopy(ch)
            
            nc, gd, gw = d['nc'], d['depth_multiple'], d['width_multiple']
            na = (nc + 15) // 16  # attach index
            no = na * (nc + 4) if 'segment' in d.get('task', '') else na * nc  # output channels
            layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
            
            # 辅助函数
            def make_divisible(x, divisor):
                if isinstance(divisor, torch.Tensor):
                    divisor = int(divisor.max())
                return math.ceil(x / divisor) * divisor
            
            for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
                # 获取模块类
                if isinstance(m, str):
                    if m not in modules:
                        raise KeyError(f"Module '{m}' not found in modules dictionary")
                    m = modules[m]
                
                # 处理参数
                for j, a in enumerate(args):
                    if isinstance(a, str):
                        args[j] = locals().get(a, a)
                
                # 深度增益
                n = n_ = max(round(n * gd), 1) if n > 1 else n
                
                # 处理不同模块的参数
                if m.__name__ in {
                    'Conv', 'GhostConv', 'DWConv', 'ConvTranspose', 
                    'Bottleneck', 'GhostBottleneck', 'SPP', 'SPPF',
                    'Focus', 'Stem', 'C1', 'C2', 'C2f', 'C3', 'C3TR',
                    'C3SPP', 'C3Ghost', 'BottleneckCSP', 'ContextModule'
                }:
                    c1, c2 = ch[f], args[0]
                    if c2 != no:
                        c2 = make_divisible(c2 * gw, 8)
                    
                    args = [c1, c2, *args[1:]]
                    if m.__name__ in {'C2', 'C2f', 'C3', 'C3TR', 'C3Ghost', 'BottleneckCSP'}:
                        args.insert(2, n)
                        n = 1
                
                elif m is nn.BatchNorm2d:
                    args = [ch[f]]
                elif m.__name__ == 'Concat':
                    c2 = sum(ch[x] for x in f)
                elif m.__name__ in {'Detect', 'Segment', 'Pose', 'Classify'}:
                    args.append([ch[x] for x in f])
                    if m.__name__ == 'Segment':
                        args[2] = make_divisible(args[2] * gw, 8)
                elif m.__name__ == 'Proto':
                    args = [ch[f], d['proto']]
                else:
                    c2 = ch[f]
                
                # 创建模块
                if n > 1:
                    m = nn.Sequential(*(m(*args) for _ in range(n)))
                else:
                    m = m(*args) if len(args) > 0 else m()
                
                # 添加模块信息
                t = str(m)[8:-2].replace('__main__.', '')
                np = sum(x.numel() for x in m.parameters())
                m.i, m.f, m.type, m.np = i, f, t, np
                
                # 保存列表
                save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
                layers.append(m)
                
                # 更新通道数
                if i == 0:
                    ch = []
                ch.append(c2)
                
                if verbose:
                    print(f'{i:>3}{f:>10}{t:<40}{np:>10.0f}')
            
            return nn.Sequential(*layers), sorted(save)
        
        # 替换parse_model函数
        ul_tasks.parse_model = custom_parse_model
        print("✅ 已替换parse_model函数，添加CBAM支持")
        
        return original_parse_model
        
    except Exception as e:
        print(f"⚠️ 修改parse_model失败：{e}，但CBAM仍可通过其他方式注册")
    
    print("✅ CBAM模块注册完成！")

# 自动注册
register_cbam_to_ultralytics()