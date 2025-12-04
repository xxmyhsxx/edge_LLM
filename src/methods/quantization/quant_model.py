# 该文件创建量化模块
import gc
import torch
import torch.nn as nn

from tqdm import tqdm
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from src.methods.quantization.quant_tensor import dequant_tensor,real_quantize_tensor,real_quantize_channel


class WeightQuantLinear(nn.Module):
    '''
    这是标准线性量化
    这是一个用于替换线性层的算子，必须传入scale、zero_point，目标bit
    TODO:
    1.实现线性算子的分组量化 2.实现fake quantization 3.尝试其他的量化
    '''
    def __init__(self, in_features, out_features,w_bit=8,bias=True,zp=False):
        super(WeightQuantLinear, self).__init__()
        if w_bit == 8:
            type = torch.int8
        else:
            raise TypeError('w_bit must be 8')
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight",
            torch.zeros(
                self.out_features,
                self.in_features,
                dtype=type,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.bfloat16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

        self.zero_point = None
        self.scale = None

    def to(self, *args, **kwargs):
        '''
        此处重写to方法
        '''
        super(WeightQuantLinear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.scale = self.scale.to(*args, **kwargs)
        self.zero_point = self.zero_point.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    def fused_quant_linear_forward(x, weight, scale, zero_point, bias):
        # weight_float = weight.to(torch.bfloat16)
        # JIT的优化：
        # 1. 内核融合：将多个操作合并为单一内核
        # 2. 内存优化：预分配输出缓冲区，消除临时张量
        # 3. 循环展开：优化内部循环
        # 4. 指令级优化：生成特定硬件的优化指令
        # weight_dequant =
        # if zero_point.numel() > 1:
        #     weight_dequant = (weight_float - zero_point) * scale
        # else:
        #     weight_dequant = weight_float * scale

        # output = torch.matmul(x, weight_dequant.transpose(0, 1))


        return torch.matmul(x, (weight * scale).transpose(0, 1))

    @torch.no_grad()
    @staticmethod
    def forward(self, input):
        '''
        注意:我们的权重是8bit,需要转换到BF16(和激活值相同即可)
        '''

        # self.scale = self.scale.to(self.weight.device).to(torch.bfloat16)
        # self.zero_point = self.zero_point.to(self.weight.device).to(torch.bfloat16)
        # output = self.fused_quant_linear_forward(input, self.weight, self.scale, self.zero_point, self.bias)
        # if self.bias is not None:
        #     # print(bias)
        #     output += self.bias.to(torch.bfloat16)

        return self.fused_quant_linear_forward(input, self.weight.to(torch.bfloat16), self.scale, self.zero_point, self.bias)


        # x_shape = x.shape
        # # 将输入 reshape 为 2D 矩阵 (M, K)
        # x_2d = x.reshape(-1, x_shape[-1])
        #
        # # 确保输入是 bf16
        # x_2d = x_2d.to(torch.bfloat16)
        # # x_2d = x_2d.contiguous()
        # # 调用 Triton 内核
        # print(x_2d.shape,self.weight.shape,self.scale.shape,self.zero_point.shape)
        #
        # output_2d = triton_dequant_matmul(x_2d, self.weight, self.scale, self.zero_point)
        #
        # # 添加 bias (如果存在)
        # if self.bias is not None:
        #     output_2d += self.bias
        #
        # # 将输出 reshape 回原始形状
        # output = output_2d.reshape(*x_shape[:-1], self.out_features)
        # return output





        # weight_float = self.weight.to(torch.bfloat16)  # 类型转换，但保持计算图融合
        #
        # if self.zero_point is not None:
        #     weight_dequant = (weight_float - self.zero_point) * self.scale
        # else:
        #     weight_dequant = weight_float * self.scale
        #
        # output = torch.matmul(input, weight_dequant.transpose(0, 1))
        # del weight_float
        # quant_output = torch.functional.F.linear(input, dequant_tensor(self.weight, self.scale, self.zero_point), self.bias)
        # # quant_output = self.output_quant(y)
        #
        # return output

    @staticmethod
    def from_float(module, w_bit=8, weight_quant="channel",zp=False):
        '''量化模型的初始化入口'''
        assert isinstance(module, torch.nn.Linear)
        new_module = WeightQuantLinear(
            module.in_features,
            module.out_features,
            w_bit,
            module.bias is not None,
            zp=zp
        )

        if weight_quant == "channel":
            new_weight,scale,zero_point  = real_quantize_channel(
                module.weight, n_bits=w_bit, zp=zp
            )  # use 8-bit integer for weight
        elif weight_quant == "tensor":
            new_weight,scale,zero_point = real_quantize_tensor(
                module.weight, n_bits=w_bit, zp=zp
            )

        else:
            raise ValueError(f"Invalid weight_quant")
        new_module.weight = new_weight
        new_module.zero_point = zero_point.cuda()
        new_module.scale = scale.cuda()
        if module.bias is not None:
            new_module.bias = module.bias
        # del module
        # gc.collect()
        return new_module

    def __repr__(self):
        return f"WQuantLinear({self.in_features}, {self.out_features}, bias={self.bias is not None})"

def quantize_layers(layers,weight_quant = "channel",w_bit=8,zp=False):
    '''
    目前支持对qwen3模型进行替换
    '''
    for i in tqdm(range(len(layers)),desc="Quantizing Layers"):
        m = layers[i]
        m.to("cuda")
        if isinstance(m,Qwen3DecoderLayer):
            m.mlp.gate_proj = WeightQuantLinear.from_float(
                m.mlp.gate_proj, w_bit,weight_quant=weight_quant,zp=zp
            )
            m.mlp.up_proj = WeightQuantLinear.from_float(
                m.mlp.up_proj, w_bit,weight_quant=weight_quant, zp=zp
            )
            m.mlp.down_proj = WeightQuantLinear.from_float(
                m.mlp.down_proj, w_bit, weight_quant=weight_quant, zp=zp
            )
            m.self_attn.q_proj = WeightQuantLinear.from_float(
                m.self_attn.q_proj, w_bit, weight_quant=weight_quant, zp=zp
            )
            m.self_attn.k_proj = WeightQuantLinear.from_float(
                m.self_attn.k_proj, w_bit, weight_quant=weight_quant, zp=zp
            )
            m.self_attn.v_proj = WeightQuantLinear.from_float(
                m.self_attn.v_proj, w_bit, weight_quant=weight_quant, zp=zp
            )
            m.self_attn.o_proj = WeightQuantLinear.from_float(
                m.self_attn.o_proj, w_bit, weight_quant=weight_quant, zp=zp
            )
            # gc.collect()
            # torch.cuda.empty_cache()
        else:
            raise ValueError(f"目前不支持该模型!!")
        m.to("cuda")
        gc.collect()
        torch.cuda.empty_cache()
    return layers



