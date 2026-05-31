import math
import torch

# LoRA classes were used from
# https://github.com/rasbt/LLMs-from-scratch/blob/main/appendix-E/01_main-chapter-code/appendix-E.ipynb
# class LoRALayer(torch.nn.Module):
#     def __init__(self, in_dim, out_dim, rank, alpha):
#         super().__init__()
#         #self.A = torch.nn.Parameter(torch.zeros(in_dim, rank))
#         self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
#         torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
#         self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
#         self.alpha = alpha
#         #self.d = torch.nn.Dropout(0.05)

#     def forward(self, x):
#         x = self.alpha * (x @ self.A @ self.B)
#         #x = self.d(x)
#         return x
class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        # 建议改为标准正态分布初始化
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        self.rank = rank # 用于缩放

    def forward(self, x):
        # 标准 LoRA 缩放公式
        scale = self.alpha / self.rank
        x = scale * (x @ self.A @ self.B)
        return x

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        x = self.linear(x) + self.lora(x)
        return x


