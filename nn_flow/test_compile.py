import torch
import torch.nn as nn
from utils.profile import print_module_summary

# def compile_wr
@custom_compile()
class Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(100, 100)
    def forward(self, x):
        return self.proj(x)

class Large(nn.Module):
    def __init__(self):
        super().__init__()
        self.sm1 = Small()
        self.sm2 = Small()
    def forward(self, x):
        return self.sm1(x) + self.sm2(x)

if __name__ == "__main__":
    model = Large()
    print("BEFORE")
    f1 = model(torch.randn(20, 100))

    print("PROFILING")
    print_module_summary(model, inputs=(torch.randn(20, 100),), max_nesting=4)
    print("AFT")
    f3 = model(torch.randn(20, 100))
    print(f3)