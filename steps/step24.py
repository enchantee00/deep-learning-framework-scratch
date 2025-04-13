import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable


def sphere(x, y):
    z = x ** 2 + y ** 2
    return z


def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z


def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z


x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = goldstein(x, y)  # sphere(x, y) / matyas(x, y)
z.backward()
print(x.grad, y.grad)

"""
딥러닝 프레임워크 동작방식
1. Define-and-Run
    계산 그래프를 먼저 정의하고 데이터 흘려보내기(2 step)
        - 사용성이 좋으며 디버깅이 쉬움
        - 도메인 특화 언어(프레임워크 특화 언어)를 배우지 않고도 파이썬으로 계산 그래프 정의 가능
        - 성능이 낮을 수 있음

2. Define-by-Run
    데이터를 흘려보내면서 계산 그래프를 정의(1 step)
        - 계산 그래프를 먼저 정의할 때 도메인 특화 언어로 정의해야 함 -> 사용성 낮음
        - 데이터를 흘려보내기 전에 계산 그래프를 정의하기 때문에 계산 그래프를 원하는 대로 최적화 가능 -> 계산 효율을 높여 성능을 올림
        - 분산 학습을 해야 하는 경우 유리(계산 그래프 자체를 분할하여 여러 컴퓨터로 분배해야 하는 경우 <- 계산 그래프가 이미 구축되어 있어야 함)
"""

torch.set_float32_matmul_precision('high')

class BigModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.seq(x)

# 실험 설정
batch_sizes = [128, 256, 512, 1024, 2048, 4096]
repeat = 100
dynamic_times = []
static_times = []

for batch_size in batch_sizes:
    print(f"배치 크기: {batch_size}")

    x = torch.randn(batch_size, 1024).to("cuda")

    # 동적 그래프
    model1 = BigModel().to("cuda")
    for _ in range(5): model1(x)

    start = time.time()
    for _ in range(repeat):
        _ = model1(x)
    torch.cuda.synchronize()
    dynamic_times.append(time.time() - start)

    # 정적 그래프
    model2 = torch.compile(BigModel().to("cuda"))
    for _ in range(5): model2(x)

    start = time.time()
    for _ in range(repeat):
        _ = model2(x)
    torch.cuda.synchronize()
    static_times.append(time.time() - start)

# 시각화
plt.plot(batch_sizes, dynamic_times, marker='o', label='Dynamic')
plt.plot(batch_sizes, static_times, marker='s', label='Static (torch.compile)')
plt.xlabel('Batch Size')
plt.ylabel("Execution Time (s)")
plt.title('Performance Comparison (torch.compile)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("torch_compile_comparison.png", dpi=300) 