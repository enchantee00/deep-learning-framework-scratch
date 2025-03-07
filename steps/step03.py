import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data

# 입출력을 Variable로 통일해주므로 여러 함수 연속 적용 가능
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2


class Exp(Function):
    def forward(self, x):
        return np.exp(x)


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)
print(y.data)

"""
하나의 계산이 아닌 계산 그래프로 표현 -> 각 변수에 대한 미분을 효율적으로 계산 가능
"""