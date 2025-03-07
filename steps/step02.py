import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, in_data):
        raise NotImplementedError()

# Function은 데이터를 꺼내고 다시 상자에 담는 역할 -> 모든 함수가 공통적으로 제공하는 기능만 담아둠
class Square(Function):
    def forward(self, x):
        return x ** 2


x = Variable(np.array(10))
f = Square()
y = f(x)
print(type(y))
print(y.data)