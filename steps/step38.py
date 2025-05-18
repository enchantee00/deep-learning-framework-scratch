if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([[0, 1, 2], [3, 4, 5]]))
y = F.reshape(x, (6,))  # y = x.reshape(6)
y.backward(retain_grad=True)
print(y.grad)
print(x.grad)


x = np.random.randn(1,2,3) # 랜덤 값으로 채워진 1*2*3 크기의 행렬
print(x)
"""
shape 형태를 어떤 식으로 넣어도 괜찮게 하기 위함,,
"""
y = x.reshape((2,3))
print(y)
print(type(y))
y = x.reshape([2,3])
print(y)
print(type(y))
y = x.reshape(2,3)
print(y)
print(type(y))


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.transpose(x)  # y = x.T
print(y)
y.backward()
print(x.grad)

x = np.random.randn(1,2,3,4)
print(x)
y = x.transpose(1,0,3,2)
print(y)