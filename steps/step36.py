if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable

x = Variable(np.array(2.0))
y = x ** 2
y.backward(create_graph=True)
gx = x.grad
x.cleargrad()

z = gx ** 3 + y
z.backward()
print(x.grad)

"""
헤세 행렬: 벡터 x(input)의 두 원소에 대한 2차 미분 행렬
- lr가 아닌 헤세 행렬의 역행렬을 이용해서 갱신 진행 거리를 조정
"""