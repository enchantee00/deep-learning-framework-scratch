if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    """
    2차 미분을 위해 역전파 계산에 대해서도 계산 그래프를 만들게 함
    """
    y.backward(create_graph=True)

    gx = x.grad
    x.cleargrad()
    gx.backward()
    gx2 = x.grad

    """
    뉴턴 방법
    - 정밀하게 최적점 찾기 가능
    - but, 계산 비용 크고 메모리 사용량 많아서 실제론 잘 안쓰임
    """
    x.data -= gx.data / gx2.data