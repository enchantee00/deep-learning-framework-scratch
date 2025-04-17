if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
# import dezero's simple_core explicitly
import dezero
if not dezero.is_simple_core:
    from dezero.core_simple import Variable
    from dezero.core_simple import setup_variable
    setup_variable()


def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


def gx2(x):
    return 12 * x ** 2 - 4


x = Variable(np.array(2.0))
iters = 10

"""
경사하강법: lr을 사용자가 정해야 함
뉴턴방법: lr을 함수의 2차 미분으로 표현하기 때문에 최적화 되어 있음
-> 뉴턴방법이 경사하강법보다 더 빠르게 수렴하지만, 2차 미분을 해야 한다는 단점이 존재함
"""
for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward()

    """
    .grad로 1차 미분까진 자동으로 구할 수 있지만(사칙연산으로 이루어져 있다면), 2차 미분부터는 직접 계산해서 하드코딩해야 함
    """
    x.data -= x.grad / gx2(x.data)