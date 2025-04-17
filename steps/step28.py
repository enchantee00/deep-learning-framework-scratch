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


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y


x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
lr = 0.001
iters = 50000

for i in range(iters):
    print(x0, x1)

    y = rosenbrock(x0, x1)
    """
    Variable 인스턴스 반복해서 사용 -> 미분값이 누적 -> cleargrad()로 누적된 미분값 초기화
    """
    x0.cleargrad()
    x1.cleargrad()
    y.backward()
    
    """
    경사하강법 -> 변수의 값을 미분값의 반대 방향으로 조금씩 움직임
    """
    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad