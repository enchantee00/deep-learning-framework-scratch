if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP


np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
max_iter = 10000
hidden_size = 10

model = MLP((hidden_size, 1))
# optimizer = optimizers.SGD(lr).setup(model)
optimizer = optimizers.MomentumSGD(lr).setup(model)


for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()
    if i % 1000 == 0:
        print(loss)

"""
MomentumSGD: SGD에 관성을 추가한 버전
- 이전 이동 방향을 누적시켜서 업데이트
- gradient의 방향이 일관되면 가속 -> 더 빠르게 수렴 가능
"""