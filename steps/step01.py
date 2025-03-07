import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data


data = np.array(1.0)
x = Variable(data)
print(x.data)

x.data = np.array([[[1.0,2.0,3.0], [4.0,5.0,6.0], [7.0, 8.0, 9.0]]])
print(x.data)

# ndim: 다차원 배열의 차원 수
print(x.data.ndim)


"""
차원 = 축 ex) 3차원 배열 = 축 3개

- 3차원 "배열": 차원 수 ex) [1,2,3] -> 1차원 배열
- 3차원 "벡터": 원소 개수 ex) [1,2,3] -> 3차원 벡터
"""