# Add import path for the dezero directory.
"""
현재 파일이 위치한 디렉토리의 부모 디렉토리를 모듈 검색 경로에 추가한다.
-> root에서 dezero import 하듯이 바로 접근 가능
"""
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable


x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)