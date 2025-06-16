if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import test_mode
import dezero.functions as F

x = np.ones(5)
print(x)

# When training
"""
일반적인 dropout
- dropout은 훈련 시에만 적용, 추론 시에는 모든 뉴런 다 씀
- 매 배치마다 랜덤으로 뉴런 삭제
- 테스트 시엔 뉴런이 더 많이 활성화되므로 출력이 훈련 때보다 커짐 -> 테스트 결과에 (1 - dropout rate)을 곱해서 줄임

역 dropout
- 학습할 때 dropout 하고 남은 뉴런에 미리 (1 - dropout rate)로 나눠줌
- 테스트 시엔 아무런 동작 X
-> 추론 속도가 살짝 개선됨
"""
y = F.dropout(x)
print(y)

# When testing (predicting)
with test_mode():
    y = F.dropout(x)
    print(y)