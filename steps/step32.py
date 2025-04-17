"""
Mul class backward 메서드 비교
- x0 = self.inputs[0].data
  x1 = self.inputs[1].data
  return gy * x1, gy * x0

- x0, x1 = self.inputs
  return gy * x1, gy * x0

1) 수정 전에는 Variable 인스턴스 안에 있는 데이터를 꺼내야 했지만, 수정 후에는 Variable 인스턴스 그대로 사용
2) gy * x1과 같은 계산에 Variable 그대로 사용하고 연산자 오버로드도 돼 있음 -> Mul 클래스의 순전파 호출 -> 자동으로 계산 그래프 만들어짐



"""