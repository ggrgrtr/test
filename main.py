import torch as q
import numpy as np

# x = np.array([[1, 2, 3, 4],
#               [4, 3, 2, 1]])
# print(x)
# print()
#
# x=q.from_numpy(x)
# print(x)
#
# mask=(x>=3)
# print(x[mask])
#
# print(q.cuda.is_available()) # если можно использовать GPU -> set GPU
# device=q.device('cuda:0' if q.cuda.is_available() else 'cpu')
#
# print('next---------------------------------------------')


x=q.tensor([[1.,  2.,  3.,  4.],
     [5.,  6.,  7.,  8.],
     [9., 10., 11., 12.]],requires_grad=True)

device=q.device('cuda:0' if q.cuda.is_available() else 'cpu')
x=x.to(device)

print()
print(x)

F=10*(x**2).sum()
F.backward() # производная функции

print()
print('gradient[x]:\n',x.grad) # производная(градиент) тензора по функции

a=0.001


x.data=x-a*x.grad  # градиентный шаг
x.grad.zero_() # ОБНУЛЕНИЕ ГРАДИЕНТА

print()
print('градиентный спуск (1):\n',x)

# print()
# print("Enter the file name: ", end="")
# s = input()
#
# try:
#     file = open(s)
#     print(" --> Done")
# except:
#     print("\nno such file")

w = q.tensor([8., 8.],requires_grad=True)

optimizer = q.optim.SGD([w], lr=0.001)

def parabola(x):
  return 10*(x**2).sum()

def grad_step(F,tensor1):
  F_res = F(tensor1)
  F_res.backward()
  # tensor1.data -= 0.001*tensor1.grad   --- equals
  # tensor1.grad.zero_()
  optimizer.step()
  optimizer.zero_grad()

for i in range(500):
  grad_step(parabola, w)