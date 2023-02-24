import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

print(x + y, x * y, x / y, x ** y)

x = torch.arange(4.)
print(x)
print(x[3])
print(len(x))
print(x.shape)

A = torch.arange(20.).reshape(5, 4)
print(A)
print(A.T)

B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B)
print(B == B.T)

X = torch.arange(24).reshape(2, 3, 4)
print(X)
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
print(A)
print(A + B)
print(A * B)
print(A.shape, A.sum())

A_sum_axis1 = A.sum(axis=1, keepdims=True)
print(A_sum_axis1, A_sum_axis1.shape)

print(A / A_sum_axis1)

print(A.shape, x.shape)
print(torch.mv(A, x))

B = torch.ones(4, 3)
print(torch.mm(A, B))
