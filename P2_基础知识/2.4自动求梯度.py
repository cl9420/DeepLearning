import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)
print(x.grad_fn)

y = x + 2
print(y)
print(y.grad_fn)
print(x.is_leaf, y.is_leaf)  # True False

z = y * y * 3
out = z.mean()
print(z, out)

a = torch.randn(2, 2)  # 缺失情况下默认 requires_grad = False
a = ((a * 3) / (a - 1))
print(a.requires_grad)  # False
a.requires_grad_(True)
print(a.requires_grad)  # True
b = (a * a).sum()
print(b.grad_fn)

out.backward()  # 等价于 out.backward(torch.tensor(1.))
print(x.grad)

x = torch.ones(1,requires_grad=True)

print(x.data)  # 还是一个tensor
print(x.data.requires_grad)  # 但是已经是独立于计算图之外

y = 2 * x
x.data *= 100  # 只改变了值，不会记录在计算图，所以不会影响梯度传播

y.backward()
print(x)  # 更改data的值也会影响tensor的值
print(x.grad)
