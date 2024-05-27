import torch

a = torch.tensor([1, 2, 3.], requires_grad=True)
print(a.grad)
out = a.sigmoid()
print(out)
c = a
print(c)
out.sum().backward()
print(a.grad)
print(c.grad)

'''
None
tensor([0.7311, 0.8808, 0.9526], grad_fn=<SigmoidBackward>)
tensor([0.7311, 0.8808, 0.9526])
tensor([0.1966, 0.1050, 0.0452])
'''