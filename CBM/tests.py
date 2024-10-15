import torch

# tensor_2d = torch.zeros((4, 2))
#
# for i in range(tensor_2d.shape[0]):
#     for j in range(tensor_2d.shape[1]):
#         if i != 0:
#            tensor_2d[i,j] = 1
# new_list =tensor_2d.tolist()
# print(type(new_list))
# print(len((new_list)))
# for i in range(len(new_list)):
#     new_list[i] = torch.tensor(new_list[i])
#     print(type(new_list[i]))
matrix1 = torch.tensor([1, 0.2, 0.3, 0.4]).T
matrix2 = torch.tensor([0.5, 0.6, 0.7, 0.8])
matrix1 = torch.unsqueeze(matrix1,1)
#sum_matrix = matrix1 @ matrix2
# print(matrix2)
# print(matrix1)
#print(sum_matrix)
print(torch.nn.Sigmoid()(matrix1))