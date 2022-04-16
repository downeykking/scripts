import torch

label = torch.LongTensor([1, 0, 2, 3, 5])
num_class = label.max().item() + 1

one_hot_label = torch.zeros((label.size(0), num_class))
index = label.view(-1, 1)
one_hot_label.scatter_(dim=1, index=index, value=1)

print(one_hot_label)
