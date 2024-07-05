import torch
torch.manual_seed(0)

tensor1 = torch.randn(2, 182, 218, 182)
tensor2 = torch.randn(2, 182, 218, 182)
tensor3 = torch.randn(2, 182, 218, 182)

dropout_predictions = torch.empty((0,tensor1.size(0),tensor1.size(1),tensor1.size(2),tensor1.size(3)))
list_of_tensors = [tensor1, tensor2, tensor3]
for tensor in list_of_tensors:
    dropout_predictions=torch.cat((dropout_predictions, tensor.softmax(dim=1).unsqueeze(dim=0)), dim=0)
dropout_predictions_mean = dropout_predictions.mean(dim=0)

print(dropout_predictions_mean.shape)