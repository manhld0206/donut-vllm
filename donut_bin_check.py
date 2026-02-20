import torch

checkpoint = torch.load("./pytorch_model.bin")

# print(checkpoint.keys())


# Assuming the weights are in a state_dict
for name, param in checkpoint.items():
    print(f"Layer: {name} | Size: {param.size()}")
    # You can also use visualization libraries like Matplotlib to plot histograms
    # or visualize convolutional filters as images
