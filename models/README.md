# How to quantize existing pytorch CNN

1. Go through model architecture 
2. Replace successive nn.Conv2d, nn.BatchNorm2d[, nn.ReLU] with tinyquant.ConvBNnofold
3. Double check to make sure modules are added to model during init in exactly the order in which they are called
4. If last module in fwd pass is not an activation, add an nn.Identity() after the activation
5. Replace control flow for residuals (additive connections) with QuantizableResConnection, placing it right before the next activation
6. Replace everything not done by an nn.Module with an nn.Module, e.g. torch.nn.functional.relu => torch.nn.ReLU()
7. always use relu6 over relu when possible






