import torch
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None

        def save_gradient(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_backward_hook(save_gradient)

    def forward_hook(self, module, input, output):
        self.activations = output

    def generate(self, input_tensor, target_class):
        self.model.zero_grad()
        out = self.model(input_tensor)
        out[0, target_class].backward()

        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()

        cam = np.maximum(cam.cpu().detach().numpy(), 0)
        cam = cv2.resize(cam, (224,224))
        cam = cam / cam.max()

        return cam
