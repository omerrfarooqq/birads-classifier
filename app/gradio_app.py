import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
from models.resnet50 import get_resnet50

model = get_resnet50()
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location="cpu"))
model.eval()

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

labels = ["birads1","birads2","birads3","birads4","birads5"]

def predict(img):
    img = tf(img).unsqueeze(0)
    with torch.no_grad():
        out = model(img)
    pred = out.argmax().item()
    return labels[pred]

def launch_app():
    interface = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs="text",
        title="BIRADS Classification",
        description="Upload a mammography image to get its BIRADS prediction."
    )
    interface.launch()
