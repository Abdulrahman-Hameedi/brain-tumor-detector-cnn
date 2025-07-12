import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from model import CancerTumorCNN
from config import label_map


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_cnn_model():
    model = CancerTumorCNN().to(device)
    model.load_state_dict(torch.load("datasets/brain_tumor_cnn.pth", map_location=device))
    model.eval()
    return model

model = load_cnn_model()

def preprocess_image(img_pil):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(img_pil).unsqueeze(0)

def classify_image(img_pil, model):
    tensor = preprocess_image(img_pil).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs.data, 1)
    return label_map[predicted.item()]


st.title("Brain Tumor Detector and Classifier")

uploaded_file = st.file_uploader("Upload an MRI scan", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img_pil = Image.open(uploaded_file).convert("RGB")
    
    st.image(img_pil, caption="Uploaded MRI Scan", width=450)

    if st.button("Detect and Classify"):
        prediction = classify_image(img_pil, model)
        st.success(f"Predicted Tumor Type: **{prediction.upper()}**")
