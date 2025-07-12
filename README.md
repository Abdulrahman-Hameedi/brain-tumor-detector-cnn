<------------------Brain Tumor Detector and Classifier------------------>

A deep learning model for brain tumor detection and classification using MRI scans. Upload and MRI image and with 86% accuracy and a trained Convolutional Neural Network, the tumor type (if any) will be detected and classified.


~ Features
- Simple upload of MRI scan for input
- Clean UI
- Model can classify tumors into:
    - Glioma 
    - Meningioma
    - Pituitary
    - No Tumor if no tumor is present

~ Tech Stack
- Python
- PyTorch
- Streamlit
- Torchvision
- PIL, OpenCV
- TensorFlow

~ How to Install
1. Clone the repository
    ```bash
    git clone https://github.com/Abdulrahman-Hameedi/brain-tumor-detector-cnn.git
    cd brain-tumor-detector-cnn

2. Install Dependencies
    ```bash
    pip install -r requirements.txt

3. Run the app
    ```bash
    streamlit run app.py
