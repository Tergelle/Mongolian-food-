import streamlit as st
from fastai.vision.all import *
import pathlib
from PIL import Image
import gdown

# Set the path for Windows compatibility
plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

# Google Drive file ID
file_id = "1Bmeyo8bTS-owVyUUZyeiHrM1adJ69FNU"
model_path = "Foods.pkl"

# Download the model if not exists
if not Path(model_path).exists():
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

# Load the model
learn_inf = load_learner(model_path)

# Streamlit app
st.title("Mongolian Food Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload an image of Buuz, Khuushuur, Tsuivan, or Niislel Salad", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Perform inference
    pred, pred_idx, probs = learn_inf.predict(image)

    # Display the prediction
    st.write(f"**Prediction:** {pred}")
    st.write(f"**Probability:** {probs[pred_idx]:.2f}")

    # Display probabilities for all classes
    st.write("**Probabilities for all classes:**")
    for class_name, prob in zip(learn_inf.dls.vocab, probs):
        st.write(f"{class_name}: {prob:.2f}")
