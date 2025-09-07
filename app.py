import streamlit as st
import torch
from PIL import Image
from inference import main, get_model_inputs, test_inference, load_hf_model, PaliGemmaProcessor

st.title("Paligemma Financial Slide Analyzer")

uploaded_file = st.file_uploader("Upload an image or slide", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    prompt = st.text_input("Prompt", "Describe the financial trend in this chart")
    max_tokens = st.number_input("Max tokens to generate", value=100, step=10)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.8)
    top_p = st.slider("Top-p sampling", 0.1, 1.0, 0.9)
    do_sample = st.checkbox("Use sampling", value=False)

    if st.button("Analyze"):
        with st.spinner("Running Paligemma..."):
            # Call your inference pipeline
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_path = "/Users/hello./projects/paligemma-weights/paligemma-3b-pt-224"

            model, tokenizer = load_hf_model(model_path, device)
            model = model.to(device).eval()
            num_image_tokens = model.config.vision_config.num_image_tokens
            image_size = model.config.vision_config.image_size
            processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

            result = test_inference(
                model, processor, device, prompt, uploaded_file,
                max_tokens, temperature, top_p, do_sample
            )
            st.subheader("Paligemma Insight")
            st.write(result)
