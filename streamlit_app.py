import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import google.generativeai as genai
import torch

# Configure the API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Load the LLaMA-Mesh model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Zhengyi/LLaMA-Mesh")
model = AutoModelForCausalLM.from_pretrained("Zhengyi/LLaMA-Mesh")

# Streamlit App UI
st.title("Ever AI - 3D CAD Model Generator")
st.write("Use generative AI to create 3D CAD models based on your prompt.")

# Prompt input field
prompt = st.text_input("Enter your prompt:", "Create a 3D model of a house.")

# Button to generate the CAD model
if st.button("Generate CAD Model"):
    try:
        # Generate response from the model
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Save the generated CAD content to a file
        cad_file_path = "generated_model.cad"
        with open(cad_file_path, "w") as f:
            f.write(response)
        
        # Display response in Streamlit
        st.write("CAD Model Generated:")
        st.code(response, language='plaintext')
        
        # Provide a download link for the CAD file
        st.download_button(
            label="Download CAD File",
            data=response,
            file_name="generated_model.cad",
            mime="text/plain"
        )
    except Exception as e:
        st.error(f"Error: {e}")
