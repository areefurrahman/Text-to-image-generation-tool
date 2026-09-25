import io
import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image

# ---------- Page config ----------
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="centered",
)

st.title("🎨 AI Text-to-Image Generator")
st.caption("Powered by Hugging Face Inference Providers")

# ---------- Sidebar: token & model ----------
st.sidebar.header("⚙️ Settings")

if "HF_TOKEN" in st.secrets:
    hf_token = st.secrets["HF_TOKEN"]
    st.sidebar.success("Token loaded from secrets ✅")
else:
    hf_token = st.sidebar.text_input(
        "Enter your Hugging Face token",
        type="password",
        help="Get one at huggingface.co/settings/tokens",
    )

model_choice = st.sidebar.selectbox(
    "Model",
    [
        "black-forest-labs/FLUX.1-schnell",
        "black-forest-labs/FLUX.1-dev",
        "stabilityai/stable-diffusion-xl-base-1.0",
    ],
)

# ---------- Main area: prompt & style ----------
prompt = st.text_area(
    "Describe the image you want to create",
    placeholder="A futuristic city skyline at sunset...",
    height=100,
)

style = st.selectbox(
    "Style preset (optional)",
    ["None", "Cinematic", "Anime", "Digital Art", "Photographic"],
)

style_suffixes = {
    "None": "",
    "Cinematic": ", cinematic lighting, dramatic composition, 8k",
    "Anime": ", anime style, vibrant colors, studio quality",
    "Digital Art": ", digital art, trending on artstation, highly detailed",
    "Photographic": ", photorealistic, shot on DSLR, sharp focus",
}

generate_btn = st.button("Generate Image 🎨", type="primary", use_container_width=True)

# ---------- Generation logic ----------
if generate_btn:
    if not hf_token:
        st.error("Please provide a Hugging Face token in the sidebar.")
    elif not prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        final_prompt = prompt.strip() + style_suffixes[style]

        with st.spinner("Generating your image... this can take 10-30 seconds"):
            try:
                client = InferenceClient(provider="auto", api_key=hf_token)
                image = client.text_to_image(final_prompt, model=model_choice)

                st.image(image, caption=final_prompt, use_container_width=True)

                # Prepare download
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                buf.seek(0)

                st.download_button(
                    label="⬇️ Download PNG",
                    data=buf,
                    file_name="generated_image.png",
                    mime="image/png",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"Generation failed: {e}")
                st.info("Try a different model from the sidebar, or wait a moment and retry.")
