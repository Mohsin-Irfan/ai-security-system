import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from train_model import SimpleCNN

# Page Config
st.set_page_config(
    page_title="AI Security System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ultra-modern CSS Styling with Animated Gradient
st.markdown("""
<style>
/* Fonts & body */
body {
    font-family: 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: #333;
    transition: background 1s ease;
}

/* Gradient animation */
@keyframes gradientAnimation {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Full-width header banner */
.header-banner {
    background: linear-gradient(270deg, #667eea, #764ba2, #ff416c, #ff4b2b);
    background-size: 400% 400%;
    animation: gradientAnimation 15s ease infinite;
    color: white;
    padding: 2rem 1rem;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    position: relative;
}
.header-banner h1 {
    font-size: 3rem;
    margin-bottom: 0.3rem;
    font-weight: 800;
}
.header-banner p {
    font-size: 1.5rem;
    font-weight: 400;
    margin: 0;
}

/* Floating Premium Button */
.floating-btn {
    position: absolute;
    top: 1rem;
    right: 1rem;
    background: linear-gradient(135deg,#ff416c,#ff4b2b);
    color: white;
    padding: 0.7rem 1.5rem;
    border-radius: 30px;
    font-weight: 600;
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
}
.floating-btn:hover {
    transform: translateY(-3px) scale(1.05);
    box-shadow: 0 8px 20px rgba(0,0,0,0.35);
}

/* Premium card with animated gradient */
.premium-card {
    background: linear-gradient(270deg, #667eea, #764ba2, #ff416c, #ff4b2b);
    background-size: 400% 400%;
    animation: gradientAnimation 20s ease infinite;
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.18);
    color: white;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.premium-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(0,0,0,0.25);
}

/* Upload zone */
.upload-zone {
    border: 3px dashed rgba(255,255,255,0.8);
    border-radius: 20px;
    padding: 4rem;
    text-align: center;
    background: rgba(255,255,255,0.1);
    transition: all 0.3s ease;
}
.upload-zone:hover {
    background: rgba(255,255,255,0.2);
    border-color: #ff416c;
}

/* Image container */
.image-container {
    border-radius: 25px;
    overflow: hidden;
    box-shadow: 0 12px 25px rgba(0,0,0,0.2);
    margin-bottom: 1rem;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.image-container:hover {
    transform: scale(1.05);
    box-shadow: 0 20px 40px rgba(0,0,0,0.3);
}

/* Prediction cards */
.prediction-card {
    background: linear-gradient(270deg, #667eea, #764ba2, #ff416c, #ff4b2b);
    background-size: 400% 400%;
    animation: gradientAnimation 15s ease infinite;
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    color: white;
    font-weight: 600;
    box-shadow: 0 8px 30px rgba(0,0,0,0.2);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.prediction-card:hover {
    transform: translateY(-5px) scale(1.02);
    box-shadow: 0 15px 35px rgba(0,0,0,0.25);
}
.prediction-label {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.confidence-text {
    font-size: 1.2rem;
}

/* Status badges */
.status-badge {
    padding: 1rem 2rem;
    border-radius: 35px;
    font-weight: 600;
    font-size: 1.2rem;
    text-align: center;
    margin: 0.5rem 0;
    transition: transform 0.3s ease, box-shadow 0.3s ease, background 1s ease;
}
.status-badge:hover {
    transform: scale(1.08);
    box-shadow: 0 10px 25px rgba(0,0,0,0.25);
}
.success-badge { background: linear-gradient(270deg,#00b09b,#96c93d,#00b09b); background-size: 400% 400%; animation: gradientAnimation 10s ease infinite; color:white; }
.danger-badge { background: linear-gradient(270deg,#ff416c,#ff4b2b,#ff416c); background-size: 400% 400%; animation: gradientAnimation 10s ease infinite; color:white; }
.info-badge { background: linear-gradient(270deg,#2193b0,#6dd5ed,#2193b0); background-size: 400% 400%; animation: gradientAnimation 10s ease infinite; color:white; }

/* Hide default Streamlit elements */
header, #MainMenu, footer {visibility: hidden;}
.stFileUploader > div > div {
    border: 2px dashed rgba(255,255,255,0.8) !important;
    background: rgba(255,255,255,0.05) !important;
    border-radius: 15px !important;
}

            
</style>
""", unsafe_allow_html=True)

# Floating Button
st.markdown('<div class="floating-btn">Premium AI Security</div>', unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model = SimpleCNN()
    model.load_state_dict(torch.load('cifar10_model.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()
classes = ['airplane','car','bird','cat','deer','dog','frog','horse','ship','truck']
transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

def fgsm_attack(image, epsilon=0.1):
    image = image.clone().detach().requires_grad_(True)
    output = model(image)
    loss = nn.functional.cross_entropy(output, torch.argmax(output,1))
    model.zero_grad()
    loss.backward()
    sign_data_grad = image.grad.sign()
    perturbed_image = torch.clamp(image + epsilon*sign_data_grad,0,1)
    return perturbed_image

def detect_adversarial(perturbed_image, original_image, threshold=0.02):
    noise = torch.abs(perturbed_image - original_image)
    avg_noise = torch.mean(noise).item()
    return avg_noise > threshold, avg_noise

# Header Banner
st.markdown("""
<div class="header-banner">
    <h1>AI Security System 🛡️</h1>
    <p>Adversarial Attack + Defense Detection</p>
</div>
""", unsafe_allow_html=True)

# Upload Section
st.markdown('<div class="premium-card"><div class="upload-zone"><h2>📤 Upload an Image</h2><p><b>See how AI reacts to adversarial noise!</b></p></div></div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an Image", type=['jpg','png','jpeg'], label_visibility="collapsed")
if uploaded_file is not None:
    st.markdown(f"**{uploaded_file.name}** - {uploaded_file.size/1024:.1f}KB")
else:
    st.markdown("- **Drag and drop file here**  \n- Limit 200MB • JPG, PNG, JPEG")

# Process Image
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    original_image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(original_image)
        probs = torch.softmax(output, dim=1)
        orig_prob, orig_class = torch.max(probs,1)
        orig_label = classes[orig_class.item()]
        orig_conf = orig_prob.item()

    attacked_image = fgsm_attack(original_image)
    with torch.no_grad():
        attacked_out = model(attacked_image)
        att_prob, att_class = torch.max(torch.softmax(attacked_out,1),1)
        att_label = classes[att_class.item()]
        att_conf = att_prob.item()

    is_adv, noise_level = detect_adversarial(attacked_image, original_image)

    def tensor_to_image(tensor):
        tensor = tensor.squeeze(0).detach().cpu()
        tensor = tensor*0.5 + 0.5
        np_img = tensor.permute(1,2,0).numpy()
        return (np_img*255).astype('uint8')

    orig_img = tensor_to_image(original_image)
    att_img = tensor_to_image(attacked_image)

    # Results Section
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Original Image")
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(orig_img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="prediction-card"><div class="prediction-label">Prediction: {orig_label}</div><div class="confidence-text">Confidence: {orig_conf:.2%}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown("### Attacked Image")
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(att_img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="prediction-card"><div class="prediction-label">Prediction: {att_label}</div><div class="confidence-text">Confidence: {att_conf:.2%}</div></div>', unsafe_allow_html=True)

    # Status Row with Animated Badges
    st.markdown("---")
    col3, col4, col5 = st.columns(3)
    with col3:
        st.markdown("### Attack Status")
        badge = "SUCCESS ✔" if orig_label!=att_label else "FAILED ❌"
        badge_class = "success-badge" if orig_label!=att_label else "danger-badge"
        st.markdown(f'<div class="status-badge {badge_class}">{badge}</div>', unsafe_allow_html=True)
    with col4:
        st.markdown("### Defense Status")
        badge = "DETECTED 🌟" if is_adv else "MISSED ⚠"
        badge_class = "success-badge" if is_adv else "danger-badge"
        st.markdown(f'<div class="status-badge {badge_class}">{badge}</div>', unsafe_allow_html=True)
    with col5:
        st.markdown("### Noise Level")
        st.markdown(f'<div class="status-badge info-badge">{noise_level:.4f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<div style='text-align:center;color:#fff;margin-top:3rem;'>AI Security System • Advanced Adversarial Protection</div>", unsafe_allow_html=True)
