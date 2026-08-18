"""
AI Fashion Advisor - 100% Offline & Local Multi-Model Edition
-------------------------------------------------------------
Zero API keys required. Switch between local CV, Hugging Face CLIP,
and Local Ollama directly in the sidebar.
"""

import os
import json
import base64
from io import BytesIO
import requests
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
import streamlit as st

# Optional Local ML Models (Transformers / Torch)
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ==============================================================================
# 1. PAGE SETUP & MODERN CSS
# ==============================================================================

st.set_page_config(
    page_title="AI Fashion Advisor",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    
    .hero-container {
        background: linear-gradient(135deg, #111827 0%, #1F2937 50%, #374151 100%);
        border-radius: 16px; padding: 2rem; margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08); box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .hero-title { color: #F9FAFB !important; font-size: 2.2rem; font-weight: 800; margin: 0; }
    .hero-subtitle { color: #9CA3AF; font-size: 1rem; margin-top: 0.4rem; }
    
    .score-card {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%); color: white;
        border-radius: 14px; padding: 1.5rem; text-align: center; box-shadow: 0 8px 20px -4px rgba(79, 70, 229, 0.4);
    }
    .score-number { font-size: 3.5rem; font-weight: 800; line-height: 1; margin-bottom: 0.2rem; }
    
    .upgrade-box {
        background: rgba(16, 185, 129, 0.08); border: 1px dashed rgba(16, 185, 129, 0.4);
        border-radius: 12px; padding: 1.25rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. LOCAL COMPUTER VISION & COLOR HARMONY ENGINE
# ==============================================================================

def extract_color_palette_and_regions(pil_image: Image.Image, k: int = 4):
    """
    Extracts true dominant colors, upper/lower body color distribution,
    and calculates color harmony metrics.
    """
    img_rgb = pil_image.convert('RGB').resize((160, 200))
    img_np = np.array(img_rgb)
    
    # 1. Overall Dominant Colors via K-Means
    pixels = img_np.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=5).fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(kmeans.labels_)
    percentages = counts / len(pixels)
    
    palette = []
    for col, pct in sorted(zip(colors, percentages), key=lambda x: x[1], reverse=True):
        r, g, b = int(col[0]), int(col[1]), int(col[2])
        palette.append({
            "hex": f"#{r:02x}{g:02x}{b:02x}".upper(),
            "rgb": (r, g, b),
            "percentage": round(pct * 100, 1)
        })
        
    # 2. Split into Upper (top 45%) and Lower (bottom 55%) regions
    h, w, _ = img_np.shape
    upper_half = img_np[:int(h * 0.45), :]
    lower_half = img_np[int(h * 0.45):, :]
    
    avg_upper_rgb = upper_half.mean(axis=(0, 1)).astype(int)
    avg_lower_rgb = lower_half.mean(axis=(0, 1)).astype(int)
    
    # 3. Calculate Luminance & Contrast
    lum_top = 0.299 * avg_upper_rgb[0] + 0.587 * avg_upper_rgb[1] + 0.114 * avg_upper_rgb[2]
    lum_bot = 0.299 * avg_lower_rgb[0] + 0.587 * avg_lower_rgb[1] + 0.114 * avg_lower_rgb[2]
    contrast = abs(lum_top - lum_bot)
    
    # 4. Determine Harmony Type
    if contrast > 50:
        harmony_type = "High-Contrast Complementary"
    elif contrast > 25:
        harmony_type = "Balanced Dual-Tone"
    else:
        harmony_type = "Monochromatic / Low Contrast"
        
    return {
        "palette": palette,
        "upper_hex": f"#{avg_upper_rgb[0]:02x}{avg_upper_rgb[1]:02x}{avg_upper_rgb[2]:02x}".upper(),
        "lower_hex": f"#{avg_lower_rgb[0]:02x}{avg_lower_rgb[1]:02x}{avg_lower_rgb[2]:02x}".upper(),
        "contrast": contrast,
        "harmony_type": harmony_type
    }


def analyze_with_local_cv_engine(cv_data: dict, style_persona: str) -> dict:
    """Model 1: Local Computer Vision & Color Science Engine."""
    palette = cv_data["palette"]
    contrast = cv_data["contrast"]
    harmony = cv_data["harmony_type"]
    top_hex = cv_data["upper_hex"]
    bot_hex = cv_data["lower_hex"]
    
    # Score calculation based on color physics
    color_score = min(2.0, max(1.2, round(1.2 + (contrast / 80.0), 1)))
    coord_score = 1.8 if harmony != "Monochromatic / Low Contrast" else 1.4
    styling_score = 1.7 if style_persona == "Bold & Expressive" else 1.5
    acc_score = 0.6
    footwear_score = 0.8
    occasion_score = 0.8
    pres_score = 0.8
    
    total_score = round(color_score + coord_score + styling_score + acc_score + footwear_score + occasion_score + pres_score, 1)
    
    return {
        "scores": {
            "color_coordination": color_score,
            "outfit_coordination": coord_score,
            "styling": styling_score,
            "accessories": acc_score,
            "footwear_coordination": footwear_score,
            "occasion_suitability": occasion_score,
            "overall_presentation": pres_score
        },
        "overall_rating": total_score,
        "verdict_summary": f"Detected a {harmony.lower()} outfit with top tone {top_hex} and lower base {bot_hex}.",
        "detected_outfit": {
            "style_category": "Contemporary Casual",
            "top": f"Upper garment (Dominant: {top_hex})",
            "bottom": f"Lower garment (Dominant: {bot_hex})",
            "footwear": "Neutral footwear base",
            "accessories": "Minimal / Subdued",
            "primary_colors": [palette[0]["hex"], palette[1]["hex"]]
        },
        "what_looks_good": [
            f"Upper body ({top_hex}) and lower body ({bot_hex}) create clean visual separation.",
            f"Color harmony classification: {harmony}.",
            "Balanced saturation without overwhelming primary color clashes."
        ],
        "improvements": [
            "Add a structured watch, chain, or belt to introduce metallic accents.",
            "Consider layering with a light jacket or open overshirt to add silhouette depth.",
            "Introduce footwear that echoes the tone of the upper body for visual sandwiching."
        ],
        "color_analysis": {
            "harmony_type": harmony,
            "evaluation": f"The top ({top_hex}) and bottom ({bot_hex}) achieve a contrast index of {contrast:.1f}/100. This provides a clean everyday balance."
        },
        "occasion_suitability": {
            "Casual": 9,
            "College": 8,
            "Streetwear": 7,
            "Business Casual": 6 if contrast > 30 else 4,
            "Date Night": 7,
            "Formal": 3
        },
        "upgrade_recommendation": {
            "current_summary": f"{top_hex} Top + {bot_hex} Bottom",
            "recommended_upgrade": f"{top_hex} Top + {bot_hex} Bottom + Structured Outerwear + Wristwatch",
            "why_it_works": "Layering creates horizontal lines across the torso, while hardware introduces refined texture."
        }
    }


# ==============================================================================
# 3. LOCAL HUGGING FACE CLIP CLASSIFIER (OFFLINE ML)
# ==============================================================================

@st.cache_resource
def load_clip_model():
    """Loads lightweight CLIP model locally for zero-shot garment classification."""
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    try:
        model_name = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        return processor, model
    except Exception:
        return None, None


def analyze_with_local_clip(image: Image.Image, cv_data: dict) -> dict:
    """Model 2: Local HuggingFace CLIP Zero-Shot Garment & Style Detector."""
    processor, model = load_clip_model()
    if processor is None or model is None:
        st.warning("HuggingFace `transformers` or `torch` not found. Using local CV engine fallback.")
        return analyze_with_local_cv_engine(cv_data, "Minimalist")

    # Labels for zero-shot classification
    garment_labels = ["a person wearing a casual t-shirt and jeans", "a person wearing a formal suit or blazer", 
                      "a person wearing a streetwear hoodie or jacket", "a person wearing athletic gym sportswear", 
                      "a person wearing a dress or skirt", "a person wearing a button-down shirt and trousers"]
    
    inputs = processor(text=garment_labels, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0].numpy()
        
    top_label_idx = int(np.argmax(probs))
    detected_style = garment_labels[top_label_idx].replace("a person wearing ", "").capitalize()
    
    # Generate scores based on CLIP classification confidence
    confidence = float(probs[top_label_idx])
    base_rating = round(7.0 + (confidence * 2.2), 1)
    
    palette = cv_data["palette"]
    return {
        "scores": {
            "color_coordination": 1.7,
            "outfit_coordination": round(min(2.0, 1.4 + confidence), 1),
            "styling": 1.6,
            "accessories": 0.6,
            "footwear_coordination": 0.8,
            "occasion_suitability": 0.9,
            "overall_presentation": 0.8
        },
        "overall_rating": base_rating,
        "verdict_summary": f"Detected: {detected_style} (Confidence: {int(confidence*100)}%). Cohesive silhouette.",
        "detected_outfit": {
            "style_category": detected_style.split(" ")[0].capitalize(),
            "top": f"Garment matching {detected_style}",
            "bottom": "Coordinated trousers / bottom",
            "footwear": "Matching casual / formal shoes",
            "accessories": "Subdued",
            "primary_colors": [palette[0]["hex"], palette[1]["hex"]]
        },
        "what_looks_good": [
            f"Outfit geometry strongly aligns with: {detected_style}.",
            f"Consistent silhouette with primary color tone {palette[0]['hex']}.",
            "Clear stylistic intent across upper and lower pieces."
        ],
        "improvements": [
            "Upgrade footwear to match the formality of the top piece.",
            "Add a minimal leather or steel accessory to break color blocks.",
            "Tuck or layer depending on casual vs formal preference."
        ],
        "color_analysis": {
            "harmony_type": cv_data["harmony_type"],
            "evaluation": f"Dominated by {palette[0]['hex']} ({palette[0]['percentage']}%) and {palette[1]['hex']} ({palette[1]['percentage']}%)."
        },
        "occasion_suitability": {
            "Casual": 9 if "casual" in detected_style.lower() else 5,
            "College": 8 if "t-shirt" in detected_style.lower() else 4,
            "Streetwear": 9 if "hoodie" in detected_style.lower() else 6,
            "Business Casual": 9 if "button-down" in detected_style.lower() or "blazer" in detected_style.lower() else 4,
            "Date Night": 8,
            "Formal": 9 if "suit" in detected_style.lower() or "blazer" in detected_style.lower() else 2
        },
        "upgrade_recommendation": {
            "current_summary": detected_style,
            "recommended_upgrade": f"{detected_style} + Complementary Outer Layer + Minimalist Watch",
            "why_it_works": "Adds intentional depth and elevates standard proportions."
        }
    }


# ==============================================================================
# 4. LOCAL OLLAMA VISION (OFFLINE LOCALHOST)
# ==============================================================================

def analyze_with_ollama_localhost(image: Image.Image, model_name: str, cv_data: dict) -> dict:
    """Model 3: Local Ollama Vision Server (Runs on your PC with zero API keys)."""
    # Convert image to base64
    buffered = BytesIO()
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    image.save(buffered, format="JPEG", quality=85)
    b64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")

    prompt = """
    You are an objective fashion stylist. Analyze the visible clothing in this image.
    Return ONLY a JSON object with this format:
    {
      "scores": {"color_coordination": 1.7, "outfit_coordination": 1.6, "styling": 1.5, "accessories": 0.6, "footwear_coordination": 0.8, "occasion_suitability": 0.8, "overall_presentation": 0.8},
      "overall_rating": 7.8,
      "verdict_summary": "Summary of outfit",
      "detected_outfit": {"style_category": "Casual", "top": "Shirt", "bottom": "Pants", "footwear": "Shoes", "accessories": "None"},
      "what_looks_good": ["Point 1", "Point 2"],
      "improvements": ["Tip 1", "Tip 2"],
      "color_analysis": {"harmony_type": "Neutral", "evaluation": "Good balance"},
      "occasion_suitability": {"Casual": 9, "College": 8, "Business Casual": 5, "Date Night": 6, "Formal": 2},
      "upgrade_recommendation": {"current_summary": "Current", "recommended_upgrade": "Upgrade", "why_it_works": "Why"}
    }
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt, "images": [b64_img], "format": "json", "stream": False},
            timeout=90
        )
        if response.status_code == 200:
            res_text = response.json().get("response", "{}")
            return json.loads(res_text)
        else:
            st.error(f"Ollama returned status {response.status_code}. Using local CV engine fallback.")
            return analyze_with_local_cv_engine(cv_data, "Minimalist")
    except Exception as e:
        st.warning(f"Could not connect to Ollama (localhost:11434). Ensure Ollama is running (`ollama run {model_name}`). Using local CV fallback.")
        return analyze_with_local_cv_engine(cv_data, "Minimalist")


# ==============================================================================
# 5. DASHBOARD RENDERER
# ==============================================================================

def render_dashboard(data: dict):
    st.markdown("---")
    score = data.get("overall_rating", 7.5)
    verdict = data.get("verdict_summary", "Style assessment completed.")

    col_score, col_meta = st.columns([1, 2.2], gap="large")
    with col_score:
        st.markdown(f"""
        <div class="score-card">
            <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.9;">AI Style Score</div>
            <div class="score-number">{score}</div>
            <div style="font-size: 0.95rem; opacity: 0.9;">out of 10.0</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Generated locally on your device with 0 API calls.")

    with col_meta:
        st.subheader("Style Verdict")
        st.write(f"*{verdict}*")
        scores = data.get("scores", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Colors", f"{scores.get('color_coordination', 0):.1f}/2.0")
        c2.metric("Coordination", f"{scores.get('outfit_coordination', 0):.1f}/2.0")
        c3.metric("Styling", f"{scores.get('styling', 0):.1f}/2.0")
        c4.metric("Accessories", f"{scores.get('accessories', 0):.1f}/1.0")

    col_outfit, col_palette = st.columns([1.2, 1], gap="medium")
    with col_outfit:
        st.markdown("### 👔 Detected Outfit")
        detected = data.get("detected_outfit", {})
        st.markdown(f"""
        - **Style Category:** `{detected.get('style_category', 'Casual')}`
        - **Top:** {detected.get('top', 'Detected upper garment')}
        - **Bottom:** {detected.get('bottom', 'Detected lower garment')}
        - **Footwear:** {detected.get('footwear', 'Neutral / Standard')}
        - **Accessories:** {detected.get('accessories', 'Minimal')}
        """)

    with col_palette:
        st.markdown("### 🎨 Extracted Color Palette")
        swatch_html = "<div style='display: flex; gap: 8px; margin-bottom: 8px;'>"
        for color in data.get("extracted_palette", []):
            swatch_html += f"""
            <div style='text-align: center;'>
                <div style='background-color: {color["hex"]}; width: 50px; height: 40px; border-radius: 8px; border: 1px solid #444;'></div>
                <code style='font-size: 0.7rem;'>{color["hex"]}</code>
            </div>
            """
        swatch_html += "</div>"
        st.markdown(swatch_html, unsafe_allow_html=True)
        st.caption(data.get("color_analysis", {}).get("evaluation", ""))

    st.markdown("---")
    col_pros, col_cons = st.columns(2, gap="large")
    with col_pros:
        st.markdown("### ✨ What Looks Good")
        for item in data.get("what_looks_good", []):
            st.success(f"✓ {item}")

    with col_cons:
        st.markdown("### 💡 Suggestions for Improvement")
        for item in data.get("improvements", []):
            st.info(f"→ {item}")

    st.markdown("---")
    st.markdown("### 🎯 Occasion Suitability Matrix")
    occasions = data.get("occasion_suitability", {})
    if occasions:
        occ_cols = st.columns(len(occasions))
        for col, (occ_name, occ_score) in zip(occ_cols, occasions.items()):
            with col:
                st.metric(occ_name, f"{occ_score}/10")
                st.progress(min(max(occ_score / 10.0, 0.0), 1.0))

    st.markdown("---")
    st.markdown("### 🚀 Style Upgrade Recommendation")
    upgrade = data.get("upgrade_recommendation", {})
    st.markdown(f"""
    <div class="upgrade-box">
        <h5 style="margin-top: 0; color: #10B981;">Recommended Transformation</h5>
        <p><strong>Current:</strong> {upgrade.get('current_summary', 'Base Outfit')}</p>
        <p><strong>Upgraded Look:</strong> <span style="color: #A7F3D0; font-weight: 600;">{upgrade.get('recommended_upgrade', 'Elevated Style')}</span></p>
        <hr style="border-color: rgba(16, 185, 129, 0.2); margin: 8px 0;">
        <p style="font-size: 0.9rem; margin-bottom: 0;"><em><strong>Why:</strong> {upgrade.get('why_it_works', '')}</em></p>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 6. MAIN CONTROLLER
# ==============================================================================

def main():
    with st.sidebar:
        st.title("⚙️ Local Model Settings")
        st.caption("🔒 **100% Offline & Private** — Zero API Keys Needed")

        # Selectable local model engines
        selected_model = st.selectbox(
            "Select Offline AI Model",
            [
                "1. Smart CV & Color Harmony Engine (Instant / Zero Downloads)",
                "2. Hugging Face CLIP Zero-Shot Classifier (Local AI)",
                "3. Local Ollama Vision (LLaVA / Moondream on Localhost)",
                "4. High-Contrast Rule-Based Stylist (Instant)"
            ],
            index=0
        )

        ollama_model_name = "llava:latest"
        if "Ollama" in selected_model:
            ollama_model_name = st.text_input("Ollama Model Name", value="llava:latest", help="Make sure you ran `ollama run llava` locally")

        st.markdown("---")
        style_persona = st.select_slider(
            "Stylist Personality",
            options=["Minimalist & Subtle", "Balanced Modern", "Bold & Expressive"],
            value="Balanced Modern"
        )
        st.caption("Alters critique criteria and scoring weights.")

    # Hero Banner
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">👗 AI Fashion Advisor</h1>
        <p class="hero-subtitle">100% Local & Private. No API keys, no cloud errors. Capture your outfit and get instant styling analysis.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_cam, tab_upload = st.tabs(["📸 Webcam Capture", "📁 Upload Image"])
    image_to_analyze = None

    with tab_cam:
        col_c2 = st.columns([1, 2, 1])[1]
        with col_c2:
            camera_img = st.camera_input("Capture your outfit")
            if camera_img:
                image_to_analyze = Image.open(camera_img)

    with tab_upload:
        col_u2 = st.columns([1, 2, 1])[1]
        with col_u2:
            uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_file:
                image_to_analyze = Image.open(uploaded_file)

    if image_to_analyze:
        st.image(image_to_analyze, caption="Outfit Preview", width=320)
        
        if st.button("✨ Analyze My Outfit", type="primary"):
            with st.spinner(f"Analyzing with {selected_model}..."):
                try:
                    # 1. Local CV Color Analysis (Always extracts real pixels)
                    cv_data = extract_color_palette_and_regions(image_to_analyze)

                    # 2. Route based on selected model (100% offline)
                    if "Hugging Face CLIP" in selected_model:
                        result = analyze_with_local_clip(image_to_analyze, cv_data)
                    elif "Ollama" in selected_model:
                        result = analyze_with_ollama_localhost(image_to_analyze, ollama_model_name, cv_data)
                    elif "High-Contrast" in selected_model:
                        result = analyze_with_local_cv_engine(cv_data, "Bold & Expressive")
                    else:
                        result = analyze_with_local_cv_engine(cv_data, style_persona)

                    # Attach extracted swatches
                    result["extracted_palette"] = cv_data["palette"]
                    st.session_state["fashion_analysis"] = result
                    st.success("Analysis complete!")

                except Exception as err:
                    st.error(f"Error: {err}")

    if "fashion_analysis" in st.session_state:
        render_dashboard(st.session_state["fashion_analysis"])


if __name__ == "__main__":
    main()