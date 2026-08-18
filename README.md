# 👗 AI Fashion Analyzer

> **AI-powered fashion analysis with 100% local processing — no API keys required.**

AI Fashion Analyzer is a Streamlit-based application that analyzes an outfit from a **webcam capture or uploaded image** and provides an AI-powered fashion assessment.

The application combines **Computer Vision, color analysis, Hugging Face CLIP, and local Ollama vision models** to detect outfit style, analyze color harmony, generate a fashion score, identify areas for improvement, and recommend style upgrades.

🔒 **Privacy-focused:** Your images can be processed locally without sending them to external cloud APIs.

---

## ✨ Features

* 📸 **Webcam Outfit Capture**
* 📁 **Upload JPG, JPEG, PNG, or WebP images**
* 🎨 **Automatic Color Palette Extraction**
* 🌈 **Color Harmony & Contrast Analysis**
* 🤖 **Hugging Face CLIP Zero-Shot Classification**
* 🧠 **Local Ollama Vision Models**
* ⚡ **Instant Computer Vision Analysis**
* ⭐ **AI Fashion Score out of 10**
* 👔 **Outfit & Garment Detection**
* 💡 **Personalized Style Suggestions**
* 🎯 **Occasion Suitability Scores**
* 🚀 **Style Upgrade Recommendations**
* 🔒 **100% Local / Private Processing**
* 🔑 **No API Keys Required**

---

## 🧠 How It Works

The application provides multiple analysis engines that can be selected from the sidebar.

### 1. 🎨 Smart CV & Color Harmony Engine

This is the lightweight analysis mode and requires **no additional AI model downloads**.

It uses:

* OpenCV
* NumPy
* PIL
* K-Means clustering

The application extracts dominant colors from the image and separates the image into upper and lower body regions.

It then calculates:

* Dominant colors
* RGB values
* HEX colors
* Color percentages
* Upper-body color
* Lower-body color
* Luminance
* Color contrast
* Color harmony type

The harmony is classified as:

* **High-Contrast Complementary**
* **Balanced Dual-Tone**
* **Monochromatic / Low Contrast**

The project uses K-Means clustering to extract the dominant color palette from the outfit.

---

### 2. 🤗 Hugging Face CLIP

The project can optionally use the local:

**OpenAI CLIP ViT-B/32**

model through Hugging Face Transformers.

CLIP performs zero-shot outfit classification using categories such as:

* Casual
* Formal
* Streetwear
* Athletic
* Dress / Skirt
* Button-down / Trousers

The detected category and model confidence are then used as part of the fashion analysis.

---

### 3. 🦙 Local Ollama Vision

For more advanced local vision analysis, the application can communicate with an Ollama server running on:

```text
localhost:11434
```

Models such as:

```text
llava:latest
```

can be used to analyze the uploaded outfit.

The model is instructed to return structured JSON containing:

* Fashion score
* Outfit category
* Clothing items
* Color analysis
* What looks good
* Improvements
* Occasion suitability
* Style upgrade recommendations

The Ollama implementation sends the image directly to the local Ollama endpoint, so an external API key is not required.

---

## 🏗️ Project Architecture

```text
                 ┌─────────────────────┐
                 │   User Image Input  │
                 │ Webcam / Upload     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Image Processing   │
                 │ PIL + NumPy + OpenCV│
                 └──────────┬──────────┘
                            │
                            ▼
              ┌────────────────────────────┐
              │ Color Palette Extraction   │
              │ K-Means + Color Harmony    │
              └─────────────┬──────────────┘
                            │
              ┌─────────────┼──────────────┐
              ▼             ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Local CV │  │   CLIP   │  │  Ollama  │
        │  Engine  │  │  Model   │  │  Vision  │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                ┌──────────────────────┐
                │ Fashion Analysis     │
                │ Score + Suggestions  │
                └──────────┬───────────┘
                           ▼
                ┌──────────────────────┐
                │ Streamlit Dashboard  │
                └──────────────────────┘
```

---

## 📊 Analysis Dashboard

After analyzing an outfit, the dashboard displays:

### ⭐ AI Style Score

A score out of **10** representing the overall fashion assessment.

### 👔 Detected Outfit

Displays:

* Style category
* Top
* Bottom
* Footwear
* Accessories

### 🎨 Extracted Color Palette

The application displays the dominant colors detected from the image along with their HEX values.

### ✨ What Looks Good

Highlights positive aspects of the outfit.

### 💡 Suggestions for Improvement

Provides recommendations for improving the overall appearance.

### 🎯 Occasion Suitability

The outfit can be evaluated for occasions such as:

| Occasion        | Score |
| --------------- | ----: |
| Casual          |   /10 |
| College         |   /10 |
| Streetwear      |   /10 |
| Business Casual |   /10 |
| Date Night      |   /10 |
| Formal          |   /10 |

### 🚀 Style Upgrade

The application provides a recommended upgraded version of the current outfit and explains why the upgrade works.

The dashboard implementation includes the score, detected outfit, color palette, improvements, occasion matrix, and upgrade recommendation sections.

---

## 🎛️ Available AI Modes

From the sidebar, you can choose between four modes:

```text
1. Smart CV & Color Harmony Engine
2. Hugging Face CLIP Zero-Shot Classifier
3. Local Ollama Vision
4. High-Contrast Rule-Based Stylist
```

You can also select the stylist personality:

```text
Minimalist & Subtle
Balanced Modern
Bold & Expressive
```

These settings are available directly from the Streamlit sidebar.

---

# 🛠️ Tech Stack

| Technology                | Purpose                        |
| ------------------------- | ------------------------------ |
| Python                    | Core programming language      |
| Streamlit                 | Web application interface      |
| OpenCV                    | Computer vision                |
| NumPy                     | Numerical processing           |
| Pillow                    | Image processing               |
| Scikit-learn              | K-Means color clustering       |
| PyTorch                   | Local ML inference             |
| Hugging Face Transformers | CLIP model                     |
| OpenAI CLIP               | Zero-shot image classification |
| Ollama                    | Local vision AI                |
| HTML/CSS                  | Dashboard styling              |

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/ai-fashion-analyzer.git
cd ai-fashion-analyzer
```

Replace `YOUR-USERNAME` with your GitHub username.

---

## 2. Create a Virtual Environment

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

Create a `requirements.txt` file containing:

```txt
streamlit
numpy
opencv-python
Pillow
scikit-learn
requests
torch
transformers
```

Then run:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start Streamlit with:

```bash
streamlit run app.py
```

Then open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

---

# 🤗 Optional: Enable Hugging Face CLIP

The CLIP model is loaded locally using:

```text
openai/clip-vit-base-patch32
```

The application can fall back to the local Computer Vision engine if Transformers or PyTorch is unavailable.

---

# 🦙 Optional: Enable Ollama

If you want to use local vision models through Ollama, install Ollama separately and make sure the Ollama server is running.

For example:

```bash
ollama run llava
```

Then select:

```text
Local Ollama Vision
```

from the application sidebar.

The default model is:

```text
llava:latest
```

You can also enter another locally available Ollama model through the application interface.

---

# 📸 How to Use

### Step 1

Launch the application:

```bash
streamlit run app.py
```

### Step 2

Choose an analysis model from the sidebar.

### Step 3

Either:

* 📸 Capture your outfit using the webcam
* 📁 Upload an existing outfit image

### Step 4

Click:

```text
✨ Analyze My Outfit
```

### Step 5

View your:

* Fashion score
* Detected outfit
* Color palette
* Color harmony
* Style analysis
* Improvements
* Occasion suitability
* Style upgrade recommendation

The application supports both webcam capture and image uploads.

---

# 🔐 Privacy

Privacy is one of the main goals of this project.

The application is designed around local processing:

```text
❌ No OpenAI API
❌ No Gemini API
❌ No external fashion API
❌ No API key required
✅ Local Computer Vision
✅ Local CLIP
✅ Local Ollama
```

When using the local CV engine, image analysis is performed directly on the device.

---

# 📁 Suggested Project Structure

```text
ai-fashion-analyzer/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── assets/
    └── screenshots/
```

If you later split the application into multiple modules, a structure such as this can be used:

```text
ai-fashion-analyzer/
│
├── app.py
├── engines/
│   ├── cv_engine.py
│   ├── clip_engine.py
│   └── ollama_engine.py
│
├── requirements.txt
├── README.md
└── assets/
```

---

# 🚀 Future Improvements

Possible future enhancements include:

* 👕 Individual clothing item detection
* 🧍 Body-shape analysis
* 🎨 Personal color-season analysis
* 👟 Dedicated footwear detection
* ⌚ Accessory recognition
* 🛍️ Outfit recommendation system
* 🌦️ Weather-based outfit suggestions
* 📍 Occasion-aware recommendations
* 👔 Wardrobe management
* 🔄 Outfit combination generator
* 📊 Fashion history and score tracking
* 🧠 Fine-tuned fashion classification model
* 📱 Mobile-friendly interface

---

# ⚠️ Limitations

The accuracy of the analysis depends on:

* Image quality
* Lighting
* Camera angle
* Visibility of clothing
* Model selected
* Local model capabilities

The CV engine primarily focuses on **color and visual harmony**, while CLIP and Ollama provide more advanced semantic interpretation.

The fashion score should therefore be considered an **AI-generated styling assessment rather than an objective measurement**.

---

# 🤝 Contributing

Contributions are welcome!

```bash
git checkout -b feature/your-feature
```

Make your changes, commit them, and create a pull request.

Example:

```bash
git add .
git commit -m "Add new fashion analysis feature"
git push origin feature/your-feature
```

---

# 📜 License

This project is intended for educational and experimental purposes.

You can add your preferred open-source license here, such as:

```text
MIT License
```

---

# 👨‍💻 Author

**Kawaljot Singh**

Built as an AI/ML and Computer Vision project exploring **local AI, fashion analysis, image processing, and multimodal models**.

---

## ⭐ Support

If you found this project interesting, consider giving the repository a ⭐ on GitHub!

> **AI Fashion Analyzer — Your personal stylist, running locally on your machine. 👗🤖**
