# 👁️ Sightless Navigator

> Empowering visually impaired users with real-time Indian currency recognition using YOLOv8 and TFLite on Android.

---

### 📱 Overview

Sightless Navigator is an Android application designed to assist visually impaired individuals by detecting Indian currency notes in real time   using the device camera. Leveraging **YOLOv8 object detection**, converted to **TFLite**, and **Text-to-Speech (TTS)** capabilities, the app speaks out the denomination of detected notes, enabling independent financial interactions.

---

### 🧠 How It Works

```mermaid
graph TD
    A[Camera Feed] --> B[YOLOv8 TFLite Model]
    B --> C{Currency/Object Detected?}
    C -- Yes --> D[Extract Denomination]
    D --> E[Text-to-Speech]
    E --> F[Audio Feedback]
    C -- No --> G[Wait for Detection]
```

---

### 🔍 Features

- 🔎 Real-time object detection using YOLOv8n.
- 🎙️ Text-to-Speech for voice feedback of detected currency.
- 📲 Lightweight TFLite model optimized for mobile devices.
- 🇮🇳 Focused on Indian currency denominations.
- 🧑‍🦯 Accessibility-first design.

---

### 🧰 Tech Stack

| Component        | Technology                          |
|------------------|-------------------------------------|
| Model Training   | YOLOv8n (Ultralytics)               |
| Conversion       | ONNX → TFLite                       |
| App Development  | Android Studio (Java/Kotlin)        |
| Voice Feedback   | Android TTS Engine                  |
| Deployment       | Android Device                      |

--- 

## 🚀 Project Flow Diagram

```mermaid
flowchart TD
    subgraph TRAINING_STAGE [Model Preparation]
        A1[🧠 Train YOLOv8n Model<br>Ultralytics]
        A2[🔄 Convert YOLOv8n → TFLite]
        A1 --> A2 
    end

    subgraph APP_DEV [App Development]
        B1[💻 Android Studio<br>Java / Kotlin]
        B2[🗣️ Android TTS Engine]
        B3[📲 Load TFLite Model]
        A2 --> B3
        B3 --> B1 --> B2
    end

    subgraph DEPLOYMENT [Deployment & Usage]
        C1[📱 Deploy to Android Device]
        C2[👁️ Detect Currency/Objects in Real-time]
        C3[🔊 Voice Feedback to User]
        B2 --> C1 --> C2 --> C3
    end
```

---

### 🏁 Getting Started
🔧 Prerequisites

Before you begin, make sure you have the following:

- 📱 **Android Studio** installed
- 🔌 **Android device** with camera & speaker
- 🧠 Basic knowledge of **Java/Kotlin** (helpful but not mandatory)
- 🛠️ **Git** installed
- ⚡ For training YOLOv8n:
  - ✅ **GPU with CUDA** support (recommended), **or**
  - 🐍 Install PyTorch with CPU support:  
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    ```
  - 🧰 Install the ultralytics library
    ```bash
    pip install ultralytics
    ```
- 📂 Dataset

    The custom dataset used for training the YOLOv8n model is available on Google Drive:
  
    - 🔗 [Download Dataset](https://drive.google.com/drive/folders/1vSwiNxIfMKrbS2MCvLdEm8Fhji5rJbNB?usp=sharing)


---

### 🚀 Installation
- Clone the repo:

```bash
git clone https://github.com/yourusername/sightless-navigator.git
cd sightless-navigator
```
- Open in Android Studio.
- Add the TFLite model (currency_model.tflite) to the assets folder.
- Build and run on your Android device.

---

### 🙌 Acknowledgements
- Ultralytics YOLOv8
- TensorFlow Lite
- Android TTS

---

### 👤 Author

- Aditya Patil
- 📧 adityapatil0718@gmail.com
- 🔗 www.linkedin.com/in/aditya-patil-83bab626a
