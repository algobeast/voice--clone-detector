import torch
import torch.nn as nn
import librosa
import numpy as np
import gradio as gr

TARGET = 64000

model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1),   nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1),  nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1),  nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 2),
)
model.load_state_dict(torch.load('model_final.pth', map_location='cpu'))
model.eval()

def to_spectrogram(path):
    a, _ = librosa.load(path, sr=16000)
    a = a[:TARGET] if len(a) > TARGET else np.pad(a, (0, TARGET - len(a)))
    m = librosa.feature.melspectrogram(y=a, sr=16000, n_mels=128)
    return librosa.power_to_db(m, ref=np.max).astype(np.float32)

def analyse(audio_path):
    if audio_path is None:
        return "Please upload or record audio first."
    spec = to_spectrogram(audio_path)
    t = torch.tensor(spec).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        score = torch.softmax(model(t), 1)[0, 1].item() * 100
    if score > 70:
        verdict = "LIKELY SYNTHETIC"
    elif score > 40:
        verdict = "UNCERTAIN"
    else:
        verdict = "LIKELY HUMAN"
    return f"{verdict}\n\nSynthetic probability: {score:.1f}%"

demo = gr.Interface(
    fn=analyse,
    inputs=gr.Audio(type="filepath", label="Upload or record a voice"),
    outputs=gr.Textbox(label="Result", lines=4),
    title="Voice Clone Detector",
    description="Detects AI-generated and cloned speech in audio recordings.",
)

demo.launch(server_name="0.0.0.0", server_port=7860)
