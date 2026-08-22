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

CSS = """
.gradio-container {
    background: #0a0a0a !important;
    font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
    max-width: 560px !important;
    margin: 0 auto !important;
}
#title {
    text-align: center;
    font-size: 1.6rem;
    font-weight: 500;
    letter-spacing: -0.02em;
    color: #fafafa;
    margin: 3rem 0 0.4rem 0;
}
#sub {
    text-align: center;
    font-size: 0.85rem;
    color: #666;
    margin-bottom: 2.5rem;
    font-weight: 400;
}
#verdict {
    text-align: center;
    font-size: 1.5rem;
    font-weight: 500;
    letter-spacing: -0.01em;
    padding: 2rem 0 0.5rem 0;
    min-height: 2rem;
}
#score {
    text-align: center;
    font-size: 0.85rem;
    color: #666;
    padding-bottom: 2rem;
}
footer { display: none !important; }
"""

def analyse(audio_path):
    if audio_path is None:
        return "", ""
    spec = to_spectrogram(audio_path)
    t = torch.tensor(spec).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        score = torch.softmax(model(t), 1)[0, 1].item() * 100
    if score > 70:
        v = f'<div id="verdict" style="color:#ff4444">Synthetic</div>'
    elif score > 40:
        v = f'<div id="verdict" style="color:#ffaa00">Uncertain</div>'
    else:
        v = f'<div id="verdict" style="color:#00cc88">Human</div>'
    s = f'<div id="score">{score:.1f}% synthetic probability</div>'
    return v, s

with gr.Blocks(css=CSS, theme=gr.themes.Base()) as demo:
    gr.HTML('<div id="title">Voice Clone Detector</div>')
    gr.HTML('<div id="sub">Upload or record speech to check if it is AI-generated</div>')
    audio = gr.Audio(type="filepath", label="")
    btn = gr.Button("Analyse", variant="primary")
    verdict = gr.HTML()
    score = gr.HTML()
    btn.click(analyse, inputs=audio, outputs=[verdict, score])

demo.launch(server_name="0.0.0.0", server_port=7860)
