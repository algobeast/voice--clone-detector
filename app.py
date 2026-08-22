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
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;500&display=swap');

.gradio-container {
    background: #10505c !important;
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 !important;
    min-height: 100vh !important;
    font-family: 'Inter', sans-serif !important;
}
.contain, .main, .wrap { background: transparent !important; }

#hero {
    text-align: center;
    padding: 5rem 1.5rem 1rem 1.5rem;
    max-width: 640px;
    margin: 0 auto;
}
#hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    line-height: 1.1;
    color: #ffffff;
    margin: 0 0 1.5rem 0;
}
#hero p {
    font-size: 1.05rem;
    line-height: 1.65;
    color: #cfe3e6;
    margin: 0;
}

#panel {
    max-width: 560px;
    margin: 2.5rem auto 0 auto;
    padding: 0 1.5rem 5rem 1.5rem;
}

/* audio box */
.gradio-container .audio-container,
.gradio-container [data-testid="audio"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1.5px solid rgba(255,255,255,0.25) !important;
    border-radius: 20px !important;
    color: #fff !important;
}

/* button */
button.primary, .gradio-container button.lg {
    background: #ffffff !important;
    color: #10505c !important;
    border: 2px solid #2ecc8f !important;
    border-radius: 999px !important;
    font-size: 1.05rem !important;
    font-weight: 500 !important;
    padding: 1rem 2rem !important;
    width: 100% !important;
    margin-top: 1.5rem !important;
}
button.primary:hover { background: #2ecc8f !important; color: #fff !important; }

#verdict {
    font-family: 'Playfair Display', serif;
    text-align: center;
    font-size: 3rem;
    padding: 2.5rem 0 0.5rem 0;
}
#score {
    text-align: center;
    font-size: 0.95rem;
    color: #cfe3e6;
    letter-spacing: 0.02em;
    padding-bottom: 1rem;
}
footer { display: none !important; }
label, .label-wrap span { color: #cfe3e6 !important; }
"""

def analyse(audio_path):
    if audio_path is None:
        return "", ""
    spec = to_spectrogram(audio_path)
    t = torch.tensor(spec).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        score = torch.softmax(model(t), 1)[0, 1].item() * 100
    if score > 70:
        v = '<div id="verdict" style="color:#ff8a80">Synthetic</div>'
    elif score > 40:
        v = '<div id="verdict" style="color:#ffd166">Uncertain</div>'
    else:
        v = '<div id="verdict" style="color:#2ecc8f">Human</div>'
    s = f'<div id="score">{score:.1f}% synthetic probability</div>'
    return v, s

with gr.Blocks(css=CSS, theme=gr.themes.Base()) as demo:
    gr.HTML("""
    <div id="hero">
      <h1>Is That Voice<br>Really Human?</h1>
      <p>AI can now clone anyone's voice from a few seconds of audio.
         Upload or record a clip and we'll tell you whether a machine made it.</p>
    </div>
    """)
    with gr.Column(elem_id="panel"):
        audio = gr.Audio(type="filepath", label="Upload or record audio")
        btn = gr.Button("Analyse Voice", variant="primary", size="lg")
        verdict = gr.HTML()
        score = gr.HTML()
    btn.click(analyse, inputs=audio, outputs=[verdict, score])

demo.launch(server_name="0.0.0.0", server_port=7860)
