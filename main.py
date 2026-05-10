from fastapi import FastAPI, UploadFile, File, HTTPException
import io
import uvicorn
import torch
import torchaudio.transforms as transforms
import torch.nn as nn

class CheckAudio(nn.Module):
    def __init__(self):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((8, 8))
        )

        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 35)
        )

    def forward(self, x):
      x = x.unsqueeze(1)
      x = self.first(x)
      x = self.second(x)
      return x

device =torch.device('cuda' if torch.cuda.is_available()else 'cpu')

labels = torch.load('labels.pth')
index_to_label={ind: lab for ind,lab in  enumerate(labels)}
model = CheckAudio()
model.load_state_dict(torch.load("audio_model.pth", map_location=device))
model.to(device)
model.eval()

transform=transforms.MelSpectrogram(
    sample_rate=16000,
    n_mels=64
)
max_len = 100

def process_audio(waveform, sample_rate):
    if simple_rade != 1600:
        new_sample_rate=transform .Resample(orig_freq=sample_rate, new_freq = 16000)
        waveform =new_sample_rate(torch.tensor(waveform))

    spec=transform(waveform).squeeze(0)

    if spec.shape[1]> max_len:
        spec = spec[:, :max_len]

    if spec.shape[1]< max_len:
        count_len=max_len - spec.shape[1]
        f.pad(spec_audio, (0, count_len))

        return spec

check_audio = FastAPI()

@check_audio.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=500, detail="Файл пуст")

        wf,sf = sr.read(io.BytesIO(data), dtype='float32')
        wf = torch.tensor(wf).T

        spec = process_audio(wf, sr).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(spec)
            pred_ind = y_pred.argmax(y_pred,dim=1).item()
            pred_class=index_to_label[pred_ind]

        return {'индекс':pred_ind,"класс": index_to_label[pred_class]}

    except Exception as e:
        raise HTTPException(status_code=504, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(check_audio, host="127.0.0.1", port=8009)
