from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile
import numpy as np
import scipy

model = VitsModel.from_pretrained("facebook/mms-tts-vie")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")

text = "Một con vịt xòe ra hai cái cánh, nó kêu rằng cáp cáp cáp, cạp cạp cạp. Gặp hồ nước nó bì bò bì bõm, lúc lên bờ vẫy cái cánh cho khô."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform

output = output.cpu()
data_np = output.numpy()
data_np_squeezed = np.squeeze(data_np)

scipy.io.wavfile.write("tts.wav", rate=model.config.sampling_rate, data=data_np_squeezed)