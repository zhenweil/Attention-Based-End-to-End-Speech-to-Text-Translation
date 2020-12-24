# Attention-based End-to-end Speech-to-text Translation Neural Network
The model translates speech recordings to their corresponding transcripts with an attention mechanism. 
## Introduction
This is a character-based prediction model that consists of an encoder and an attention-based decoder. The encoder uses a pyramidal-LSTM to extract speech embeddings, and the decoder learns to focus on important portion of embeddings and is able to generate corresponding translations. Recordings were converted to spectrograms ahead of time, which contain 40 frequency bands. 
## Installation
```
conda env create -f environment.yaml`
pip install -r requirements.txt
mkdir Data
mkdir Checkpoints
```
## Training
```
cd Code
python3 main.py
```
Dynamic teacher-forcing rate was implemented to improve performance. It takes around 10 epoch for the model to form a diagonal attention plot. After around 35 epochs, the model started converging and achieved edit distance of about 10. 
<div align="center">
  <img src="Attention/attention.png" width="200"/>
</div>
## Results
