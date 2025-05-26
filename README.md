# 🔮 Next Word Predictor using LSTM (PyTorch)

This project implements a next word prediction model using a simple LSTM architecture in PyTorch. The model is trained on the Wikitext-2 dataset, a clean and curated subset of English Wikipedia articles. Given a sequence of words, the model learns to predict the most likely next word, capturing basic language patterns through training.
- Trained on real Wikipedia text (Wikitext-2)
- Uses `nn.Embedding`, `nn.LSTM`, and `nn.Linear`
- Sequence-to-one prediction for next word inference
 
## 🧰 Technologies Used
- Python 3
- PyTorch
- Hugging Face `datasets` (for loading Wikitext-2)

## 📊 Training Graphs and Results


![inference](https://github.com/user-attachments/assets/a1a4fe1c-a585-4c68-a695-02bba7741203)![lossVsEpochs](https://github.com/user-attachments/assets/fb5571c2-8c1b-4192-ab5d-974c9807d30e)

