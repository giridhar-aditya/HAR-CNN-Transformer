# HAR-CNN-Transformer 🚶‍♂️🤖🧠

A hybrid CNN and Transformer model for classifying human activities from smartphone sensor time series data.

---

## Dataset 📊

This project uses the [Human Activity Recognition Using Smartphones Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) from the UCI Machine Learning Repository.

The dataset contains sensor signals (accelerometer and gyroscope) collected from 30 subjects performing six different activities:  
🚶‍♂️ Walking, 🆙 Walking Upstairs, 🆗 Walking Downstairs, 🪑 Sitting, 🧍 Standing, 🛌 Laying.

---

## Model Description 🧱✨

The model combines convolutional neural networks (CNN) for local feature extraction with Transformer encoder layers to capture temporal dependencies across the sequence data. Input sensor data is processed in windows of 128 time steps with 9 sensor channels.

---

## Performance 📈

- Test Accuracy: **93%** ✅  
- Classification report and confusion matrix provided in the evaluation script.

---

## Usage ▶️

1. Download and extract the dataset from the [UCI repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).  
2. Ensure the folder structure remains intact (`train` and `test` folders with inertial signals).  
3. Run the training script to train the model.  
4. Use the evaluation script to evaluate performance on the test set.

---

## Requirements 🛠️

- Python 3.8+  
- PyTorch  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  

Install dependencies via:

```bash
pip install torch numpy scikit-learn matplotlib seaborn
```
