# 🖼️ Image Captioning with Deep Learning

This repository contains two Jupyter notebooks implementing **image captioning** using different architectures:

1. **CNN + LSTM** (Custom encoder-decoder model)
2. **InceptionV3 + Pretrained Word Embeddings**

Both approaches aim to generate meaningful captions from input images by extracting visual features and decoding them into natural language.

---

## 📁 Contents

### 🔹 1. `image-captioning-based-on-cnn-based-model-lstm.ipynb`

This notebook implements an encoder-decoder model from scratch:

- **Encoder**: CNN-ResNet50 model for extracting image features.
- **Decoder**: LSTM-based language model that generates captions.
- **Tokenizer**: Converts captions into sequences.
- **Training**: Trained end-to-end on MS COCO-style image-caption pairs.
- **Loss**: CrossEntropyLoss with padding ignored.
- **Optimizer**: Adam.

#### 🧠 Model Architecture

- CNN: Sequential layers (Conv2d → ReLU → Pooling).
- LSTM: Embedded input → LSTM → Linear projection to vocabulary.

#### 📊 Hyperparameters

| Parameter     | Value       |
|---------------|-------------|
| Batch size    | 64          |
| Embedding dim | 256         |
| Hidden size   | 512         |
| Learning rate | 0.001       |
| Epochs        | 20          |

---

### 🔹 2. `image-captioning-with-inception-v3-and-pretrained_embedded_vector.ipynb`

This notebook builds an advanced captioning system with:

- **Encoder**: Pretrained `InceptionV3` model to extract image embeddings.
- **Decoder**: LSTM that decodes image features into text, using pretrained word embeddings (e.g., GloVe).
- **Beam Search**: Optional decoding enhancement for better caption generation.

#### 🧠 Model Architecture

- Encoder: `InceptionV3` up to global average pooling.
- Decoder: Embedded caption tokens → LSTM → Dense → Softmax.

#### 📊 Hyperparameters

| Parameter        | Value          |
|------------------|----------------|
| Image size       | 299x299        |
| Embedding dim    | 300 (GloVe)    |
| LSTM hidden size | 512            |
| Optimizer        | Adam           |
| Loss             | CrossEntropy   |


