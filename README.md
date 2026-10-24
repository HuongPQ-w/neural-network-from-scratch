# Handwritten Digit Recognition – Neural Network from Scratch

A feed-forward neural network implemented **from scratch with NumPy** (no TensorFlow, Keras or PyTorch) to recognize handwritten digits, plus a **real-time camera demo** that reads digits written on paper.

## Highlights

- **Fully manual implementation**: forward pass, backpropagation, ReLU, softmax, cross-entropy loss, He initialization and mini-batch gradient descent.
- **Architecture**: `Input(64) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)`.
- **Dataset**: scikit-learn `load_digits` (8×8 images), 80/20 stratified train/test split.
- **Data augmentation (train set only)**: random shift (±1 px) and rotation (±12°) to handle off-center and tilted handwriting, while keeping the test set untouched to avoid data leakage.
- **Camera demo**: preprocessing (threshold, invert, denoise, crop to the ink bounding box, resize to 8×8) that mimics how `load_digits` images are produced, so camera input matches the training distribution.

## Results

| Metric | Value |
|---|---|
| Test accuracy | **98.61%** (360 test images) |
| Training | 300 epochs, batch size 32, learning rate 0.5 |

The notebook also includes the loss/accuracy curves, a confusion matrix and a per-class classification report.

## Getting started

```bash
git clone https://github.com/HuongPQ-w/neural-network-from-scratch.git
cd neural-network-from-scratch
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook digit_recognition_nn.ipynb
```

Run the cells in order to train and evaluate the model.

### Camera demo (optional)

The last cells run real-time recognition using a phone as an IP camera (e.g. the *IP Webcam* app):

1. Connect the phone and the computer to the same Wi-Fi network.
2. In `CAMERA_CONFIG`, replace `<IP-DIEN-THOAI>` in `IP_WEBCAM_URL` with your phone's IP address (or use `0` for a local webcam).
3. Run the last cell and press `q` to quit.

## Limitations

- `load_digits` is a small, clean dataset; training accuracy reaches 100%, so the model fits the training data very well and may be sensitive to handwriting styles not seen in training.
- Camera accuracy depends on lighting, pen thickness and background.

## Tech stack

Python · NumPy · SciPy · scikit-learn · OpenCV · Matplotlib · Jupyter
