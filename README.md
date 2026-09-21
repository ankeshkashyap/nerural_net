# NumPy Digit Recognizer

A handwritten digit recognition system built **from scratch using
NumPy**, trained on MNIST, exposed through a FastAPI backend, and
deployed as an interactive web application.

**No PyTorch. No TensorFlow. No Keras.**

The neural network, forward pass, backpropagation, gradient descent,
ReLU, softmax, and training loop were implemented manually with NumPy.

## Live Demo

[Open the live application](https://nerural-net.onrender.com/)

## Project Repository

[GitHub Repository](https://github.com/ankeshkashyap/nerural_net)

## Training Notebook

The complete model-building and training process is available in the
Google Colab notebook:

[Open the Colab
Notebook](https://colab.research.google.com/drive/1nGYPCdfPgXL_H9e-LJAbwD4HBz-fBaXG?usp=sharing)

> The Colab notebook is useful if you want to see how the neural network
> was built and trained rather than only seeing the deployed
> application.

------------------------------------------------------------------------

## Overview

This project implements a fully connected neural network for classifying
handwritten digits from **0 to 9**.

The trained model is then integrated into a FastAPI application that
allows users to:

-   Draw a digit on an interactive canvas.
-   Submit the drawing to the backend.
-   Receive a predicted digit and softmax confidence.
-   Select random MNIST samples directly from the application.
-   Send those samples through the same prediction pipeline.
-   Interact with the model through a browser without requiring Python
    locally.

------------------------------------------------------------------------

## Neural Network Architecture

``` text
Input Image
28 × 28 = 784 pixels
        │
        ▼
┌─────────────────┐
│ Dense Layer     │
│ 784 → 128       │
└─────────────────┘
        │
        ▼
      ReLU
        │
        ▼
┌─────────────────┐
│ Dense Layer     │
│ 128 → 10        │
└─────────────────┘
        │
        ▼
     Softmax
        │
        ▼
  Digit Prediction
     0 → 9
```

### Parameters

  Layer            Input   Output
  -------------- ------- --------
  Input              784      784
  Hidden Layer       784      128
  Output Layer       128       10

The input image is flattened from:

``` text
28 × 28
```

into:

``` text
784
```

features.

------------------------------------------------------------------------

## Machine Learning Implementation

The neural network was implemented without a deep learning framework.

### Forward Propagation

``` text
X
↓
Z₁ = XW₁ + b₁
↓
A₁ = ReLU(Z₁)
↓
Z₂ = A₁W₂ + b₂
↓
Softmax(Z₂)
↓
Prediction
```

### Activation Function

The hidden layer uses ReLU:

``` text
ReLU(x) = max(0, x)
```

### Output

The final layer uses numerically stable softmax to convert logits into
class probabilities.

### Loss Function

The model uses categorical cross-entropy:

``` text
L = -log(P(correct class))
```

### Backpropagation

Gradients are calculated manually using matrix operations and propagated
backward through the network.

The output gradient simplifies to:

``` text
dZ₂ = probabilities - one_hot_labels
```

### Optimization

The parameters are updated using gradient descent:

``` text
W ← W - η · dW
b ← b - η · db
```

Training uses mini-batches rather than processing the entire dataset in
a single update.

------------------------------------------------------------------------

## Training

The model was trained on the MNIST handwritten digit dataset.

### Training configuration

-   Architecture: `784 → 128 → 10`
-   Hidden activation: ReLU
-   Output activation: Softmax
-   Loss: Cross-entropy
-   Optimizer: Mini-batch gradient descent
-   Batch size: `64`
-   Weight initialization: He initialization
-   Framework: NumPy

### Test Performance

**MNIST test accuracy: 94.49%**

The deployed model uses the trained NumPy weights stored in:

``` text
model_weights.npz
```

------------------------------------------------------------------------

## Web Application

The browser interface provides two ways to test the model.

### 1. Draw a Digit

Users can draw a digit directly on the canvas.

``` text
Canvas
   ↓
PNG / Base64
   ↓
FastAPI
   ↓
Preprocessing
   ↓
NumPy Neural Network
   ↓
Prediction + Confidence
```

> Hand-drawn digits may have lower prediction accuracy because user
> drawings can differ from the MNIST data distribution used during
> training.

### 2. Choose an MNIST Image

The application can also randomly select 10 images directly from the
MNIST `.ubyte` dataset.

``` text
Choose Image
     ↓
GET /random-images
     ↓
Random MNIST samples
     ↓
Select an image
     ↓
Canvas
     ↓
Predict
```

This provides a useful way to test the model using images from the same
distribution it was trained/evaluated on.

------------------------------------------------------------------------

## Backend API

The application is powered by FastAPI.

### `GET /`

Serves the web application.

### `GET /random-images`

Returns 10 randomly selected MNIST images encoded as PNG data URLs.

### `POST /predict`

Accepts a canvas image and returns:

``` json
{
  "prediction": 5,
  "confidence": 0.9936
}
```

------------------------------------------------------------------------

## Tech Stack

### Machine Learning

-   Python
-   NumPy
-   MNIST
-   Manual forward propagation
-   Manual backpropagation
-   Gradient descent

### Backend

-   FastAPI
-   Uvicorn
-   Pillow

### Frontend

-   HTML
-   CSS
-   JavaScript
-   HTML Canvas API

### Deployment

-   Render
-   GitHub

------------------------------------------------------------------------

## Project Structure

``` text
nerural_net/
│
├── index.html
├── main.py
├── model.py
├── model_weights.npz
├── t10k-images.idx3-ubyte
├── requirements.txt
├── .gitignore
└── README.md
```

### Important files

  -----------------------------------------------------------------------
  File                                Purpose
  ----------------------------------- -----------------------------------
  `model.py`                          Loads trained weights and performs
                                      inference

  `main.py`                           FastAPI server and image-processing
                                      endpoints

  `index.html`                        Interactive browser interface

  `model_weights.npz`                 Trained neural network parameters

  `t10k-images.idx3-ubyte`            MNIST images used for random sample
                                      selection

  `requirements.txt`                  Python dependencies
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Run Locally

Clone the repository:

``` bash
git clone https://github.com/ankeshkashyap/nerural_net.git
cd nerural_net
```

Create a virtual environment:

``` bash
python -m venv .venv
```

Activate it on Windows:

``` powershell
.venv\Scripts\activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Start the FastAPI server:

``` bash
uvicorn main:app --reload
```

Open:

``` text
http://127.0.0.1:8000
```

------------------------------------------------------------------------

## Why Build It From Scratch?

The main goal of this project was not simply to achieve the highest
possible MNIST accuracy.

The goal was to understand what actually happens inside a neural
network:

-   How images become numerical tensors.
-   How matrix multiplication produces activations.
-   How ReLU transforms activations.
-   How softmax produces class probabilities.
-   How cross-entropy measures prediction error.
-   How gradients are calculated.
-   How backpropagation moves those gradients through the network.
-   How gradient descent updates the weights.
-   How mini-batch training affects optimization.
-   How trained weights can be separated from the training process and
    deployed for inference.

------------------------------------------------------------------------

## What I Learned

This project provided hands-on experience with:

-   Vectorized numerical computation using NumPy.
-   Neural network parameter initialization.
-   Matrix dimensions and broadcasting.
-   Forward propagation.
-   Backpropagation.
-   Gradient descent.
-   Mini-batch training.
-   Cross-entropy loss.
-   Softmax numerical stability.
-   Model serialization with NumPy.
-   Image preprocessing.
-   REST APIs with FastAPI.
-   Browser-to-backend communication.
-   Deploying a Python ML application.

One particularly important lesson was **distribution shift**: a model
can perform well on MNIST while performing differently on free-form
drawings because the input distribution is not identical.

------------------------------------------------------------------------

## Future Improvements

Possible improvements include:

-   CNN-based digit recognition.
-   Better preprocessing for free-form drawings.
-   Data augmentation.
-   Calibration of prediction confidence.
-   Training on user-generated handwriting samples.
-   Improved responsive UI.
-   Prediction probability visualization.
-   Model comparison between the NumPy MLP and a CNN.
-   More comprehensive automated testing.

------------------------------------------------------------------------

## License

This project is intended as a learning and portfolio project.
