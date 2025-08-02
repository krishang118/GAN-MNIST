# GAN for Handwritten Digit Generation 

This is a PyTorch implementation of a Generative Adversarial Network (GAN) for generating handwritten digits using the MNIST dataset. The model learns to create realistic-looking digits (0-9) by training two neural networks in an adversarial setting.
 
## Features

- Multi-device Support: Automatic detection and accelerated GPU support for CUDA and MPS (Apple Silicon), with CPU fallback
- Progress Tracking: Real-time training progress with progress bars
- Image Generation: Saves generated images during training every 10 epochs
- Easy Model Loading: Functions to load and generate using the saved model

## Architecture

### Generator
- Input: 100-dimensional noise vector
- Output: 28×28 grayscale images
- Architecture: Linear layer + Upsampling with convolutions
- Activation: LeakyReLU + Tanh output

### Discriminator  
- Input: 28×28 grayscale images
- Output: Binary classification (real/fake)
- Architecture: Convolutional layers with progressive downsampling
- Activation: LeakyReLU + Sigmoid output

The model trains for 100 epochs on the MNIST training dataset, which consists of 60,000 training images of handwritten digits (0-9). 

The generator network creates new images, while the discriminator network evaluates the authenticity of the images.

## Project Structure

```
├── MNIST GAN.ipynb           # Main GAN implementation Jupyter Notebook
├── gan_model.pth             # Trained GAN model file
├── generated_images.zip      # Images generated at every 10 epochs during training (you can unzip to see the images)
├── LICENSE                   # The MIT License file
├── README.md                 # This file
```

## How to Run

1. Make sure Python 3.8 + is installed in your system.
2. Clone this repository on your local machine.
3. Install the required dependencies:
```bash
pip install torch torchvision matplotlib numpy tqdm
```
4. Open and run the `MNIST GAN.ipynb` Jupyter Notebook. With the `gan_model.pth` model file present in the project directory, the model will automatically be loaded in, and the generated handwritten digits will be displayed.
5. You can also choose to train the model from scratch. For that, remove the `gan_model.pth` model file from the project directory and run the Jupyter Notebook again. The model will now train from scratch, which includes: downloading the MNIST dataset, setting up the GAN architecture, training for 100 epochs with progress tracking with handwritten digit images generated at every 10 epochs, and then final model saving and image generation at 100 epochs.



GAN-MNIST
A GAN implementation for generating handwritten digits using the MNIST dataset.