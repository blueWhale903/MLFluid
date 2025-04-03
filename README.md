# Fluid Simulation Using Machine Learning

-![fluid](https://github.com/user-attachments/assets/eedfa125-e967-47ed-ac3d-01f60288f445)

## Project overview
This project leverages deep learning techniques to predict the velocity field of a fluid simulation over time. The goal is to enhance traditional computational fluid dynamics (CFD) simulations using machine learning, improving speed and efficiency while maintaining accuracy.

## Features
- Predicts velocity fields based on historical fluid state data

- Uses PyTorch for deep learning-based fluid simulation

- Supports visualization of velocity fields and magnitudes

- Generates animations of fluid motion over time

- Implements custom loss functions for improved training accuracy

## Dataset
The dataset is generated using Mantaflow 0.13

## Model Architecture
The neural network takes in past velocity and density fields and predicts the next velocity field. The model is designed with:

- Convolutional layers to capture spatial fluid dynamics

- Recurrent or Transformer layers for temporal dependencies

- Custom loss functions based on physics-informed constraints
