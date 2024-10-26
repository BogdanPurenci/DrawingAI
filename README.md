
# AI Drawing Classifier

This project is a machine learning model that classifies hand-drawn images into five categories: **apple**, **envelope**, **eye**, **parachute**, and **television**. The model is built using Convolutional Neural Networks (CNNs) with TensorFlow, Keras, Numpy, and Matplotlib. The goal is to predict the class of a drawing based on its image.

## Table of Contents
- [Project Overview](#project-overview)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Real-time Plotting](#real-time-plotting)
- [Testing and Evaluation](#testing-and-evaluation)
- [Making Predictions](#making-predictions)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project demonstrates how to build and train a CNN to classify grayscale images into one of five classes. The images are preprocessed into 50x50 pixel grayscale images and normalized for input into the model.

The code includes:
- Image loading and preprocessing
- Model training
- Real-time visualization of loss and accuracy during training
- Image prediction after training

## Dependencies
To run this project, you need the following Python libraries:

- `TensorFlow`
- `Keras`
- `Numpy`
- `Matplotlib`
- `Pillow`
- `scikit-learn`

You can install the required libraries using the following command:
```bash
pip install tensorflow keras numpy matplotlib pillow scikit-learn
```

## Dataset
The dataset should be structured as follows:
- Create separate folders for each class in the format `class_0` for **apple**, `class_1` for **envelope**, and so on up to `class_4` for **television**.
- Place the corresponding images inside each class folder. The images should be in grayscale or will be converted to grayscale during preprocessing.

Each image is resized to 50x50 pixels and normalized (pixel values between 0 and 1) before being fed into the model.

## Model Architecture
The model is a Convolutional Neural Network (CNN) built using Keras, with the following architecture:
1. **Conv2D Layer**: 32 filters of size 3x3, ReLU activation
2. **MaxPooling Layer**: Pool size of 3x3, stride of 2
3. **Conv2D Layer**: 64 filters of size 3x3, ReLU activation
4. **MaxPooling Layer**: Pool size of 3x3, stride of 2
5. **Conv2D Layer**: 128 filters of size 3x3, ReLU activation
6. **MaxPooling Layer**: Pool size of 3x3, stride of 2
7. **Flatten Layer**
8. **Dense Layer**: 64 neurons, ReLU activation
9. **Output Dense Layer**: 5 neurons (for the 5 classes), softmax activation for classification

### Compilation
The model uses Adam optimizer with a learning rate of 0.005 and categorical cross-entropy loss, since it is a multi-class classification problem.

```python
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

## Training the Model
The data is split into training (80%) and testing (20%) sets using `train_test_split` from `scikit-learn`. The training data is reshaped to include an additional channel for grayscale images (50x50x1).

The model is trained for 50 epochs with a batch size of 32:
```python
history = model.fit(X_antrenare, y_antrenare, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[real_time_plot])
```

## Real-time Plotting
A custom `RealTimePlot` callback is included to visualize the evolution of training and validation loss/accuracy after each epoch. This helps track the model's performance during training.

## Testing and Evaluation
After training, the model is evaluated on the test set using:
```python
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Acuratețea pe setul de testare: {test_acc * 100:.2f}%')
```

The accuracy of the model is printed as a percentage.

## Making Predictions
To make predictions on new images, use the `prezice_imagine` function. It takes an image path, processes the image, and predicts the class:

```python
clasa_prezisa = prezice_imagine(image_path)
print(f'Predicted class: {clasa_prezisa}')
```

The predicted class is displayed alongside the input image using Matplotlib.

## Results
After training, the model can classify drawings into one of the five categories with reasonable accuracy. Training and test accuracy, as well as the evolution of loss and accuracy, are visualized during training.

## Contributing
Feel free to submit issues or pull requests to enhance the project, such as adding more classes, improving the model architecture, or extending functionality.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
