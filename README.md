# Fashion MNIST: CNN vs ANN Optimizer Comparison

A comprehensive deep learning project comparing Convolutional Neural Networks (CNN) and Artificial Neural Networks (ANN) for image classification on the Fashion MNIST dataset, with a focus on optimizer performance (Adam, RMSprop, SGD). Includes model training, evaluation, and visualization of results.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview
This project benchmarks CNN and ANN models on Fashion MNIST using different optimizers. Includes code for:
- Data preprocessing
- Model building
- Training pipelines
- Evaluation metrics
- Visualization of accuracy/loss curves
- Confusion matrices
- Sample predictions

## Dataset
**Fashion MNIST**  
70,000 grayscale images (28x28 pixels) of 10 clothing categories  
Source: [TensorFlow Keras Datasets](https://www.tensorflow.org/datasets/catalog/fashion_mnist)  

**Classes**:  
T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

## Model Architectures
### CNN Architecture
Sequential([
Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
MaxPooling2D((2,2)),
Conv2D(64, (3,3), activation='relu'),
MaxPooling2D((2,2)),
Conv2D(64, (3,3), activation='relu'),
Flatten(),
Dense(64, activation='relu'),
Dense(10, activation='softmax')
])


### ANN Architecture
Sequential([
Flatten(input_shape=(28,28,1)),
Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
Dropout(0.4),
Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
Dropout(0.5),
Dense(10, activation='softmax')
])


This executes:
- Data loading/preprocessing
- Model training with Adam/RMSprop/SGD
- Accuracy/loss tracking
- Performance visualization

## Results

### Performance Metrics
| Model Type | Optimizer | Test Accuracy (%) |
|------------|-----------|------------------|
| CNN        | Adam      | 89.96            |
| CNN        | RMSprop   | 89.49            |
| CNN        | SGD       | 85.05            |
| ANN        | Adam      | 84.26            |
| ANN        | RMSprop   | 78.51            |
| ANN        | SGD       | 84.96            |

### Visualizations
- **Accuracy/Loss Curves**: Training vs validation metrics
- **Optimizer Comparison**: Validation accuracy across optimizers
- **Confusion Matrix**: Class prediction heatmap
- **Sample Predictions**: Random test images with predictions

![Accuracy Comparison](https://via.placeholder.com/600x400.png?text=Accuracy+Curves)
*Example accuracy comparison plot*

**Low Accuracy**  
- Try different learning rates
- Adjust regularization parameters
- Increase training epochs

## Contributing
Contributions welcome! Please:
1. Fork the repository
2. Create your feature branch
3. Commit changes
4. Push to branch
5. Open pull request

## License
MIT License - see [LICENSE](LICENSE) for details

## Acknowledgements
- [Zalando Research](https://github.com/zalandoresearch/fashion-mnist) for dataset
- TensorFlow/Keras teams
- Matplotlib/Seaborn communities


