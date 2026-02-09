# Galaxy Morphology Classification with CNNs

This project explores **galaxy morphology classification** using **Convolutional Neural Networks (CNNs)** in TensorFlow/Keras.  
The goal is to understand how different CNN design choices affect performance, generalization, and training stability.



## Project Goals

- Build a CNN to classify galaxy images into morphology classes
- Understand **overfitting vs underfitting** in image models
- Compare:
  - Flatten vs Global Average Pooling
  - Small vs larger-capacity models
  - Regularization strategies
- Establish strong baselines before moving to **transfer learning**

---

## Dataset

- **Task**: Multi-class galaxy morphology classification
- **Input**: RGB galaxy images
- **Classes**: `NUM_CLASSES = 10`
- **Image size**: `IMAGE_SIZE = (128, 128)`
- **Train / validation split** handled via Keras generators

### Version 1 — Baseline CNN (Flatten)

**Architecture**
- 3 convolutional blocks
- Conv2D + ReLU
- MaxPooling
- Flatten
- Dense(128)
- Dense(NUM_CLASSES, softmax)

**Summary**
- ~3.3M trainable parameters
- High training accuracy
- Strong overfitting after a few epochs

**Results**
- Train accuracy: ~97%
- Validation accuracy: ~59–61%
- Validation loss: increasing

### Version 1.1 — Parameter Reduction

**Motivation**
- Reduce parameter count
- Improve generalization
- Remove fully-connected over-reliance

**Key Changes**
- Flatten → GlobalAveragePooling2D
- BatchNormalization after convolution layers
- Removed dense hidden layer

**Summary**
- ~95k parameters
- Much smaller capacity
- More stable training

**Results**
- Train accuracy: ~56%
- Validation accuracy: ~20–43%
- Model underfits
- Capacity too small for complex visual morphology
- GlobalAveragePooling removes critical spatial information too early


### Version 2 — Transfer Learning with EfficientNetB0

**Why EfficientNetB0**
- Strong pretrained visual features
- Good accuracy–efficiency tradeoff
- Suitable for limited data and CPU training

**Setup**
- EfficientNetB0 pretrained on ImageNet
- Backbone frozen
- Custom classification head
- Model-specific preprocessing (`preprocess_input`)

**Result**
- Stable training
- Validation accuracy ~62%
- Improved generalization compared to models trained from scratch


## Training Details

- **Framework**: TensorFlow / Keras
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy
- **Batch size**: 32
- **Callbacks**:
  - EarlyStopping (monitoring `val_loss`)
  - ModelCheckpoint (best validation model)


 python -m src.training.train


 ## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib