# Understanding Digital Handwriting Digit Classification with CNN and MNIST

This repository demonstrates a deep learning project for classifying handwritten digits using the MNIST dataset. It utilizes a Convolutional Neural Network (CNN) to extract spatial features from images and includes steps for training, evaluating, and saving the model. The implementation is done entirely in Kaggle.

[View on Kaggle](https://www.kaggle.com/code/cardinalacre/mnist-dataset-cnn)

---

## 📁 Dataset

We use the standard MNIST dataset:

* `train-images.idx3-ubyte`
* `train-labels.idx1-ubyte`
* `t10k-images.idx3-ubyte`
* `t10k-labels.idx1-ubyte`

These files are loaded using custom logic to read and parse the binary IDX format.

---

## 🧠 Model Architecture

```text
Conv2D(1, 16, kernel_size=3) 
↓ MaxPool2D
Conv2D(16, 32, kernel_size=3)
↓ MaxPool2D
Conv2D(32, 64, kernel_size=3)
↓ MaxPool2D
Flatten
Linear(576, 64)
Linear(64, 10)
```

The CNN is implemented with three convolutional layers, each followed by ReLU and max-pooling operations, ending in two fully connected layers for classification.

---

## ⚙️ Training Details

* **Optimizer**: Adam
* **Loss Function**: CrossEntropyLoss
* **Learning Rate**: 0.001
* **Batch Size**: 16
* **Epochs**: 10
* **Trainable Parameters**: 60,874

---

## 📈 Performance

The model reaches an accuracy of \~99% on the test set by epoch 5. Training and test losses are tracked and stored.

---

## 💾 Model Saving

The trained model is saved to `modelku.pth` along with:

* Model state dict
* Optimizer state dict
* Epoch count
* Training & test losses

```python
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_losses,
    'test_loss': test_losses,
}, '/kaggle/working/modelku.pth')
```

---

## 🛠 Dependencies

* Python 3
* PyTorch
* torchvision
* tqdm
* PIL
* numpy
* matplotlib

---

## 📌 Notes

* Entire code is run and tested in a Kaggle notebook.
* Ideal for learning about image classification and CNN fundamentals.

---

## 📜 License

This project is provided for educational purposes.
