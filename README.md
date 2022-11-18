# Digit classification on SVHN

Develop a neural network to implement digit classification on the SVHN streetscape dataset. 

The neural network uses the simplest fully-connected neural network with BP backpropagation algorithm to update the weights (also using CNN). 

The loss function and accuracy rate are used to evaluate the classification effect of the model by modifying the number of layers and learning rate of the neural network and using the variable control method.

## Structure

```
-SVHN Digit Classification
|---model
	|---CNN.py
	|---Network.py  # Fully-connected NN
|---data_loader.py  # Load the SVHN dataset
|---main.py 		 
|---module.py       # Core module for NN
|---utils.py
```

## Requirements

- Python3.7 
- Numpy1.19.5 
- matplotlib3.3.1 
- scipy1.4.1 

## Dataset

| Dataset | Shape        | Label       | Shape      |
| ------- | ------------ | ----------- | ---------- |
| x_train | (3072,73257) | train_label | (10,73257) |
| x_test  | (3072,26032) | test_label  | (10,26032) |

As seen from the figure below, compared with the distribution of the MNIST Dataset (left), the distribution of images in the SVHN Dataset (right) is more scattered and does not have apparent aggregation, indicating that the recognition of SVHN street view numbers is much more complex than MNIST.



Figure 1 Visualize image distribution using PCA. 

## Experiment Result



Figure 2 Loss and accuracy when L=4, Epoch=50, alpha=0.1