# Will the customer leave the bank?

**Task: Customer behaviour predictions**

We are currently working within the data science team of a bank. We have been tasked to produce a machine learning model that predicts whether a customer will leave or stay with the bank, depending on some independent variables. 

We will be using an "Artificial Neural Network" to complete this business task.

## Artificial Neural Networks

An Artificial Neural Network (ANN) is a "Deep Learning" model that can be used for regression and classification. We create an artificial structure where we have nodes that represent neurons. The image below shows a neuron.

<img src = 'Screen1.png' width='700'>

Neurons by themselves are not of much use but when you have many neurons together, they work together to produce magic. The digram below shows an ANN with one hidden layer.

<img src = 'Screen2.png' width='700'>

Each neuron independently in the hidden layer will not be able to predict the outcome y, but together, combining will be able to produce the output layer. If trained properly, they will do an accurate job.
The neurons in the hidden layer will pick up different combinations on input values (independent variables) and different weights. They will work together to produce the final output y.

**Cost function**

The error in our prediction can be evaluated using the cost function; our goal is to minimise the cost function. The cost function is given below.

<img src = 'Screen3.png' width='350'>

The cost function value C gets fed back into the neural network and the corresponding weights are updated. This process is called "Backwards Propagation".

<img src = 'Screen4.png' width='350'>

```
Epoch 1/100
8000/8000 [==============================] - 1s 179us/step - loss: 0.4801 - accuracy: 0.7960
Epoch 2/100
8000/8000 [==============================] - 1s 149us/step - loss: 0.4252 - accuracy: 0.7960

...

Epoch 99/100
8000/8000 [==============================] - 1s 153us/step - loss: 0.4007 - accuracy: 0.8354
Epoch 100/100
8000/8000 [==============================] - 1s 152us/step - loss: 0.4008 - accuracy: 0.8356
```
### References

Neuron diagram:  https://askabiologist.asu.edu/neuron-anatomy

ANN digram: https://www.udemy.com/course/machinelearning/
