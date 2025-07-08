# Mnist basics

## How is a grayscale image represented on a computer? How about a color image?
Generally matrix of 1 byte per pixel. Colors in RGB so 3 bytes per pixel.

## How are the files and folders in the MNIST_SAMPLE dataset structured? Why?
By numbers

## Explain how the "pixel similarity" approach to classifying digits works.
Compute the average pixel value

## What is a list comprehension? Create one now that selects odd numbers from a list and doubles them.
for loop used to generate a new list with its enumerations (for o in sevens)
```python
[odd_number * 2 for odd_number in number_list if odd_number % 2 != 0]
```
## What is a "rank-3 tensor"?
3D tensor. It'd be a cube representation. The next dimension to a matrix
> The rank of a tensor is the number of dimensions it has. An easy way to identify the rank is the number of indices you would need to reference a number within a tensor. A scalar can be represented as a tensor of rank 0 (no index), a vector can be represented as a tensor of rank 1 (one index, e.g., v[i]), a matrix can be represented as a tensor of rank 2 (two indices, e.g., a[i,j]), and a tensor of rank 3 is a cuboid or a “stack of matrices” (three indices, e.g., b[i,j,k]). In particular, the rank of a tensor is independent of its shape or dimensionality, e.g., a tensor of shape 2x2x2 and a tensor of shape 3x5x7 both have rank 3. Note that the term “rank” has different meanings in the context of tensors and matrices (where it refers to the number of linearly independent column vectors).

## What is the difference between tensor rank and shape? How do you get the rank from the shape?
Rank refers to dimension, shape refers to the length of each dimension

## What are RMSE and L1 norm?
Root Mean Square Error *square of differences*.
L1 norm *mean absolute difference*.

## How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?
Matrix multiplication.

## Create a 3×3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.

## What is broadcasting?
[Pytorch docs](https://pytorch.org/docs/stable/notes/broadcasting.html)
The general idea is that tensors expand to be of the same size as the tensor with larger rank prior to perform operations.

## Are metrics generally calculated using the training set, or the validation set? Why?
Validation set. It wouldn't make sense to use training set because it will always be more accurate on that than on unknown samples.

## What is SGD?
Stochastic Gradient Descent

## Why does SGD use mini-batches?
> In order to take an optimization step we need to calculate the loss over one or more data items. How many should we use? We could calculate it for the whole dataset, and take the average, or we could calculate it for a single data item. But neither of these is ideal. Calculating it for the whole dataset would take a very long time. Calculating it for a single item would not use much information, so it would result in a very imprecise and unstable gradient.

## What are the seven steps in SGD for machine learning?
1. Initialize parameters.
2. Compute prediciton.
3. Calculate loss.
4. Compute gradients
5. Step weights
6. Repeat
7. Stop
> Initialize the parameters – Random values often work best.  
> Calculate the predictions – This is done on the training set, one mini-batch at a time.  
> Calculate the loss – The average loss over the minibatch is calculated  
> Calculate the gradients – this is an approximation of how the parameters need to change in order to minimize the loss function  
> Step the weights – update the parameters based on the calculated weights  
> Repeat the process  
> Stop – In practice, this is either based on time constraints or usually based on when the training/validation losses and metrics stop improving.  

## How do we initialize the weights in a model?
By setting random values on them

## What is "loss"?
Samuel referred to it as `testing the effectiveness of any current weight assignment in terms of actual performance`
> The loss function will return a value based on the given predictions and targets, where lower values correspond to better model predictions.

## Why can't we always use a high learning rate?
Final steps of the learning process should be small because this way we can avoid jumping right and left of a minima.
> The loss may “bounce” around (oscillate) or even diverge, as the optimizer is taking steps that are too large, and updating the parameters faster than it should be.

## What is a "gradient"?
Error induced by a specific neuron. That's why each neuron will have its own gradients during training and weight adjusting.

## Do you need to know how to calculate gradients yourself?
Nah

## Why can't we use accuracy as a loss function?
Because a very small change in weights in a model will almost not affect accuracy and hence not return any valuable insight about the error margins.

## Draw the sigmoid function. What is special about its shape?
I dunno. It's limited above and below.
> Sigmoid function is a smooth curve that squishes all values into values between 0 and 1. Most loss functions assume that the model is outputting some form of a probability or confidence level between 0 and 1 so we use a sigmoid function at the end of the model in order to do this.

## What is the difference between a loss function and a metric?
Loss will be helpful for an automated process of training in order to automatically compute error margins and gradients. A metric on the other side is a more human adjusted metric in order to calculate performance in specific topics we as humans are interested on.

## What is the function to calculate new weights using a learning rate?
SGD? Loss? RMSE?
> The optimizer step function

## What does the DataLoader class do?
> The DataLoader class can take any Python collection and turn it into an iterator over many batches.

## Write pseudocode showing the basic steps taken in each epoch for SGD.
```python
# for epoch in epochs:
for batch in dataset:
    batch_gradients = []
    for sample in batch:
        result = neural_net.forward_propagate(sample['input_values'])
        error = neural_net.compute_error(sample['actual_output'])
        gradients = neural_net.get_gradients(error)
        batch_gradients.append(gradients)
    average_gradients = batch_gradients.average()
    neural_net.update_weights(average_gradients)
```

## Create a function that, if passed two arguments [1,2,3,4] and 'abcd', returns [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]. What is special about that output data structure?
What is special?

> ```python
> def func(a,b): return list(zip(a,b))
> ```
> This data structure is useful for machine learning models when you need lists of tuples where each tuple would contain input data and a label.

## What does view do in PyTorch?
> It changes the shape of a Tensor without changing its contents.

## What are the "bias" parameters in a neural network? Why do we need them?
One bias neuron used to always have at least one output from the previous layer
> Without the bias parameters, if the input is zero, the output will always be zero. Therefore, using bias parameters adds additional flexibility to the model.

## What does the @ operator do in Python?
> This is the matrix multiplication operator.

## What does the backward method do?
> This method returns the current gradients.

## Why do we have to zero the gradients?
> PyTorch will add the gradients of a variable to any previously stored gradients. If the training loop function is called multiple times, without zeroing the gradients, the gradient of current loss would be added to the previously stored gradient value.

## What information do we have to pass to Learner?
> We need to pass in the DataLoaders, the model, the optimization function, the loss function, and optionally any metrics to print.

## Show Python or pseudocode for the basic steps of a training loop.
> ```python
> def train_epoch(model, lr, params):
>     for xb,yb in dl:
>         calc_grad(xb, yb, model)
>         for p in params:
>             p.data -= p.grad*lr
>             p.grad.zero_()
> for i in range(20):
>     train_epoch(model, lr, params)
> ``` 

## What is "ReLU"? Draw a plot of it for values from -2 to +2.
> max(0,x)

## What is an "activation function"?
Function used inside a neuron to compute its output
> The activation function is another function that is part of the neural network, which has the purpose of providing non-linearity to the model. The idea is that without an activation function, we just have multiple linear functions of the form y=mx+b. However, a series of linear layers is equivalent to a single linear layer, so our model can only fit a line to the data. By introducing a non-linearity in between the linear layers, this is no longer true. Each layer is somewhat decoupled from the rest of the layers, and the model can now fit much more complex functions. In fact, it can be mathematically proven that such a model can solve any computable problem to an arbitrarily high accuracy, if the model is large enough with the correct weights. This is known as the universal approximation theorem.

## What's the difference between F.relu and nn.ReLU?
> F.relu is a Python function for the relu activation function. On the other hand, nn.ReLU is a PyTorch module. This means that it is a Python class that can be called as a function in the same way as F.relu.

## The universal approximation theorem shows that any function can be approximated as closely as needed using just one nonlinearity. So why do we normally use more?
> There are practical performance benefits to using more than one nonlinearity. We can use a deeper model with less number of parameters, better performance, faster training, and less compute/memory requirements.

