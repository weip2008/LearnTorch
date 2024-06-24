Mean Squared Error (MSE) is a common loss function used in regression tasks to measure the average squared difference between the predicted values and the actual values.

\[ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \]

where:
- \( n \) is the number of data points,
- \( y_i \) is the actual value of the dependent variable for data point \( i \),
- \( \hat{y}_i \) is the predicted value of the dependent variable for data point \( i \).
- 
## Partial derivative of MSE
for many iterations, any changes on $w_t$ and $b_t$, the MSE shall be:
\[ \text{MSE}_t = \left( \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \right)_t\] 

where:
\[ \hat{y}_i = w_t x_i + b_t \]
and t is iteration at t (epoch)

Let's first expand the MSE:
\[ \text{MSE} = \frac{1}{n} (\mathbf{y} - \mathbf{X} \mathbf{w_t} - \mathbf{b_t})^T (\mathbf{y} - \mathbf{X} \mathbf{w_t} - \mathbf{b_t}) \]

![](images/Matrix_transpose.gif)

if we define:
\[ \mathbf{\theta_t} = \begin{bmatrix} b_t \\ w_t \end{bmatrix} \]

the MSE could be written as:

\[ \text{MSE} = \frac{1}{n} \mathbf{e}^T \mathbf{e} \]

where
\[ \mathbf{e} = \mathbf{X} \theta_t - \mathbf{y} \]

Notice we use:
$$\mathbf{e}^T \mathbf{e}=\sum_{i=1}^{n}e_i^2$$

Explanation
1. Vector $e$:
    - $e$ is a n-dimensional column vector resulting from the difference between the predicted values $(X\theta_t)$ and the actual values $(y)$.
   - If $X$ is a $n  d$ matrix, is a -dimensional column vector,
and is an -dimensional column vector, then is
also an -dimensional column vector.
1. Squared Error:
The squared error is a scalar value.
Here, (the transpose of ) is a row vector.
When multiplying (a row vector) by (an 
column vector), the result is a scalar.
Partial Derivative
Now, apply the chain rule:
Using the gradient of the squared error term, where , the
derivative with respect to is:

