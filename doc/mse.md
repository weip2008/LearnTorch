Mean Square Error (MSE) is a common metric used to evaluate the performance of a regression model. It measures the average of the squares of the errors, that is, the average squared difference between the estimated values and the actual values. The formula for MSE is:

\[ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \]

where:
- \( n \) is the number of data points,
- \( y_i \) is the actual value of the dependent variable for data point \( i \),
- \( \hat{y}_i \) is the predicted value of the dependent variable for data point \( i \).

MSE is a measure of the quality of an estimator—it is always non-negative, and values closer to zero indicate better fit. However, it has the drawback of not being in the same unit as the original data, making its interpretation somewhat challenging.

Standard deviation is a measure of the amount of variation or dispersion of a set of values. It is calculated as the square root of the variance. The formula for standard deviation is:

\[ \text{SD} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^2} \]

where:
- \( N \) is the number of observations,
- \( x_i \) is the value of the i-th observation,
- \( \bar{x} \) is the mean of the observations.

Standard deviation gives an indication of how spread out the values in a dataset are. A smaller standard deviation indicates that the values are closer to the mean, while a larger standard deviation indicates that the values are more spread out.

* [calculate MSE](../src/mse.py)
* [Understand Gradient to zerro](../src/mse1.py)
* [](../src/mse2.py)