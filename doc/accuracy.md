The accuracy of a model is a metric used to evaluate the performance of a classification model. It measures the proportion of correct predictions made by the model out of the total number of predictions. In other words, accuracy represents how often the model correctly predicts the class labels.

The formula for accuracy is:

\[ \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} \]

Here is a breakdown of the terms involved:
- **Number of Correct Predictions**: This includes both true positives (TP) and true negatives (TN). True positives are instances where the model correctly predicts the positive class, and true negatives are instances where the model correctly predicts the negative class.
- **Total Number of Predictions**: This is the sum of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN). False positives are instances where the model incorrectly predicts the positive class, and false negatives are instances where the model incorrectly predicts the negative class.

So, the formula can also be written as:

\[ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} \]

### Example
Consider a binary classification problem where a model is used to predict whether an email is spam or not. Let's say we have the following confusion matrix after evaluating the model on a test dataset:

|               | Predicted Spam | Predicted Not Spam |
|---------------|----------------|--------------------|
| Actual Spam   | 50             | 10                 |
| Actual Not Spam | 5              | 100                |

- **True Positives (TP)**: 50 (correctly predicted as spam)
- **True Negatives (TN)**: 100 (correctly predicted as not spam)
- **False Positives (FP)**: 5 (incorrectly predicted as spam)
- **False Negatives (FN)**: 10 (incorrectly predicted as not spam)

Using the accuracy formula:

\[ \text{Accuracy} = \frac{50 + 100}{50 + 100 + 5 + 10} = \frac{150}{165} \approx 0.909 \]

So, the accuracy of the model is approximately 90.9%.

### Limitations of Accuracy
While accuracy is a useful metric, it may not always be the best measure of a model's performance, especially in cases of imbalanced datasets where one class is much more frequent than the other. For instance, if 95% of emails are not spam and only 5% are spam, a model that always predicts "not spam" will have a high accuracy of 95%, but it would fail to identify any spam emails. In such cases, other metrics like precision, recall, F1-score, or area under the ROC curve (AUC-ROC) might provide a better evaluation of the model's performance.