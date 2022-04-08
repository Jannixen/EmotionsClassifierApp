import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
import seaborn as sns


def plot_confusion_matrix(ds,n_categories,model):
    """
    Evaluates model's performance on a dataset. Prints confusion
    and returns full report.
    ds - Dataset object with data to predict labels for.
    n_categories - number of possible label outcomes.
    model - model used to predict labels.
    """
    plt.figure(figsize=(10,8))
    y_pred = tf.math.argmax(model.predict(ds),axis=1).numpy()
    y_true = np.concatenate([y for x, y in ds],axis=0)
    confusion = tf.math.confusion_matrix(y_true,y_pred,num_classes=n_categories)
    report = classification_report(y_true,y_pred,zero_division=0)
    sns.heatmap(confusion,annot=confusion)
    return report