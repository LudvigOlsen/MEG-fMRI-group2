import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


def evaluate(actual_classes, predicted_classes):
    conf_mat = confusion_matrix(actual_classes, predicted_classes)
    if (len(conf_mat) == 2):
        tn, fp, fn, tp = conf_mat.ravel()
        total = sum([tn, fp, fn, tp])
        recall = sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = ppv = tp / (tp + fp)
        npv = tn / (tn + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        accuracy = (tn + tp) / total
        balanced_accuracy = (sensitivity + specificity) / 2
        return pd.DataFrame({"Accuracy": [accuracy],
                             "Balanced Accuracy": [balanced_accuracy],
                             "F1": [f1],
                             "Precision": [precision],
                             "Recall": [recall],
                             "NPV": [npv],
                             "TP": [tp],
                             "FP": [fp],
                             "TN": [tn],
                             "FN": [fn]})
    else:
        return classification_report(actual_classes, predicted_classes)
