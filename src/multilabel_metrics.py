from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score
from sklearn.metrics import recall_score, matthews_corrcoef, hamming_loss

def relaxed_accuracy(y_true, y_pred):

    num_instances = len(y_true)
    correct_instances = 0

    for i in range(num_instances):
        if y_true.shape[1] == 5:
            if any(y_pred.iloc[i][y_true.iloc[i,:]==1]==1):
                correct_instances += 1
        else:
            if y_pred.iloc[i, y_true[i]]==1:
                correct_instances += 1

    return correct_instances / num_instances

def partial_accuracy(y_true, y_pred):

    """Partial accuracy is defined as the proportion of the predicted correct 
       labels to the total (predicted and actual) of labels for that instance. 
       Total partial accuracy is teh average across all instances.
    """

    num_instances = len(y_true)
    correct_instances = 0

    for i in range(num_instances):
        if any(y_pred.iloc[i][y_true.iloc[i,:]==1]==1):
            correct_instances += \
                sum(y_pred.iloc[i, :] & y_true.iloc[i, :]) / sum(y_pred.iloc[i, :] | y_true.iloc[i, :])

    return correct_instances / num_instances

def partial_precision(y_true, y_pred):

    num_instances = len(y_true)
    correct_instances = 0

    for i in range(num_instances):
        if any(y_pred.iloc[i][y_true.iloc[i,:]==1]==1):
            correct_instances += \
                sum(y_pred.iloc[i, :] & y_true.iloc[i, :]) / sum(y_pred.iloc[i, :])

    return correct_instances / num_instances

def partial_recall(y_true, y_pred):

    num_instances = len(y_true)
    correct_instances = 0
    
    for i in range(num_instances):
        if any(y_pred.iloc[i][y_true.iloc[i,:]==1]==1):
            correct_instances += \
                sum(y_pred.iloc[i, :] & y_true.iloc[i, :]) / sum(y_true.iloc[i, :])

    return correct_instances / num_instances

def partial_f1_score(y_true, y_pred):

    num_instances = len(y_true)
    correct_instances = 0
    
    for i in range(num_instances):
        if any(y_pred.iloc[i][y_true.iloc[i,:]==1]==1):
            correct_instances += \
                2 * sum(y_pred.iloc[i, :] & y_true.iloc[i, :]) / (sum(y_true.iloc[i, :]) + sum(y_pred.iloc[i, :]))

    return correct_instances / num_instances


def print_all_scores(y_test, predictions):

    print('\nTest accuracy: {}'.format(accuracy_score(y_test, predictions)))
    print('\nPartial accuracy: {}'.format(partial_accuracy(y_test, predictions)))
    print('Test Hamming loss: {}\n'.format(hamming_loss(y_test, predictions)))

    print('Test precision (weighted): {}'.\
        format(precision_score(y_test, predictions, 
                                average='weighted', zero_division=1)))
    print('Test recall (weighted): {}'.\
        format(recall_score(y_test, predictions, 
                            average='weighted', zero_division=1)))
    print('Test f1 score (weighted): {}\n'.\
        format(f1_score(y_test, predictions, 
                        average='weighted', zero_division=1)))

    print('Test precision (macro): {}'.\
        format(precision_score(y_test, predictions, 
                                average='macro', zero_division=1)))
    print('Test recall (macro): {}'.\
        format(recall_score(y_test, predictions, 
                            average='macro', zero_division=1)))
    print('Test f1 score (macro): {}\n'.\
        format(f1_score(y_test, predictions, 
                        average='macro', zero_division=1)))

    print('Test precision (micro): {}'.\
        format(precision_score(y_test, predictions, 
                                average='micro', zero_division=1)))
    print('Test recall (micro): {}'.\
        format(recall_score(y_test, predictions, 
                            average='micro', zero_division=1)))
    print('Test f1 score (micro): {}\n'.\
        format(f1_score(y_test, predictions, 
                        average='micro', zero_division=1)))
