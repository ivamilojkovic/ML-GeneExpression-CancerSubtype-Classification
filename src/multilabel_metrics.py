from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score, \
    recall_score, matthews_corrcoef, hamming_loss, \
    label_ranking_average_precision_score, average_precision_score

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


def print_all_scores(y_test, predictions, prob_predictions, label_orig, label_pam50, txt_file_name):

    # Compute scores on test set
    subset_acc = accuracy_score(y_test, predictions)
    relax_pam50_acc = relaxed_accuracy(label_pam50, predictions)
    relax_orig_acc = relaxed_accuracy(label_orig, predictions)
    partial_acc = partial_accuracy(y_test, predictions)
    hamm_loss = hamming_loss(y_test, predictions)
    rank_avg_prec = label_ranking_average_precision_score(y_test, prob_predictions)
    avg_prec = average_precision_score(y_test, prob_predictions)

    print('\nTest accuracy: {}'.format(subset_acc))
    print('Test relaxed accuracy (PAM50): {}'.format(relax_pam50_acc))
    print('Test relaxed accuracy (original): {}'.format(relax_orig_acc))
    print('Partial accuracy: {}'.format(partial_acc))
    print('Test Hamming loss: {}\n'.format(hamm_loss))
    print('Ranking average precision: ', rank_avg_prec)
    print('Average precision: ', avg_prec)

    prec_weighted = precision_score(y_test, predictions, 
                                    average='weighted', zero_division=1)
    rec_weighted = recall_score(y_test, predictions, 
                            average='weighted', zero_division=1)
    f1_weighted = f1_score(y_test, predictions, 
                        average='weighted', zero_division=1)

    print('Test precision (weighted): {}'.format(prec_weighted))
    print('Test recall (weighted): {}'.format(rec_weighted))
    print('Test f1 score (weighted): {}\n'.format(f1_weighted))

    prec_macro = precision_score(y_test, predictions, 
                                average='macro', zero_division=1)
    rec_macro = recall_score(y_test, predictions, 
                            average='macro', zero_division=1)
    f1_macro = f1_score(y_test, predictions, 
                        average='macro', zero_division=1)

    print('Test precision (macro): {}'.format(prec_macro))
    print('Test recall (macro): {}'.format(rec_macro))
    print('Test f1 score (macro): {}\n'.format(f1_macro))

    prec_micro = precision_score(y_test, predictions, 
                                average='micro', zero_division=1)
    rec_micro = recall_score(y_test, predictions, 
                            average='micro', zero_division=1)
    f1_micro = f1_score(y_test, predictions, 
                        average='micro', zero_division=1)
    
    print('Test precision (micro): {}'.format(prec_micro))
    print('Test recall (micro): {}'.format(rec_micro))
    print('Test f1 score (micro): {}\n'.format(f1_micro))

    
    with open(txt_file_name, 'w') as file:

        file.write('--- Test scores ---\n')
        file.write(f'Subset accuracy: {subset_acc}\n')
        file.write(f'Relaxed (PAM50) accuracy: {relax_pam50_acc}\n')
        file.write(f'Relaxed (original) accuracy: {relax_orig_acc}\n')
        file.write(f'Partial accuracy: {partial_acc}\n')
        file.write(f'Hamming loss: {hamm_loss}\n')
        file.write(f'Ranking average precision: {rank_avg_prec}\n')
        file.write(f'Average precision: {avg_prec}\n\n')

        file.write(' - Weighted scores -\n')
        file.write(f'Precision: {prec_weighted}\n')
        file.write(f'Recall: {rec_weighted}\n')
        file.write(f'F1 score: {f1_weighted}\n\n')

        file.write(' - Macro scores -\n')
        file.write(f'Precision: {prec_macro}\n')
        file.write(f'Recall: {rec_macro}\n')
        file.write(f'F1 score: {f1_macro}\n\n')

        file.write(' - Micro scores -\n')
        file.write(f'Precision: {prec_micro}\n')
        file.write(f'Recall: {rec_micro}\n')
        file.write(f'F1 score: {f1_micro}\n\n')






