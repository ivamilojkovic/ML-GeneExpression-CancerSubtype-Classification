from sklearn.metrics import accuracy_score, f1_score, precision_score, \
    recall_score, hamming_loss, label_ranking_average_precision_score, average_precision_score
import numpy as np
from utils import rank_indices

def relaxed_accuracy(y_true, y_pred):

    """ Relaxed accuracy is an indicator how well the model predicts the primary class, 
        not taking into account if the predicted label is the one with the higest probability 
        prediction i.e. all the predicted labels are threated equally. 
    """

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

def semi_relaxed_accuracy(y_prob_pred, y_true):

    """Semi-relaxed accuracy is an indicator how well the model predicts the primary class, 
        taking into account if the predicted label is the one with the higest probability 
        prediction i.e. the predicted labels are not threated equally. 
    """

    num_instances = y_prob_pred.shape[0]
    correct_instances = 0

    for i in range(num_instances):
        max_label = y_prob_pred.iloc[i,:].idxmax()
        if y_true.iloc[i, :][max_label] == 1:
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

def ordered_subset_accuracy(y_test_mcut, predictions, y_test_corr, prob_predictions):

    # Use y_test_corr and prediction probabilities to compute ranks
    ranked_corr = y_test_corr.apply(rank_indices, axis=1)
    ranked_probs = prob_predictions.apply(rank_indices, axis=1)

    # Indices as in mcuts
    ranked_probs.index = ranked_corr.index
    predictions.index = y_test_mcut.index
    prob_predictions .index = y_test_mcut.index

    # Now multiply by mcut labels only the ranked correlations and 
    # by predictions the ranked probabilties
    ranked_corr *= y_test_mcut
    ranked_probs *= predictions

    cnt_exact_ovelaps = ((ranked_probs == ranked_corr).sum(axis=1) == 5).sum()
    return cnt_exact_ovelaps / y_test_mcut.shape[0]

def k_orders_subset_accuracy(y_test_mcut, predictions, y_test_corr, prob_predictions, k=1):

    # Use y_test_corr and prediction probabilities to compute ranks
    ranked_corr = y_test_corr.apply(rank_indices, axis=1)
    ranked_probs = prob_predictions.apply(rank_indices, axis=1)

    # Indices as in mcuts
    ranked_probs.index = ranked_corr.index
    predictions.index = y_test_mcut.index
    prob_predictions .index = y_test_mcut.index

    # Now multiply by mcut labels only the ranked correlations and 
    # by predictions the ranked probabilties
    ranked_corr *= y_test_mcut
    ranked_probs *= predictions

    # Set zero where value > k
    ranked_corr_below_k = ranked_corr.applymap(lambda x: 0 if x > k else x)
    ranked_probs_below_k = ranked_probs.applymap(lambda x: 0 if x > k else x)
    cnt_correct_first_k_positions = (ranked_corr_below_k == ranked_probs_below_k).all(axis=1).sum()

    return cnt_correct_first_k_positions / y_test_mcut.shape[0]

def secondary_accuracy(y_test_mcut, predictions, y_test_corr, prob_predictions, k=2):

    # Use y_test_corr and prediction probabilities to compute ranks
    ranked_corr = y_test_corr.apply(rank_indices, axis=1)
    ranked_probs = prob_predictions.apply(rank_indices, axis=1)

    # Indices as in mcuts
    ranked_probs.index = ranked_corr.index
    predictions.index = y_test_mcut.index
    prob_predictions .index = y_test_mcut.index

    # Now multiply by mcut labels only the ranked correlations and 
    # by predictions the ranked probabilties
    ranked_corr *= y_test_mcut
    ranked_probs *= predictions

    # Set zero where value > k
    ranked_corr_below_k = ranked_corr.applymap(lambda x: k if x == k else 0)
    ranked_probs_below_k = ranked_probs.applymap(lambda x: k if x == k else 0)
    cnt_correct_first_k_positions = (ranked_corr_below_k == ranked_probs_below_k).all(axis=1).sum()

    return cnt_correct_first_k_positions / y_test_mcut.shape[0]

def print_all_scores(y_test, predictions, prob_predictions, label_orig, label_pam50, txt_file_name=None, y_corr=None):

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

    # Additional metrics
    # semi = semi_relaxed_accuracy(y_prob_pred=prob_predictions, y_true=y_test)
    ordered = ordered_subset_accuracy(y_test_corr=y_corr, y_test_mcut=y_test, 
                                      predictions=predictions, prob_predictions=prob_predictions)

    # print('Semi-relexed: {}'.format(semi))
    print('Ordered subset acc: {}'.format(ordered))
    ord_accs = []
    for order in [1, 2, 3]:
        k_ordered = k_orders_subset_accuracy(y_test_mcut=y_test, predictions=predictions, 
                                            y_test_corr=y_corr, prob_predictions=prob_predictions, k=order)
        ord_accs.append(k_ordered)
        print('Order {} accuracy: {}'.format(order, k_ordered))

    sec_acc = secondary_accuracy(y_test_mcut=y_test, predictions=predictions, 
                                            y_test_corr=y_corr, prob_predictions=prob_predictions, k=2)
    print('\nSecondary accuracy: ', sec_acc)

    if txt_file_name != None:
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

            file.write(' - Additional scores -\n')
            file.write(f'Ordered subset accuracy: {ordered}\n')
            file.write(f'Primary accuracy: {ord_accs[0]}\n')
            file.write(f'Secondary accuracy: {sec_acc}\n')
            file.write(f'Both accuracy: {ord_accs[1]}\n\n')







