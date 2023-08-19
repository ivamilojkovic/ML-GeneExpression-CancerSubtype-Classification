from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.model_selection import GridSearchCV

# Create a custom scoring function to calculate the desired metrics
def custom_scoring(estimator, X, y):
    y_pred = estimator.predict(X)
    weighted_precision = precision_score(y, y_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
    micro_precision = precision_score(y, y_pred, average='macro', zero_division=0)
    micro_recall = recall_score(y, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y, y_pred)
    
    # Return a dictionary of the metrics you want to track
    return {
        'accuracy': estimator.score(X, y),  # Calculate accuracy
        'MCC': mcc,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'macro_precision': micro_precision,
        'macro_recall': micro_recall,
        'macro_f1': micro_f1
    }