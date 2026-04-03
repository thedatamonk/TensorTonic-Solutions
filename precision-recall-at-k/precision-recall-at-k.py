def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    # Precision @ K = Total # of relevant items in all the retrieved items (k)
    # Recall @ K = Fraction of relevant items in top-K out of all relevant items

    precision_k = len(set(recommended[:k]).intersection(set(relevant))) / k

    recall_k = len(set(recommended[:k]).intersection(set(relevant))) / len(set(relevant))

    return [precision_k, recall_k]