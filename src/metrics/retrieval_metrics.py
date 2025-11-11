def recall_at_k(matches_idx, gt_idx, k=1):
    Q = matches_idx.shape[0]
    correct = 0
    for q in range(Q):
        preds = matches_idx[q,:k]
        g = gt_idx[q]
        if isinstance(g,(list,set)):
            if any(p in g for p in preds): correct += 1
        else:
            if g in preds: correct += 1
    return correct/Q if Q>0 else 0.0

def compute_all_recalls(I,gt,ks=(1,5,10)):
    return {f'recall@{k}': recall_at_k(I,gt,k) for k in ks}
