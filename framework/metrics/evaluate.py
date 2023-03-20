from .bleu import compute_bleu
from typing import List

def evaluate_bleu(ref: List[str], hyp: List[str], script='default') -> List[float]:
    all_metrics = []
    bleus = []
    for i in range(len(ref)):
        ref_i = [[ref[i].split()]]
        hyp_i = [hyp[i].split()]

        metrics = compute_bleu(ref_i, hyp_i, max_order=min(len(hyp[i].split()), 3), smooth=True)

        if script == 'nltk':
            metrics = corpus_bleu(refsend, gensend)
            return [metrics]

        all_metrics.append(metrics)
        bleus.append(metrics[0])

    return bleus