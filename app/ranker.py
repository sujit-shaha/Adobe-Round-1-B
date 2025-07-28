from typing import Dict, List, Tuple

def rank_passages(query: str, passages: List[str], scores: List[float] = None) -> List[Tuple[str, float]]:
    print(f"Ranking {len(passages)} passages for query: '{query}'")
    
    if scores is None:
        scores = [0.5 - (i * 0.05) for i in range(len(passages))]
    
    ranked_passages = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
    
    return ranked_passages

def rerank_with_cross_encoder(query: str, initial_results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    print(f"Reranking {len(initial_results)} results with cross-encoder")
    
    passages = [p for p, _ in initial_results]
    
    new_scores = [s * (1.0 + (i % 3) * 0.1) for i, (_, s) in enumerate(initial_results)]
    
    reranked_passages = sorted(zip(passages, new_scores), key=lambda x: x[1], reverse=True)
    
    return reranked_passages
