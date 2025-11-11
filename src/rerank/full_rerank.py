class FullReranker:
    def __init__(self, device='cpu'):
        self.device=device
        self.impl=None
        try:
            import lightglue; self.impl='lightglue'
        except Exception:
            try:
                import loftr; self.impl='loftr'
            except Exception:
                self.impl=None
    def available(self):
        return self.impl is not None
    def rerank(self, query_path, db_paths_topk):
        # stubbed: returns identity order and zero scores
        order=list(range(len(db_paths_topk)))
        scores=[0]*len(db_paths_topk)
        return order, scores
