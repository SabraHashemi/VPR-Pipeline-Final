class WandBLogger:
    def __init__(self, project='VPR_Benchmark', run_name=None, enabled=True):
        self.enabled=enabled
        try:
            import wandb; self.wandb=wandb
            if enabled: self.wandb.init(project=project, name=run_name)
        except Exception:
            self.enabled=False; self.wandb=None
    def log(self,dct):
        if self.enabled and self.wandb: self.wandb.log(dct)
    def finish(self):
        if self.wandb: self.wandb.finish()
