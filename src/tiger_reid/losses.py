from torch import nn
from pytorch_metric_learning import losses, miners, distances, reducers

NAMES = ("triplet_hard", "multi_similarity", "proxy_anchor")


class MetricObjective(nn.Module):
    def __init__(
        self,
        name,
        classes,
        triplet_margin=0.3,
        ms_alpha=2.0,
        ms_beta=50.0,
        ms_base=0.5,
        ms_epsilon=0.1,
        proxy_margin=0.1,
        proxy_alpha=32.0,
    ):
        super().__init__()
        self.miner = None
        if name == "triplet_hard":
            distance = distances.LpDistance(normalize_embeddings=True, p=2, power=1)
            self.loss = losses.TripletMarginLoss(
                margin=triplet_margin, distance=distance, reducer=reducers.MeanReducer()
            )
            self.miner = miners.BatchHardMiner(distance=distance)
        elif name == "multi_similarity":
            self.loss = losses.MultiSimilarityLoss(
                alpha=ms_alpha, beta=ms_beta, base=ms_base
            )
            self.miner = miners.MultiSimilarityMiner(epsilon=ms_epsilon)
        elif name == "proxy_anchor":
            self.loss = losses.ProxyAnchorLoss(
                num_classes=classes,
                embedding_size=128,
                margin=proxy_margin,
                alpha=proxy_alpha,
            )
        else:
            raise ValueError(name)

    def forward(self, z, y):
        z = z.float()
        if self.miner is not None:
            return self.loss(z, y, self.miner(z, y))
        return self.loss(z, y)
