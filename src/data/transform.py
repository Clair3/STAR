import copy
import torch


class RandomFeatureMask:
    """
    Randomly masks features (channels) in multivariate time series.

    For each feature independently:
        - with probability p_feature:
            - with probability p_full:
                mask the entire feature
            - otherwise:
                mask 1...max_blocks temporal blocks

    Expected tensor shape:
        (time, features)
    """

    def __init__(
        self,
        keys,
        p_feature=0.2,
        p_full=0.3,
        max_blocks=3,
        min_block=5,
        max_block=60,
    ):
        self.keys = keys
        self.p_feature = p_feature
        self.p_full = p_full
        self.max_blocks = max_blocks
        self.min_block = min_block
        self.max_block = max_block

    def __call__(self, sample):

        # avoid modifying tensors in-place
        sample = {k: v.clone() if torch.is_tensor(v) else v for k, v in sample.items()}

        for key in self.keys:

            x = sample[key]

            if x.ndim != 2:
                raise ValueError(
                    f"{key} must have shape (time, features), got {x.shape}"
                )

            T, C = x.shape

            # Scale block length with sequence length if not specified
            min_block = min(self.min_block, T)
            max_block = min(self.max_block, T)

            if min_block > max_block:
                min_block = max_block

            for c in range(C):

                if torch.rand(()) > self.p_feature:
                    continue

                # Mask entire variable
                if torch.rand(()) < self.p_full:
                    x[:, c] = torch.nan
                    continue

                # Mask random temporal blocks
                n_blocks = torch.randint(
                    1,
                    self.max_blocks + 1,
                    (1,),
                ).item()

                for _ in range(n_blocks):

                    block = torch.randint(
                        self.min_block,
                        max_block + 1,
                        (1,),
                    ).item()

                    block = min(block, T)

                    start = torch.randint(
                        0,
                        T - block + 1,
                        (1,),
                    ).item()

                    end = start + block

                    x[start:end, c] = torch.nan

        return sample


class RandomGaussianJitter:
    """
    Adds Gaussian noise to selected features in a multivariate time series.

    For each feature independently, with probability p_feature, adds
    noise ~ N(0, sigma^2), where sigma is either a fixed scalar or scaled
    by the per-feature std computed from the sample itself.

    Expected tensor shape:
        (time, features)
    """

    def __init__(
        self,
        keys,
        p_feature=0.5,
        sigma=0.05,
        relative=True,
        clip_min=None,
        clip_max=None,
    ):
        self.keys = keys
        self.p_feature = p_feature
        self.sigma = sigma
        self.relative = relative
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, sample):
        sample = {
            k: v.clone() if torch.is_tensor(v) else copy.deepcopy(v)
            for k, v in sample.items()
        }

        for key in self.keys:
            x = sample[key]

            if x.ndim != 2:
                raise ValueError(
                    f"{key} must have shape (time, features), got {x.shape}"
                )

            T, C = x.shape

            for c in range(C):
                if torch.rand(()) > self.p_feature:
                    continue

                col = x[:, c]

                if self.relative:
                    # scale noise by the feature's own variability
                    # (ignoring nan in case masking ran before jitter)
                    valid = col[~torch.isnan(col)] if torch.isnan(col).any() else col
                    std = valid.std() if valid.numel() > 1 else torch.tensor(1.0)
                    noise_std = self.sigma * std
                else:
                    noise_std = torch.tensor(self.sigma)

                noise = torch.randn(T) * noise_std

                if torch.isnan(col).any():
                    mask = ~torch.isnan(col)
                    x[mask, c] = x[mask, c] + noise[mask]
                else:
                    x[:, c] = x[:, c] + noise

                if self.clip_min is not None or self.clip_max is not None:
                    x[:, c] = x[:, c].clamp(min=self.clip_min, max=self.clip_max)

        return sample
