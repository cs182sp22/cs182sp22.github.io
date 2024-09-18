import torch

def batch_norm_example():
    # Illustrates batch norm
    print("\033[32m\n==== batch_norm_example() ====\033[0m")
    # suppose we have a matrix representing the output of a Linear layer:
    # shape=[batchsize=4, num_feats=3]
    intermed_features = torch.tensor([
        [0.0, 2.0, 1.0],
        [1.0, -1.0, 5.0],
        [4.0, 8.0, 16.0],
        [10.0, 0.0, 0.0],
    ])
    batchsize = intermed_features.shape[0]
    num_feats = intermed_features.shape[1]

    # Note: scale/shift params (aka affine) default to:
    #   scale=1.0, shift=0.0
    batch_norm_pytorch = torch.nn.BatchNorm1d(
        num_feats,
    )

    feats_post_bn = batch_norm_pytorch(intermed_features)
    print("intermed_features: ", intermed_features)
    print("feats_post_bn: ", feats_post_bn)

    # verify that scale/shift params are 1.0, 0.0:
    # Tip: shape of these is [num_feats=3] because BN uses the same norm params
    #   for every input feature dim
    print("BN weight", batch_norm_pytorch.weight)
    print("BN bias", batch_norm_pytorch.bias)
    # can also view other interesting params like "running_mean"
    print("BN state dict: ", batch_norm_pytorch.state_dict())

    # verify with our "custom" implementation
    print("\033[32m\n==== verify with our custom BN implementation ====\033[0m")
    feats_post_bn_mine_v0 = batch_norm_mine_v0(intermed_features)
    print("feats_post_bn_mine_v0: ", feats_post_bn_mine_v0)

    feats_post_bn_mine_v1 = batch_norm_mine_v1(intermed_features)
    print("feats_post_bn_mine_v1: ", feats_post_bn_mine_v1)

    # Interpretation: BatchNorm ensures that each feature dim is standardized (0 mean, 1 var), where
    #   the feature stats (mean, var) are computed from BATCH statistics
    var_pre, mean_pre = torch.var_mean(intermed_features, dim=0, correction=False)
    var_post, mean_post = torch.var_mean(feats_post_bn, dim=0, correction=False)

    print(f"mean_pre, var_pre: {mean_pre, var_pre}")
    print(f"mean_post, var_post: {mean_post, var_post}")



def batch_norm_mine_v0(input_tensor: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """A custom naive implementation of batch norm.

    Args:
        input_tensor:
            shape=[batchsize, num_feats]

    Returns:
        input_tensor_norm:
            shape=[batchsize, num_feats]

    """
    num_feats = input_tensor.shape[1]
    out = input_tensor.detach().clone()  # aka copy the input_tensor
    for ind_feat in range(num_feats):
        # compute mean, variance
        feat_var, feat_mean = torch.var_mean(
            input_tensor[:, ind_feat],
            # Minor: disable "Bessel's correction" to line up with pytorch output
            correction=0,
        )
        out[:, ind_feat] -= feat_mean
        out[:, ind_feat] /= torch.sqrt(feat_var + eps)
    return out

def batch_norm_mine_v1(input_tensor: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Just like v0, but with better vectorization (via broadcasting).
    More pytorch-y.

    Args:
        input_tensor:
            shape=[batchsize, num_feats]

    Returns:
        input_tensor_norm:
            shape=[batchsize, num_feats]

    """
    out = input_tensor.detach().clone()  # aka copy the input_tensor

    feat_var, feat_mean = torch.var_mean(
        input_tensor,
        dim=0,
        # Minor: disable "Bessel's correction" to line up with pytorch output
        correction=0,
    )
    out -= feat_mean
    out /= torch.sqrt(feat_var + eps)

    return out


def main():
    batch_norm_example()


if __name__ == '__main__':
    main()
