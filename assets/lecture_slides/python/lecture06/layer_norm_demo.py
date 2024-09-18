import torch


def layer_norm_example():
    # Illustrates layer norm
    print("\033[32m\n==== layer_norm_example() ====\033[0m")
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
    layer_norm_pytorch = torch.nn.LayerNorm(
        num_feats,
    )

    feats_post_layernorm = layer_norm_pytorch(intermed_features)
    print("intermed_features: ", intermed_features)
    print("feats_post_layernorm: ", feats_post_layernorm)

    # verify that scale/shift params are 1.0, 0.0:
    # Tip: shape of these is [num_feats=3] because LN uses the same norm params
    #   for every input feature dim
    print("LN weight", layer_norm_pytorch.weight)
    print("LN bias", layer_norm_pytorch.bias)
    # can also view other interesting params like "running_mean"
    print("LN state dict: ", layer_norm_pytorch.state_dict())

    # verify with our "custom" implementation
    print("\033[32m\n==== verify with our custom LN implementation ====\033[0m")
    feats_post_layernorm_mine_v0 = layer_norm_mine_v0(intermed_features)
    print("feats_post_layernorm_mine_v0: ", feats_post_layernorm_mine_v0)

    feats_post_layernorm_mine_v1 = layer_norm_mine_v1(intermed_features)
    print("feats_post_layernorm_mine_v1: ", feats_post_layernorm_mine_v1)

    # Interpretation: LayerNorm ensures that each input feature vec is standardized (0 mean, 1 var)
    var_pre, mean_pre = torch.var_mean(intermed_features, 1, correction=False)
    var_post, mean_post = torch.var_mean(feats_post_layernorm, 1, correction=False)

    print(f"mean_pre, var_pre: {mean_pre, var_pre}")
    print(f"mean_post, var_post: {mean_post, var_post}")


def layer_norm_mine_v0(input_tensor: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """A custom naive implementation of layer norm.

    Args:
        input_tensor:
            shape=[batchsize, num_feats]

    Returns:
        input_tensor_norm:
            shape=[batchsize, num_feats]

    """
    batchsize = input_tensor.shape[0]
    out = input_tensor.detach().clone()  # aka copy the input_tensor
    for ind_batch in range(batchsize):
        # compute mean, variance
        feat_var, feat_mean = torch.var_mean(
            input_tensor[ind_batch, :],
            # Minor: disable "Bessel's correction" to line up with pytorch output
            correction=0,
        )
        out[ind_batch, :] -= feat_mean
        out[ind_batch, :] /= torch.sqrt(feat_var + eps)
    return out

def layer_norm_mine_v1(input_tensor: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
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
        dim=1,
        # Minor: disable "Bessel's correction" to line up with pytorch output
        correction=0,
    )
    # Tip: since out is shape=[batchsize, num_feats] and feat_mean is [batchsize], must reshape feat_mean to be
    #   shape=[batchsize, 1] to enable broadcasting rules to work correctly
    out -= feat_mean.unsqueeze(1)
    out /= torch.sqrt(feat_var + eps).unsqueeze(1)

    return out


def main():
    layer_norm_example()


if __name__ == '__main__':
    main()
