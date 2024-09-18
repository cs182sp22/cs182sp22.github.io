import torch

def little_demo():
    # Scenario: show that when feature scale differences are very large, then the large magnitude
    # features can dominate the activations
    print("\033[32m\n==== demo: features with big scale differences ====\033[0m")
    input_feats = torch.tensor(
        [
            # [float number_of_pennies_in_market, float time_of_day_in_hours]
            [1.2e9, 0.2],
            [1.4e9, 12.0],
            [2.8e9, 23.9],
        ]
    )
    linear1 = torch.nn.Linear(in_features=2, out_features=4)
    out1 = linear1(input_feats)

    print(f"out1: {out1}")

    # Possible issue: very large logits cause Sigmoid to "max out" at 1.0, where the gradient is very flat,
    # aka "vanishing gradient" problem
    sigmoid = torch.nn.Sigmoid()
    out1_sigmoid = sigmoid(out1)
    print(f"out1_sigmoid: {out1_sigmoid}")

    # Vs: same scenario, but where feature values are similar in magnitude
    print("\033[32m\n==== demo: features with similar magnitude scales ====\033[0m")
    input_feats_similar_scale = torch.tensor(
        [
            # [float dim0, float dim1]
            [1.2, 0.2],
            [1.4, 12.0],
            [2.8, 23.9],
        ]
    )
    out1_similar_scale = linear1(input_feats_similar_scale)
    print(f"out1_similar_scale: {out1_similar_scale}")
    out1_similar_scale_sigmoid = sigmoid(out1_similar_scale)
    print(f"out1_similar_scale_sigmoid: {out1_similar_scale_sigmoid}")

    import ipdb; ipdb.set_trace()


def linear_demo():
    A = torch.tensor([[0.001, 0.001, 0.001], [1, 1, 1]], dtype=torch.float32)
    input_feats = torch.tensor(
        [
            [1000, 1],
        ],
        dtype=torch.float32,
    )
    linear = torch.nn.Linear(2, 3, bias=False)
    print(f"linear(input_feats): ", linear(input_feats))

    linear.weight.data.copy_(A.transpose(0, 1))
    print(f"linear(input_feats) (post adjust): ", linear(input_feats))


def main():
    # little_demo()
    linear_demo()

if __name__ == '__main__':
    main()
