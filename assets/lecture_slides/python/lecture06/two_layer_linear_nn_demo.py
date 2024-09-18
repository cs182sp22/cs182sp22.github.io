import torch

class TwoLayerNNClassifierV1(torch.nn.Module):
    """A simple two-layer NN multiclass classifier of the form:
        [input] -> Linear -> act. -> Linear -> act. -> Linear -> [logits] -> Softmax -> [probs]
    where we use Relu as the activation function.
    """
    def __init__(self, num_input_feats: int = 8, num_classes: int = 10, hidden_dim: int = 32):
        """

        Args:
            num_input_feats: Dimensionality of inputs.
            num_classes: Number of target classes to predict.
            hidden_dim: Dimensionality of intermediate states.
        """
        super().__init__()
        self.num_input_feats = num_input_feats
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Define our layers
        self.linear1 = torch.nn.Linear(in_features=self.num_input_feats, out_features=self.hidden_dim)
        self.linear2 = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.linear_cls = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: Input data.
                shape=[batchsize, num_input_feats]

        Returns:
            class_probs:
                shape=[batchsize, num_classes]
        """
        print("START forward()")
        print(f"x: {x}")
        print(f"x.shape: {x.shape}")

        out1 = self.linear1(x)
        print(f"out1: {out1}")
        print(f"out1.shape: {out1.shape}")

        out1 = torch.nn.functional.relu(out1)
        print(f"out1 (post relu): {out1}")

        out2 = self.linear2(out1)
        out2 = torch.nn.functional.relu(out2)
        print(f"out2: {out2}")
        print(f"out2.shape: {out2.shape}")

        # shape=[batchsize, num_classes]
        logits = self.linear_cls(out2)
        print(f"logits: {logits}")
        print(f"logits.shape: {logits.shape}")
        # Note: here I'll have forward() output the class probs, but for numerical/perf reasons
        # it's instead more common to have the model output the logits, and have the Loss function
        # handle doing the log-softmax stuff
        class_probs = torch.nn.functional.softmax(
            logits,
            # do softmax "along" dim1, eg enforce that sum(probs[0, :]) sums to 1.0
            dim=1,
        )
        print(f"class_probs: {class_probs}")
        print(f"class_probs.shape: {class_probs.shape}")
        print("END forward()")
        return class_probs

def classifier_demo():
    # A four-class classifier
    print("\033[32m\n==== classifier_demo() ====\033[0m")
    num_input_feats = 3
    num_classes = 4
    hidden_dim = 5
    classifier = TwoLayerNNClassifierV1(
        num_input_feats=num_input_feats, num_classes=num_classes, hidden_dim=hidden_dim,
    )

    # First, let's do a forward pass to check that it indeed works
    # Create a "fake" input data matrix
    batchsize = 2
    # by convention, inputs/outputs are always [batchsize, ...]
    inputs = torch.rand(size=[batchsize, num_input_feats], dtype=torch.float32)

    # Forward pass
    print("\033[32m\n==== Classifier Forward Pass ====\033[0m")
    class_probs = classifier(inputs)
    # validate that class_probs is the right shape
    assert class_probs.shape == torch.Size([batchsize, num_classes])

    # validate that each sample's class_prob sums to 1.0 (eg Softmax)
    print("\033[32m\n==== Validate class_probs sums to 1.0 ====\033[0m")
    for ind_sample in range(class_probs.shape[0]):
        cur_class_probs = class_probs[ind_sample, :]
        print(f"ind_sample={ind_sample}, cur_class_probs={cur_class_probs}, sum: {torch.sum(cur_class_probs)}")

    # (tip) vectorized way of computing the row sums:
    print(f"(vectorized row sums): {torch.sum(class_probs, dim=1)}")

    # Now let's investigate our model weights!
    print("\033[32m\n==== Investigate model weights ====\033[0m")
    print(f"\nclassifier.linear1: {classifier.linear1}")
    # sure enough, weight.shape=[hidden_dim, num_input_feats], bias.shape=[hidden_dim]
    # the weight/bias values are also initialized randomly
    print(f"classifier.linear1.weight: {classifier.linear1.weight}, shape={classifier.linear1.weight.shape}")
    print(f"classifier.linear1.bias: {classifier.linear1.bias}, shape={classifier.linear1.bias.shape}")

    print(f"\nclassifier.linear2: {classifier.linear2}")
    print(f"classifier.linear2.weight: {classifier.linear2.weight}, shape={classifier.linear2.weight.shape}")
    print(f"classifier.linear2.bias: {classifier.linear2.bias}, shape={classifier.linear2.bias.shape}")

    print(f"\nclassifier.linear_cls: {classifier.linear_cls}")
    print(f"classifier.linear_cls.weight: {classifier.linear_cls.weight}, shape={classifier.linear_cls.weight.shape}")
    print(f"classifier.linear_cls.bias: {classifier.linear_cls.bias}, shape={classifier.linear_cls.bias.shape}")

    # another way to look at a Module's parameters is to look at their state_dict
    print("\033[32m\n==== state_dict ====\033[0m")
    print(f"\nclassifier.linear1.state_dict(): {classifier.linear1.state_dict()}")

    print(f"\nclassifier.state_dict(): {classifier.state_dict()}")

    print("\nclassifier state_dict info:")
    for ind, (param_name, param) in enumerate(classifier.state_dict().items()):
        print(f"(ind={ind}) param_name={param_name} param_shape={param.shape}")


def linear_layer_vs_matmult_demo():
    print("\033[32m\n==== linear_layer_vs_matmult_demo ====\033[0m")
    # Demonstrate that torch.nn.Linear is indeed equivalent to a matrix multiply
    A = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 1],
        ],
        dtype=torch.float32
    )
    # reshape x to be a column vector
    x = torch.tensor([1, 2, 3], dtype=torch.float32)
    x_col = x.reshape([3, 1])
    print("\033[32m\n==== matmult via torch.mm() ====\033[0m")
    print(f"A: {A} shape={A.shape}")
    print(f"x_col: {x_col} shape={x_col.shape}")
    out_matmult = torch.mm(A, x_col)
    print(f"A * x_col = {out_matmult} shape={out_matmult.shape}")

    # Construct a Linear layer that's functionally the same as "A*x"
    print("\033[32m\n==== matmult via Linear layer ====\033[0m")
    linear = torch.nn.Linear(in_features=3, out_features=2, bias=False, dtype=torch.float32)
    with torch.no_grad():
        # construct the Linear layer's weight matrix to be A
        linear.weight.copy_(A)
    # Note: Linear() computes x * A^T, thus x should be a [1, 3] (eg a row vector)
    x_row = x.reshape([1, 3])
    print(f"x_row: {x_row} shape={x_row.shape}")
    out_linear = linear(x_row)
    print(f"out_linear: {out_linear} shape={out_linear.shape}")

    # Note: via broadcasting, we can also use a 1d vector and pytorch will do the "right thing":
    #   https://pytorch.org/docs/stable/notes/broadcasting.html
    out_linear_broadcast = linear(x)
    print(f"out_linear_broadcast: {out_linear_broadcast} shape={out_linear_broadcast.shape}")

if __name__ == '__main__':
    classifier_demo()
    # linear_layer_vs_matmult_demo()
