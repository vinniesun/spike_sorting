import torch

class TopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, k):
        values, indices = torch.topk(input_, k, dim=-1, largest=False)
        ctx.save_for_backward(indices)
        ctx.input_shape = input_.shape

        return values, indices.float()  # float output so autograd tracks it

    @staticmethod
    def backward(ctx, grad_values, grad_indices):
        (indices,) = ctx.saved_tensors
        grad_input = grad_values.new_zeros(ctx.input_shape)

        grad = grad_values
        if grad_indices is not None:
            grad = grad + grad_indices  # STE: route index-grad through values

        grad_input.scatter_(-1, indices, grad)

        return grad_input, None

def topk(input_, k):
    return TopK.apply(input_, k)

class Mode(torch.autograd.Function):
    """
    torch.mode with a straight-through estimator backward.
    Gradient is routed back only to the elements that were selected as the mode.
    """

    @staticmethod
    def forward(ctx, input, dim=-1, keepdim=False):
        values, indices = torch.mode(input, dim=dim, keepdim=keepdim)
        ctx.save_for_backward(indices)
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.input_shape = input.shape

        return values if keepdim else values.squeeze(dim)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        dim = ctx.dim

        if not ctx.keepdim:
            grad_output = grad_output.unsqueeze(dim)

        # Straight-through: scatter the gradient to the mode positions
        grad_input = torch.zeros(
            ctx.input_shape, dtype=grad_output.dtype, device=grad_output.device
        )
        grad_input.scatter_(dim, indices, grad_output)

        # None for dim and keepdim (non-tensor args)
        return grad_input, None, None

def mode(input_, dim=-1, keepdim=False):
    return Mode.apply(input_, dim, keepdim)
    
class KNN(torch.nn.Module):
    def __init__(self, k=3):
        super(KNN, self).__init__()
        self.k = k
        self.train_x = None
        self.train_y = None

    def reset(self):
        self.train_x = None
        self.train_y = None

    def fit(self, x, y):
        if self.train_x is None:
            self.train_x = x
        else:
            self.train_x = torch.cat((self.train_x, x), dim=0)

        if self.train_y is None:
            self.train_y = y
        else:
            self.train_y = torch.cat((self.train_y, y), dim=0)

    def forward(self, x):
        ##### replace topk and torch.mode with the custom autograd functions defined above

        # Calculate Euclidean distance
        distances = torch.cdist(x, self.X_train)
        
        # Get indices of k nearest neighbors
        # knn_indices = distances.topk(self.k, largest=False).indices
        _, knn_indices = topk(distances, self.k)
        
        # Retrieve labels of k nearest neighbors
        knn_labels = self.y_train[knn_indices]
        
        # Majority voting: find most common label in neighbors
        # predictions = torch.mode(knn_labels, dim=1).values
        predictions = mode(knn_labels, dim=1)

        return predictions
    