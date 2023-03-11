


def count_parameters(model, verbose=True):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if num_params >= 1e6:
        num_params /= 1e6
        suffix = "M"
    elif num_params >= 1e3:
        num_params /= 1e3
        suffix = "K"
    else:
        suffix = ""
    if verbose:
        print(f"Number of trainable parameters: {num_params:.2f}{suffix}")
    return num_params