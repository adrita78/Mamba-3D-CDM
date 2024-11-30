import torch

def permute_within_batch(x, batch):
    # Enumerate over unique batch indices
    unique_batches = torch.unique(batch)

    # Initialize list to store permuted indices
    permuted_indices = []

    for batch_index in unique_batches:
        # Extract indices for the current batch
        indices_in_batch = (batch == batch_index).nonzero().squeeze()

        # Permute indices within the current batch
        permuted_indices_in_batch = indices_in_batch[torch.randperm(len(indices_in_batch))]

        # Append permuted indices to the list
        permuted_indices.append(permuted_indices_in_batch)

    # Concatenate permuted indices into a single tensor
    permuted_indices = torch.cat(permuted_indices)

    return permuted_indices

# Example usage
batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
x = torch.tensor([0, 10, 20, 30, 40, 50, 60])

# Get permuted indices
permuted_indices = permute_within_batch(x, batch)

# Use permuted indices to get the permuted tensor
permuted_x = x[permuted_indices]

print("Original x:", x)
print("Permuted x:", permuted_x)
print("Permuted indices:", permuted_indices)
