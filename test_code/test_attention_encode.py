import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / (self.hidden_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attention_weights, value)
        
        return out.squeeze()

class MyModel(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.attention = SelfAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Pass the input through the attention mechanism
        x = self.attention(x)
        # Compute the mean of the attention output along the time dimension
        x = x.mean(dim=1)
        # Pass the output through a linear layer
        x = self.fc(x)
        # Return the logits
        return x

# Initialize the model
model = MyModel(hidden_dim=64, num_classes=4)

# Generate a random input tensor of size (batch_size, seq_len, input_dim)
input_tensor = torch.randn(32, 300, 1)

# Pass the input through the model
output = model(input_tensor)

# Print the output shape
print(output.shape)
