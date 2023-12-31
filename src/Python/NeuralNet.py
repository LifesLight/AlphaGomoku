import torch.nn as nn

# The model architecture, change with caution due to possible state loading issues

class ResidualLayer(nn.Module):
    def __init__(self, filters, kernal_size=3):
        super().__init__()

        self.conv2d_sequential = nn.Sequential(
            nn.Conv2d(filters, filters, kernal_size, padding=(kernal_size - 1) // 2),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(filters, filters, kernal_size, padding=(kernal_size - 1) // 2),
            nn.BatchNorm2d(filters),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv2d_sequential(x)
        x += residual
        x = self.relu(x)

        return x

class ConvolutionLayer(nn.Module):
    def __init__(self, infilters, outfilters, kernal_size=3):
        super().__init__()

        self.conv2d_sequential = nn.Sequential(                
            nn.Conv2d(infilters, outfilters, kernal_size, padding=(kernal_size - 1) // 2),
            nn.BatchNorm2d(outfilters),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv2d_sequential(x)
        return x

class ConvolutionLayerNoPad(nn.Module):
    def __init__(self, infilters, outfilters, kernal_size=3):
        super().__init__()

        self.conv2d_sequential = nn.Sequential(                
            nn.Conv2d(infilters, outfilters, kernal_size, padding=0),
            nn.BatchNorm2d(outfilters),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv2d_sequential(x)
        return x

class PolicyHead(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.filters = filters

        self.head = nn.Sequential(
            nn.Conv2d(self.filters, 2, 1),
            nn.Flatten(),
            nn.BatchNorm1d(450),
            nn.ReLU(),
            nn.Linear(450, 225)
        )

    def forward(self, x):
        x = self.head(x)
        return x

    def forward(self, x):
        x = self.head(x)
        return x


class ValueHead(nn.Module):
    def __init__(self, Filters, LinearFilters):
        super().__init__()
        self.filters = Filters
        self.linearFilters = LinearFilters

        self.conv_layers = nn.ModuleList([ConvolutionLayerNoPad(self.filters, self.filters, kernal_size=3) for _ in range(7)])

        self.value = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.filters, self.linearFilters),
            nn.BatchNorm1d(self.linearFilters),
            nn.ReLU(),
            nn.Linear(self.linearFilters, self.linearFilters),
            nn.BatchNorm1d(self.linearFilters),
            nn.ReLU(),
            nn.Linear(self.linearFilters, 1),
            nn.Tanh()
        )

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = self.value(x)
        return x

class ResidualNetwork(nn.Module):
    def __init__(self, filters, layers, history_depth):
        super().__init__()

        self.conv_layer = ConvolutionLayer(history_depth, filters, kernal_size=3)
        self.residual_layers = nn.ModuleList([ResidualLayer(filters, kernal_size=3) for _ in range(layers)])

    def forward(self, x):
        x = self.conv_layer(x)
        for res_layer in self.residual_layers:
            x = res_layer(x)
        return x
    
class PolicyNetwork(nn.Module):
    def __init__(self, filters):
        super().__init__()

        self.policy_head = PolicyHead(filters)
    
    def forward(self, x):
        x = self.policy_head(x)
        return x
    
class ValueNetwork(nn.Module):
    def __init__(self, filters, linFilters):
        super().__init__()

        self.value_head = ValueHead(filters, linFilters)
    
    def forward(self, x):
        x = self.value_head(x)
        return x