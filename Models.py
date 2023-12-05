from typing import List
import torch
from torch import nn

class MLPDropBinary(nn.Module):
    def __init__(self,
                 layer_sizes : List[int],
                 drop_rate: float=0.5,
        ):
        """
        Multi-Layer Perceptron (MLP) with dropout.

        Parameters:
        - layer_sizes (List[int]): List of layer sizes, including input size. As it is for binary classification the final layer is always one neuron.
        - drop_rate (float, optional): Dropout rate for intermediate layers. Default is 0.5.
        """

        super().__init__()

        # Define the intermediate layers
        intermediate_layers = []
        for i in range(len(layer_sizes) - 1):
            intermediate_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            intermediate_layers.append(nn.ELU())
            intermediate_layers.append(nn.Dropout(drop_rate))

        self.intermediate = nn.Sequential(*intermediate_layers)

        # Final layer
        self.output = nn.Sequential(
            nn.Linear(layer_sizes[-1], 1)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the network.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        # Output of intermediate layers
        x = self.intermediate(x)

        # Final output
        x = self.output(x)

        return x
    
    def forwardLogistic(self, x: torch.Tensor):
        """
        Process the output using the sigmoid activation

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor after sigmoid activation.
        """
        out = self.forward(x)
        out = torch.sigmoid(out)

        return out
    
    def sample(self, x: torch.Tensor, N: int=100):
        """
        Perform Monte Carlo sampling from the neural network with dropout for a probabilistic approach.

        Parameters:
        - x (torch.Tensor): Input tensor.
        - N (int, optional): Number of samples to generate. Default is 100.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - torch.Tensor: Generated samples.
            - torch.Tensor: Mean of the samples.
            - torch.Tensor: Standard deviation of the samples.
        """
        # Set the dropout on
        self.train()

        samples = torch.zeros((x.shape[0],
                               N))
        
        with torch.no_grad():
            for i in range(N):
                sample = self.forwardLogistic(x)
                samples[:, i] = torch.squeeze(sample)


        mu = torch.mean(samples, 1)
        sd = torch.std(samples, 1)

        return samples, mu, sd
    
    
