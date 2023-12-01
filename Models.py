import torch
from torch import nn

class MLPDrop(nn.Module):
    def __init__(self, drop_rate: float=0.5):
        super().__init__()

        # Define the intermediate layers
        self.intermediate = nn.Sequential(
            nn.Linear(2, 100),
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.Linear(100, 10),
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.Linear(10, 10),
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.Linear(10, 10),
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.Linear(10, 10),
            nn.ELU(),
            nn.Dropout(drop_rate)
        )

        # Final layer
        self.output = nn.Sequential(
            nn.Linear(10, 1)
        )

    def forward(self, x: torch.Tensor):
        # Output of intermediate layers
        x = self.intermediate(x)

        # Final output
        x = self.output(x)

        return x
    
    def forwardLogistic(self, x: torch.Tensor):
        """
        Process the otput using the sigmoid activation
        """
        out = self.forward(x)
        out = torch.sigmoid(out)

        return out
    
    def sample(self, x: torch.Tensor, N: int=100):
        """
        Generate N samples from the output of the layer using dropout
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
    
