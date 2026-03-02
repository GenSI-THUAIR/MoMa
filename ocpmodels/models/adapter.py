"""
Adapter module for fine-tuning pre-trained models.

This module implements adapter layers that can be used to fine-tune pre-trained models
by adding small trainable layers between existing layers. This approach allows for
efficient fine-tuning while keeping most of the pre-trained model frozen.
"""

# Standard library imports
import math
from typing import Optional, Union

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_bert_weights(module: nn.Module) -> None:
    """Initialize weights using BERT-style initialization.
    
    Args:
        module: The module to initialize weights for
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # std defaults to 0.02, this might need to be changed
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

class AdapterLayer(nn.Module):
    """Adapter layer for fine-tuning pre-trained models.
    
    This layer implements the adapter architecture, which consists of:
    1. A down-projection layer
    2. A non-linear activation (ReLU)
    3. An up-projection layer
    4. Optional layer normalization
    5. A scaling factor
    
    The adapter can be initialized using either BERT-style or LoRA-style initialization.
    
    Attributes:
        n_embd (int): Input/output embedding dimension
        down_size (int): Bottleneck dimension for the adapter
        adapter_layernorm_option (str): Where to apply layer normalization ('in', 'out', or None)
        scale (Union[float, nn.Parameter]): Scaling factor for the adapter output
        down_proj (nn.Linear): Down-projection layer
        up_proj (nn.Linear): Up-projection layer
        non_linear_func (nn.ReLU): Non-linear activation function
        dropout (float): Dropout probability
    """

    def __init__(
        self,
        config: Optional[object] = None,
        d_model: Optional[int] = None,
        bottleneck: Optional[int] = None,
        dropout: float = 0.0,
        init_option: str = "bert",
        adapter_scalar: Union[str, float] = "1.0",
        adapter_layernorm_option: str = "in",
    ) -> None:
        """Initialize the adapter layer.
        
        Args:
            config: Configuration object containing model parameters
            d_model: Input/output embedding dimension
            bottleneck: Bottleneck dimension for the adapter
            dropout: Dropout probability
            init_option: Weight initialization method ('bert' or 'lora')
            adapter_scalar: Scaling factor for the adapter output
            adapter_layernorm_option: Where to apply layer normalization
        """
        super().__init__()
        
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck
        self.adapter_layernorm_option = adapter_layernorm_option
        self.dropout = dropout

        # Initialize layer normalization if needed
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option in ["in", "out"]:
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        # Initialize scaling factor
        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        # Initialize projection layers
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        # Initialize weights
        if init_option == "bert":
            self.apply(init_bert_weights)
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        add_residual: bool = True,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the adapter layer.
        
        Args:
            x: Input tensor
            add_residual: Whether to add the residual connection
            residual: Optional residual tensor to add
            
        Returns:
            Output tensor after passing through the adapter layer
        """
        residual = x if residual is None else residual

        # Apply layer normalization if specified
        if self.adapter_layernorm_option == "in":
            x = self.adapter_layer_norm_before(x)

        # Down projection
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = F.dropout(down, p=self.dropout, training=self.training)
        
        # Up projection
        up = self.up_proj(down)
        up = up * self.scale

        # Apply layer normalization if specified
        if self.adapter_layernorm_option == "out":
            up = self.adapter_layer_norm_before(up)

        # Add residual if specified
        if add_residual:
            output = up + residual
        else:
            output = up

        return output
