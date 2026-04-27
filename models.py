"""
Binary classifiers for EMS transport prediction.

EmsClassifier         — two-branch (tabular + dispatch embedding)  [primary]
AttentionEmsClassifier — attention over vitals + dispatch embedding
DeepResidualMLP       — 4-layer MLP with residual skip connections
MLPClassifier         — flat 2-layer MLP with BatchNorm
ShallowMLP            — single hidden layer, no BatchNorm
LogisticRegression    — linear baseline

All PyTorch models share the forward signature:
    model(x_tab: Tensor, x_text: Tensor) -> Tensor  (logit, shape [B, 1])
x_text is ignored by tabular-only models.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ── Primary model ─────────────────────────────────────────────────────────────

class EmsClassifier(nn.Module):
    """
    Two-branch classifier: tabular branch + learnable dispatch embedding.

    tabular [B, tabular_dim] → Linear → BN → ReLU → Dropout → [B, hidden]  ─┐
    text    [B]              → Embedding(vocab, embed_dim)    → [B, embed]   ─┤ cat
                                                                               └→ Linear → BN → ReLU → Dropout → Linear → logit
    """

    def __init__(
        self,
        tabular_dim: int,
        vocab_size: int,
        embed_dim: int = 16,
        hidden: int = 64,
        dropout: float = 0.2,
    ):
        """
        Initialize the EMS classifier with tabular and text-processing branches.

        Parameters
        ----------
        tabular_dim
            Number of tabular input features supplied to the tabular branch.
        vocab_size
            Size of the chief-complaint vocabulary used by the embedding layer.
        embed_dim
            Width of the learned text embedding for each encoded chief complaint.
        hidden
            Hidden dimension used by the tabular branch before fusion.
        dropout
            Dropout probability applied in the tabular branch and prediction head.

        Returns
        -------
        None
            Initializes the module layers in place.
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.tab_net = nn.Sequential(
            nn.Linear(tabular_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + embed_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_tab: torch.Tensor, x_text: torch.Tensor) -> torch.Tensor:
        """
        Compute transport logits from tabular features and encoded chief complaints.

        Parameters
        ----------
        x_tab
            Batch of tabular patient features with shape ``[batch_size, tabular_dim]``.
        x_text
            Batch of label-encoded chief-complaint indices with shape ``[batch_size]``.

        Returns
        -------
        logits
            Predicted transport logits with shape ``[batch_size, 1]``.
        """
        tab_out  = self.tab_net(x_tab)
        text_out = self.embed(x_text)
        return self.head(torch.cat([tab_out, text_out], dim=1))


# ── Attention model ───────────────────────────────────────────────────────────

class AttentionEmsClassifier(nn.Module):
    """
    Soft feature-attention over vitals before the tabular branch, plus dispatch embedding.

    Attention: a learned linear layer produces per-feature weights (softmax-normalised),
    which are multiplied element-wise with x_tab so the model down-weights uninformative
    features (e.g., a missing-then-imputed vital) dynamically per sample.
    """

    def __init__(
        self,
        tabular_dim: int,
        vocab_size: int,
        embed_dim: int = 16,
        hidden: int = 64,
        dropout: float = 0.2,
    ):
        """
        Initialize the attention-based EMS classifier.

        Parameters
        ----------
        tabular_dim
            Number of tabular input features used by the attention layer and tabular branch.
        vocab_size
            Size of the chief-complaint vocabulary used by the embedding layer.
        embed_dim
            Width of the learned embedding assigned to each encoded chief complaint.
        hidden
            Hidden dimension used after applying feature attention to the tabular inputs.
        dropout
            Dropout probability applied in the tabular branch and prediction head.

        Returns
        -------
        None
            Initializes the attention, embedding, and prediction layers in place.
        """
        super().__init__()
        self.embed    = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attn_fc  = nn.Linear(tabular_dim, tabular_dim)
        self.tab_net  = nn.Sequential(
            nn.Linear(tabular_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + embed_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_tab: torch.Tensor, x_text: torch.Tensor) -> torch.Tensor:
        """
        Apply feature attention and return transport logits for a batch of encounters.

        Parameters
        ----------
        x_tab
            Batch of tabular patient features with shape ``[batch_size, tabular_dim]``.
        x_text
            Batch of label-encoded chief-complaint indices with shape ``[batch_size]``.

        Returns
        -------
        logits
            Predicted transport logits with shape ``[batch_size, 1]``.
        """
        attn     = torch.softmax(self.attn_fc(x_tab), dim=-1)  # [B, tabular_dim]
        x_tab    = x_tab * attn                                 # attended features
        tab_out  = self.tab_net(x_tab)
        text_out = self.embed(x_text)
        return self.head(torch.cat([tab_out, text_out], dim=1))


# ── Residual model ────────────────────────────────────────────────────────────

class _ResBlock(nn.Module):
    """Pre-activation residual block: BN → ReLU → Linear → BN → ReLU → Linear + skip."""

    def __init__(self, dim: int, dropout: float):
        """
        Initialize a residual block used inside the deep tabular MLP.

        Parameters
        ----------
        dim
            Width of the hidden representation preserved through the residual connection.
        dropout
            Dropout probability applied between the two linear layers.

        Returns
        -------
        None
            Initializes the residual block layers in place.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the residual transformation and add the skip connection.

        Parameters
        ----------
        x
            Hidden representation to transform with shape ``[batch_size, dim]``.

        Returns
        -------
        output
            Updated hidden representation after adding the residual branch output.
        """
        return x + self.block(x)


class DeepResidualMLP(nn.Module):
    """
    Four-layer MLP with two residual skip connections. Tabular features only.

    input → project(hidden) → ResBlock → ResBlock → Linear(1) → logit
    """

    def __init__(self, input_dim: int, hidden: int = 64, dropout: float = 0.2):
        """
        Initialize the deep residual MLP for tabular features.

        Parameters
        ----------
        input_dim
            Number of tabular input features supplied to the network.
        hidden
            Width of the shared hidden representation used by the residual stack.
        dropout
            Dropout probability applied inside each residual block.

        Returns
        -------
        None
            Initializes the projection, residual, and output layers in place.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.res1 = _ResBlock(hidden, dropout)
        self.res2 = _ResBlock(hidden, dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x_tab: torch.Tensor, _x_text: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute transport logits from tabular features using the residual MLP.

        Parameters
        ----------
        x_tab
            Batch of tabular patient features with shape ``[batch_size, input_dim]``.
        _x_text
            Unused text input kept for API compatibility with the other classifiers.

        Returns
        -------
        logits
            Predicted transport logits with shape ``[batch_size, 1]``.
        """
        x = self.proj(x_tab)
        x = self.res1(x)
        x = self.res2(x)
        return self.head(x)


# ── Flat MLP baselines ────────────────────────────────────────────────────────

class MLPClassifier(nn.Module):
    """Two hidden layers with BatchNorm. Tabular features only."""

    def __init__(self, input_dim: int, hidden1: int = 64, hidden2: int = 32, dropout: float = 0.2):
        """
        Initialize the two-layer tabular MLP baseline.

        Parameters
        ----------
        input_dim
            Number of tabular input features supplied to the network.
        hidden1
            Width of the first hidden layer.
        hidden2
            Width of the second hidden layer.
        dropout
            Dropout probability applied after each hidden layer.

        Returns
        -------
        None
            Initializes the sequential MLP layers in place.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x_tab: torch.Tensor, _x_text: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute transport logits from tabular features using the MLP baseline.

        Parameters
        ----------
        x_tab
            Batch of tabular patient features with shape ``[batch_size, input_dim]``.
        _x_text
            Unused text input kept for API compatibility with the other classifiers.

        Returns
        -------
        logits
            Predicted transport logits with shape ``[batch_size, 1]``.
        """
        return self.net(x_tab)


class ShallowMLP(nn.Module):
    """Single hidden layer, no BatchNorm. Simplest DL model above linear."""

    def __init__(self, input_dim: int, hidden: int = 64, dropout: float = 0.2):
        """
        Initialize the shallow tabular MLP baseline.

        Parameters
        ----------
        input_dim
            Number of tabular input features supplied to the network.
        hidden
            Width of the single hidden layer.
        dropout
            Dropout probability applied before the output layer.

        Returns
        -------
        None
            Initializes the sequential MLP layers in place.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_tab: torch.Tensor, _x_text: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute transport logits from tabular features using the shallow MLP.

        Parameters
        ----------
        x_tab
            Batch of tabular patient features with shape ``[batch_size, input_dim]``.
        _x_text
            Unused text input kept for API compatibility with the other classifiers.

        Returns
        -------
        logits
            Predicted transport logits with shape ``[batch_size, 1]``.
        """
        return self.net(x_tab)


class LogisticRegression(nn.Module):
    """Linear baseline. Tabular features only."""

    def __init__(self, input_dim: int):
        """
        Initialize the linear tabular baseline.

        Parameters
        ----------
        input_dim
            Number of tabular input features supplied to the linear layer.

        Returns
        -------
        None
            Initializes the linear prediction layer in place.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x_tab: torch.Tensor, _x_text: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute transport logits from tabular features using a linear model.

        Parameters
        ----------
        x_tab
            Batch of tabular patient features with shape ``[batch_size, input_dim]``.
        _x_text
            Unused text input kept for API compatibility with the other classifiers.

        Returns
        -------
        logits
            Predicted transport logits with shape ``[batch_size, 1]``.
        """
        return self.linear(x_tab)
