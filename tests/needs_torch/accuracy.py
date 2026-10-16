"""The accuracy rule for paths that legitimately differ from the reference in rounding.

Per position, against fp64 oracle logits: e = max |logit - oracle| over the vocabulary, and
kl = KL(oracle || candidate). The candidate passes when the median and maximum of e are within
2x, and the mean of kl within 4x, of the largest such summary among the reference variants on the
same positions; NaN fails. bf16 bugs of rounding size are left to bitwise parity and operator
contracts (test_rms_norm.py). A top-1 flip
where e passes is a near-tie by construction.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

FACTORS = {"median_abs": 2.0, "max_abs": 2.0, "mean_kl": 4.0}


@dataclass(frozen=True)
class Errors:
    """Per-position e and kl in fp64."""

    max_abs: torch.Tensor
    kl: torch.Tensor

    def summary(self) -> dict[str, float]:
        return {"median_abs": self.max_abs.median().item(), "max_abs": self.max_abs.max().item(),
                "mean_kl": self.kl.mean().item()}


def errors(oracle: torch.Tensor, logits: torch.Tensor) -> Errors:
    """Errors of logits [..., V] against fp64 oracle logits of the same shape."""
    vocab = oracle.shape[-1]
    o, x = oracle.reshape(-1, vocab).double(), logits.reshape(-1, vocab).double().to(oracle.device)
    log_o, log_x = torch.log_softmax(o, dim=-1), torch.log_softmax(x, dim=-1)
    kl = (log_o.exp() * (log_o - log_x)).sum(-1).clamp_min(0)  # fp64 rounding can make it slightly negative
    return Errors((x - o).abs().amax(-1), kl)


def failures(candidate: Errors, references: Sequence[Errors]) -> list[str]:
    """The summaries the candidate exceeds; empty when it passes."""
    summary = candidate.summary()
    scale = {name: max(reference.summary()[name] for reference in references) for name in FACTORS}
    return [f"{name} {summary[name]:.3e} > {factor} x {scale[name]:.3e}"
            for name, factor in FACTORS.items() if not summary[name] <= factor * scale[name]]
