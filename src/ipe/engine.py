"""Cached generation for one static batch of equal-length prompts.

K and V ([layers, batch, KV heads, capacity, head dim]) and the decode rotary rows are allocated
once; a decode step indexes them with host integers and reads no device value. length is the
committed prefix: a step writes from it, advances it only on success, and attention reads no
further. Capacity is NaN until first written. One thread, one stream.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ipe import ops
from ipe.forward import INTEGER_DTYPES, check_dtype, check_token_ids, decoder_layer
from ipe.weights import LoadedCheckpoint


@dataclass(frozen=True, slots=True)
class Generation:
    """Host tensors; n is max_new_tokens unless every sequence stopped earlier."""

    tokens: torch.Tensor  # [B, n] int64
    logprobs: torch.Tensor  # [B, n] fp32: log_softmax of the fp32 logits at each token
    lengths: torch.Tensor  # [B] int64: tokens up to and including the first stop ID, else n


def sample(logits: torch.Tensor, temperature: float,
           generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokens [B] and their untempered log-probabilities [B] from fp32 logits [B, V].

    Temperature 0 takes the argmax, lowest ID on ties. Otherwise Gumbel-max:
    argmax(logits / temperature - log(-log(U))), U uniform on [0, 1), with no host sync.
    """
    if temperature == 0:
        tokens = logits.argmax(dim=-1)
    else:
        uniform = torch.rand(logits.shape, generator=generator, dtype=torch.float32, device=logits.device)
        tokens = (logits / temperature - torch.log(-torch.log(uniform))).argmax(dim=-1)
    logprobs = F.log_softmax(logits, dim=-1).gather(-1, tokens[:, None]).squeeze(-1)
    return tokens, logprobs


class Engine:
    """Prefill, decode and generate over a KV cache of batch rows and capacity positions."""

    def __init__(self, loaded: LoadedCheckpoint, *, batch: int, capacity: int, dtype: torch.dtype | None = None,
                 backend: str = "sdpa"):
        config = loaded.config
        dtype = loaded.weights.embed_tokens.dtype if dtype is None else dtype
        check_dtype(loaded, dtype, (torch.bfloat16, torch.float32))
        if backend not in ops.BACKENDS:
            raise ValueError(f"backend {backend!r}: expected one of {list(ops.BACKENDS)}")
        if type(batch) is not int or batch < 1:
            raise ValueError(f"batch {batch!r}: expected an integer >= 1")
        if type(capacity) is not int or not 1 <= capacity <= config.max_position_embeddings:
            raise ValueError(f"capacity {capacity!r}: expected an integer from 1 to {config.max_position_embeddings}")
        device = loaded.weights.embed_tokens.device
        self.loaded, self.dtype, self.backend, self.capacity = loaded, dtype, backend, capacity
        shape = (config.num_hidden_layers, batch, config.num_key_value_heads, capacity, config.head_dim)
        self._k = torch.full(shape, float("nan"), dtype=dtype, device=device)
        self._v = torch.full(shape, float("nan"), dtype=dtype, device=device)
        # One position at a time, as the reference evaluates a decode step: CPU cos and sin split a
        # long tensor across threads, and elements at chunk edges round differently.
        rows = [ops.rotary_tables(torch.tensor([p]), config.head_dim, config.rope_theta, dtype) for p in range(capacity)]
        self._cos = torch.cat([cos for cos, _ in rows]).to(device)
        self._sin = torch.cat([sin for _, sin in rows]).to(device)
        self.batch = 0  # rows of the current batch; 0 before the first prefill and after one that fails in compute
        self.length = 0  # committed positions of the current batch

    def prefill(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Start a new batch from prompts [B, P]; fp32 logits [B, V] at position P-1."""
        batch, length = check_token_ids(self.loaded.config, token_ids, self.capacity)
        if batch > self._k.shape[1]:
            raise ValueError(f"token_ids: batch {batch} exceeds the {self._k.shape[1]} allocated rows")
        self.batch = self.length = 0
        # Built like forward_full's, so prefill equals the uncached last position.
        config, device = self.loaded.config, self._k.device
        cos, sin = ops.rotary_tables(torch.arange(length), config.head_dim, config.rope_theta, self.dtype)
        logits = self._step(token_ids, (cos.to(device), sin.to(device)), 0)
        self.batch, self.length = batch, length
        return logits

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """One token per sequence [B] at position length; fp32 logits [B, V].

        Host IDs are bounds-checked and copied synchronously; IDs on the device (the sampler's
        output) are not checked, since that would sync every step.
        """
        if self.batch == 0:
            raise ValueError("decode: no batch; call prefill first")
        if self.length == self.capacity:
            raise ValueError(f"decode: all {self.capacity} positions are committed")
        if not isinstance(tokens, torch.Tensor) or tokens.shape != (self.batch,) or tokens.dtype not in INTEGER_DTYPES:
            raise ValueError(f"tokens: expected integer IDs of shape [{self.batch}]")
        if tokens.device.type == "cpu":
            check_token_ids(self.loaded.config, tokens[:, None], 1)
        rope = (self._cos[self.length:self.length + 1], self._sin[self.length:self.length + 1])
        logits = self._step(tokens[:, None], rope, self.length)
        self.length += 1
        return logits

    def _step(self, token_ids: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor], start: int) -> torch.Tensor:
        config, weights = self.loaded.config, self.loaded.weights
        batch = token_ids.shape[0]
        x = F.embedding(token_ids.to(device=self._k.device, dtype=torch.int64), weights.embed_tokens).to(self.dtype)
        for index, layer in enumerate(weights.layers):
            cache = (self._k[index, :batch], self._v[index, :batch])
            x = decoder_layer(x, layer, rope, config, backend=self.backend, cache=cache, start=start)
        hidden = ops.rms_norm(x[:, -1], weights.norm.to(self.dtype), config.rms_norm_eps)
        return F.linear(hidden, weights.lm_head.to(self.dtype)).float()

    def generate(self, token_ids: torch.Tensor, max_new_tokens: int, *, temperature: float = 0.0,
                 seed: int = 0, stop_ids: tuple[int, ...] = ()) -> Generation:
        """Generate max_new_tokens per prompt in token_ids [B, P]; needs P + max_new_tokens - 1
        positions, since the last token is never fed back.

        Stops early only when every sequence has emitted a stop ID; checking costs one sync per step.
        """
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError(f"max_new_tokens {max_new_tokens!r}: expected an integer >= 1")
        if not (isinstance(temperature, (int, float)) and temperature >= 0):
            raise ValueError(f"temperature {temperature!r}: expected a number >= 0")
        _, length = check_token_ids(self.loaded.config, token_ids, self.capacity)
        if length + max_new_tokens - 1 > self.capacity:
            raise ValueError(f"prompt length {length} + max_new_tokens {max_new_tokens} - 1 exceeds capacity {self.capacity}")
        device = self._k.device
        generator = torch.Generator(device).manual_seed(seed) if temperature > 0 else None
        stops = torch.tensor(stop_ids, dtype=torch.int64, device=device) if stop_ids else None

        logits = self.prefill(token_ids)
        tokens = torch.empty((self.batch, max_new_tokens), dtype=torch.int64, device=device)
        logprobs = torch.empty((self.batch, max_new_tokens), dtype=torch.float32, device=device)
        lengths = torch.full((self.batch,), max_new_tokens, dtype=torch.int64, device=device)
        stopped = torch.zeros(self.batch, dtype=torch.bool, device=device)
        produced = max_new_tokens
        for step in range(max_new_tokens):
            tokens[:, step], logprobs[:, step] = sample(logits, temperature, generator)
            if stops is not None:
                first = torch.isin(tokens[:, step], stops) & ~stopped
                lengths = torch.where(first, step + 1, lengths)
                stopped |= first
                if bool(stopped.all()):
                    produced = step + 1
                    break
            if step + 1 < max_new_tokens:
                logits = self.decode(tokens[:, step])
        return Generation(tokens[:, :produced].cpu(), logprobs[:, :produced].cpu(), lengths.cpu())
