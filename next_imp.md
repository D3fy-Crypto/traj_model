
Right now your sampler computes a progress-based **weight**, but that only says:

> “later prefixes matter more in the loss.”

It does **not** force the GRU hidden state to actually encode:

> “how far along the trajectory am I?”

Those are very different things. In your current code, progress affects sample weighting in the sampler, but there is no direct prediction target for progress. 

# 1. Why progress supervision matters

Your final VLM memory should help answer questions like:

* Have I only started the route?
* Am I midway through it?
* Am I near the end?
* Did I already complete the “go past the sofa and turn left” part of the instruction?

A plain InfoNCE prefix→future model can learn trajectory identity and future compatibility, but it does not necessarily organize the latent state by **temporal position**. Your current results already hint at this: early-prefix retrieval is much weaker than later-query retrieval, which means the model benefits heavily once enough path information accumulates. 

So progress supervision is useful because it pushes the hidden state to encode not just:

[
z_t \approx \text{which future matches?}
]

but also:

[
z_t \approx \text{where am I along this trajectory?}
]

That is much closer to the memory signal your downstream VLM needs.

# 2. Theoretical intuition

Suppose a trajectory has length (L_i), and the sampled prefix ends at timestep (t). Then a natural normalized progress target is:

[
p_t = \frac{t}{L_i}
]

This maps every prefix to a number in ([0,1]):

* near 0: beginning
* near 0.5: middle
* near 1: near the end

You then add a small head on top of the GRU embedding:

[
\hat{p}_t = g(z_t)
]

where:

* (z_t) is the GRU latent for prefix (1{:}t)
* (g) is a small MLP

and train it with a regression loss such as:

[
\mathcal{L}_{\text{progress}} = |\hat{p}_t - p_t|_2^2
]

or binary cross-entropy if you constrain output to ([0,1]) with a sigmoid.

The reason this helps is that it imposes a geometric structure on the latent space. Without this, two prefixes from very different temporal positions could still be close in embedding space if they imply similar future chunks. With progress supervision, the model is encouraged to separate “same direction but early” from “same direction but late.”

# 3. Why weighting is not enough

Your current sampler computes:

[
w(t) = w_{\min} + (1-w_{\min}) \cdot \text{progress}
]

and uses that to weight the InfoNCE loss. 

That does **not** mean the model knows progress.

It only means the optimizer pays more attention to certain samples.

Analogy:

* sample weighting is like telling the student which questions matter more
* progress prediction is like actually asking the student to tell you what chapter they are in

For your project, you want the second.

# 4. Concrete example

Take a simple trajectory:

```text
Forward, Forward, Left, Forward, Forward, Right, Forward, Stop
```

Say the full length is (L=8).

Now consider three prefixes:

### Prefix A

```text
Forward, Forward
```

Here (t=2), so:
[
p_t = 2/8 = 0.25
]

This means: early in the trajectory.

### Prefix B

```text
Forward, Forward, Left, Forward
```

Here (t=4), so:
[
p_t = 4/8 = 0.5
]

This means: middle.

### Prefix C

```text
Forward, Forward, Left, Forward, Forward, Right
```

Here (t=6), so:
[
p_t = 6/8 = 0.75
]

This means: later.

Now imagine the current image looks like a corridor corner. That image alone may be ambiguous. But if the trajectory memory says:

* image is ambiguous,
* hidden state predicts progress (\approx 0.75),

then the VLM can reason:

> I am probably not at the start of the instruction anymore. I have likely already completed the first turn and should now be in the latter phase of the route.

That is exactly the kind of structure you want.

# 5. Why this is especially good for your VLM integration

Later, when you project the GRU memory into Qwen3-VL or another VLM, the model will not decode your raw motion history explicitly. So you want the latent to already carry interpretable control-relevant factors.

Progress is one of the most valuable such factors because it helps connect:

* instruction decomposition
* path history
* current visual context
* next decision

In other words, progress supervision gives the memory token a stronger semantic meaning:
not just “this is some trajectory vector,” but “this is where I am in the unfolding route.”

# 6. Best target: absolute progress or bucketed progress?

You have two good options.

## Option A: Continuous progress

Target:
[
p_t = \frac{t}{L_i}
]

Predict:
[
\hat{p}_t \in [0,1]
]

Loss:
[
\mathcal{L}_{\text{progress}} = \text{MSE}(\hat{p}_t, p_t)
]

This is simple and usually good.

## Option B: Bucketed progress

Convert progress into bins:

* 0–0.33 = early
* 0.33–0.66 = middle
* 0.66–1.0 = late

Then predict a 3-way class.

This is easier to learn and often more stable if the exact numeric progress is noisy.

For your project, I would start with **continuous progress**, and optionally log bucket accuracy too.

# 7. Implementation plan

You already have all the ingredients. Your sampler already knows:

* prefix length (t)
* trajectory length (L_i)

So just return progress targets from the sampler.

## Step 1: modify sampler to return progress targets

Add a `progress_targets_list`.

Inside the loop where you append each prefix/future pair, compute:

[
p_t = \frac{t}{L_i}
]

### Code change

```python
def sample_prefix_future_pairs(
    sequences, lengths,
    min_prefix_len=1,
    min_future_len=5,
    max_future_len=25,
    max_prefixes_per_traj=None,
    w_min=0.3,
    return_traj_ids=True,
    return_progress=True,
):
    B, _, D = sequences.shape
    device = sequences.device

    prefix_seqs_list, future_seqs_list = [], []
    prefix_lengths_list, future_lengths_list, weights_list = [], [], []
    traj_ids_list = []
    progress_targets_list = []

    for i in range(B):
        L_i = lengths[i].item()

        future_len = random.randint(min_future_len, max_future_len)
        max_prefix_end = L_i - future_len

        if max_prefix_end < min_prefix_len:
            continue

        valid_ends = list(range(min_prefix_len, max_prefix_end + 1))
        if max_prefixes_per_traj:
            sampled_ends = random.sample(valid_ends, min(len(valid_ends), max_prefixes_per_traj))
        else:
            sampled_ends = valid_ends

        for t in sampled_ends:
            prefix_seqs_list.append(sequences[i, 0:t, :])
            prefix_lengths_list.append(t)

            future_end = min(t + future_len, L_i)
            future_seqs_list.append(sequences[i, t:future_end, :])
            future_lengths_list.append(future_end - t)

            progress = (t - min_prefix_len) / max(1, L_i - future_len - min_prefix_len)
            progress = max(0.0, min(1.0, progress))
            weight = w_min + (1 - w_min) * progress
            weights_list.append(weight)

            # New: absolute normalized progress target
            progress_target = t / max(1, L_i)
            progress_targets_list.append(progress_target)

            if return_traj_ids:
                traj_ids_list.append(i)

    if prefix_seqs_list:
        prefix_seqs = nn.utils.rnn.pad_sequence(prefix_seqs_list, batch_first=True, padding_value=0.0)
        future_seqs = nn.utils.rnn.pad_sequence(future_seqs_list, batch_first=True, padding_value=0.0)
        prefix_lengths = torch.tensor(prefix_lengths_list, dtype=torch.long, device=device)
        future_lengths = torch.tensor(future_lengths_list, dtype=torch.long, device=device)
        weights = torch.tensor(weights_list, dtype=torch.float32, device=device)
        traj_ids = torch.tensor(traj_ids_list, dtype=torch.long, device=device) if return_traj_ids else None
        progress_targets = torch.tensor(progress_targets_list, dtype=torch.float32, device=device)
    else:
        prefix_seqs = torch.zeros((0, 1, D), dtype=sequences.dtype, device=device)
        future_seqs = torch.zeros((0, 1, D), dtype=sequences.dtype, device=device)
        prefix_lengths = torch.zeros((0,), dtype=torch.long, device=device)
        future_lengths = torch.zeros((0,), dtype=torch.long, device=device)
        weights = torch.zeros((0,), dtype=torch.float32, device=device)
        traj_ids = torch.zeros((0,), dtype=torch.long, device=device) if return_traj_ids else None
        progress_targets = torch.zeros((0,), dtype=torch.float32, device=device)

    return (
        prefix_seqs,
        future_seqs,
        prefix_lengths,
        future_lengths,
        weights,
        traj_ids,
        progress_targets,
    )
```

## Step 2: add a progress head to the encoder

Your current `GRUEncoder` returns only the embedding. 
Change it so it can optionally return both:

* normalized embedding for InfoNCE
* raw hidden/projection features for progress prediction

### Cleaner design

```python
class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

        self.progress_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1)
        )

    def forward(self, sequences, lengths, return_progress=False):
        packed = nn.utils.rnn.pack_padded_sequence(
            sequences, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        hidden = hidden.squeeze(0)                       # [B, hidden_dim]

        feat = self.proj(hidden)                         # [B, embedding_dim]
        z = torch.nn.functional.normalize(feat, p=2, dim=1)

        if return_progress:
            progress_pred = torch.sigmoid(self.progress_head(feat)).squeeze(-1)
            return z, progress_pred

        return z
```

Why use `feat` instead of normalized `z` for progress head?
Because L2 normalization restricts scale and can slightly hurt regression. Use the pre-normalized projected feature for the auxiliary head.

## Step 3: define progress loss

Use MSE or L1. I would start with MSE:

```python
def progress_loss_fn(progress_pred, progress_target):
    return torch.nn.functional.mse_loss(progress_pred, progress_target)
```

## Step 4: combine with InfoNCE in training

In your `train_epoch`, after sampling, do:

```python
prefix_seqs, future_seqs, prefix_lens, future_lens, weights, traj_ids, progress_targets = \
    sample_prefix_future_pairs(
        batch_sequences,
        batch_lengths,
        min_prefix_len,
        min_future_len,
        max_future_len,
        max_prefixes_per_traj,
        w_min
    )

z_prefix, progress_pred = encoder(prefix_seqs, prefix_lens, return_progress=True)
z_future = encoder(future_seqs, future_lens)

loss_nce = infonce_loss(z_prefix, z_future, weights, temperature, traj_ids=traj_ids)
loss_progress = progress_loss_fn(progress_pred, progress_targets)

lambda_progress = 0.2
loss = loss_nce + lambda_progress * loss_progress
```

Good starting value:
[
\lambda_{\text{progress}} \in [0.1, 0.3]
]

I would start with `0.2`.

## Step 5: log progress metrics

Add:

* progress MSE
* progress MAE
* optionally bucket accuracy

Example:

```python
with torch.no_grad():
    progress_mae = torch.mean(torch.abs(progress_pred - progress_targets)).item()
```

Then log it to wandb.

# 8. Better version: progress buckets too

If you want a more interpretable metric, create buckets:

```python
def progress_to_bucket(progress):
    # progress in [0,1]
    # 0: early, 1: mid, 2: late
    bucket = torch.zeros_like(progress, dtype=torch.long)
    bucket[progress >= 1/3] = 1
    bucket[progress >= 2/3] = 2
    return bucket
```

Then add a classification head and log early/mid/late accuracy.

This is useful because your downstream reasoning often really only needs:

* early
* middle
* late

rather than exact 0.47 vs 0.52.

# 9. Example with your current notebook

Your notebook currently shows strong aligned retrieval but uneven handcrafted query performance, especially weak early-query behavior. 

Progress supervision may help because it tells the latent:

* not just “this future matches”
* but also “this prefix is still early”

So for an early prefix, instead of the latent being underdetermined, it can still encode a stable notion like:

> I am at the beginning of the trajectory; many futures remain possible.

That is useful later for the VLM, because the VLM can combine:

* early progress
* current image
* instruction

to reason more cautiously.

# 10. What improvement to expect

Be realistic:

Progress supervision will probably not magically solve everything. It will not by itself make Q1 perfect.

But it should help with:

* temporal organization of latent space
* early/mid/late separability
* downstream interpretability
* better memory tokens for VLM fusion

It is especially valuable because it is cheap to add and directly aligned with your project hypothesis.

# 11. Best practical recommendation

For your project, I would do this exact order:

First:

* add continuous progress regression head

Then:

* log progress MAE and 3-bin early/mid/late accuracy

Then:

* inspect whether Q1 improves and whether embedding visualization shows better temporal ordering

If it works, keep it for VLM integration.

# 12. Clean summary

What you have now:

* progress is used only as a loss weight

What you should add:

* progress as an explicit supervised target

Why:

* because your hidden state should encode not only which future matches, but where the agent is along the route

Implementation:

* return `progress_targets` from sampler
* add `progress_head` to encoder
* train with
  [
  \mathcal{L} = \mathcal{L}*{\text{NCE}} + \lambda \mathcal{L}*{\text{progress}}
  ]

This is one of the highest-value low-cost upgrades you can make to the current GRU memory module. 

I can next write the full patched version of your notebook cells for sampler, encoder, and train loop.
