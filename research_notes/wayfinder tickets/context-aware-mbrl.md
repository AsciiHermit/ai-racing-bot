# Context-Aware MBRL paper (arXiv:2510.11501) — architecture extraction

Ticket: "Extract architecture details from the Context-Aware MBRL paper" — GitHub issue [#7](https://github.com/AsciiHermit/ai-racing-bot/issues/7).

Source: Moustafa & Dusparic, "Context-Aware Model-Based Reinforcement Learning for
Autonomous Racing," arXiv:2510.11501 (submitted Oct 13, 2025). Primary sources used:
- Abstract page: https://arxiv.org/abs/2510.11501
- Full text: https://ar5iv.labs.arxiv.org/html/2510.11501 (parsed cleanly and was the
  main source for everything below)
- PDF (https://arxiv.org/pdf/2510.11501): fetched but did not parse into readable
  text/tables via WebFetch (came back as a raw compressed binary stream); attempted
  to render it locally with the Read tool but `pdftoppm`/poppler is not installed in
  this environment, so the PDF route was abandoned in favor of ar5iv, per the
  fallback instruction in the ticket.

**Headline finding before the details: the premise in the earlier research report
does not match this paper.** This paper does not implement an RMA-style
(Rapid-Motor-Adaptation) privileged-context module, and its context vector is not
about track curvature, friction, or weather. See "Gaps" and "Implications" below.

## Cited findings

### 1. The privileged-context vector

It is a **fixed-size, 2-element vector**, `c = [c_v, c_θ]`, sampled once at the start
of each episode and held constant for the episode's duration:

> "A two-element context vector c=[cv,cθ] is initialized at the beginning of each
> episode and remains unchanged throughout the duration of the episode." (Section
> IV-B)

It does **not** encode track curvature, friction coefficient, weather, or any
environment/physics property. It parameterizes the **behavior of the opponent
("adversary") vehicles**:
- `c_v` scales adversary speed magnitude via a factor `1 + λ_v·c_v`.
- `c_θ` scales the pure-pursuit lookahead distance used for adversary steering, via
  `1 + λ_θ·c_θ` (Section IV-B).

There is no lookahead horizon in the RMA sense (no N-steps-ahead track information)
— the context is a per-episode scalar pair, not a time-indexed sequence. During
training, `c_v, c_θ` are each sampled from `[-0.15, +0.15]`; during evaluation they
are grid-searched over `[-0.3, 0.3]` in steps of 0.1 (Section IV-E), which is how the
paper defines "in-distribution" vs. "out-of-distribution" adversary behavior.

### 2. The adaptation module's architecture

**There is no online adaptation module that infers context from observation
history.** The context vector is treated as **fully observable and given** at every
step of both training and evaluation — it is not estimated from proprioceptive/
observation history by any network (no MLP/RNN/transformer history encoder exists
in this paper). What the paper instead adds is a **gating network**, part of its
proposed method "cMask":

> "uses a SAC network to predict an attenuating mask mt that is multiplied
> element-wise with the context before it is concatenated with the model state."
> (Section III-B)

So the learned network's job is to decide *how much of the already-known context to
let through* (`m_t ~ π_φ(m_t | o_t)`, a mask in `{0,1}` per context element,
conditioned on the current observation `o_t`), not to *infer the context's value*.
This is architecturally different from an RMA adaptation module, which estimates an
unobserved privileged latent from a history of ordinary sensor readings.

### 3. The fusion point

Context (masked, for cMask) is concatenated into the world model's state, following
the same overall scheme as the prior work cMask builds on (cRSSM, see below):

> "The world model equations are defined in a similar manner as the original cRSSM
> paper [9], however, the context vector c is replaced with a masked context c^m."
> (Section III-B)

cRSSM itself is **not proposed in this paper** — it's a prior method the paper
benchmarks against and builds cMask on top of:

> "Prasanna et al. [9] proposed a context-aware extension of the RSSM structure,
> coined cRSSM." (Section II-A; reference [9] = Prasanna et al., "Dreaming of many
> worlds: Learning contextual world models aids zero-shot generalization,"
> arXiv:2403.10967.)

I could not get either ar5iv or the PDF to yield the exact equation specifying
*which* internal RSSM tensor the (masked) context is concatenated with — i.e.
whether it's appended to the encoder's observation embedding, to the deterministic
recurrent state `h_t`, to the stochastic state `z_t`, or to the input of the
recurrent/dynamics model step. The paper describes it only as being "concatenated
with the model state" (Section III-B) and defers the precise formulation to the
cRSSM paper [9] it's extending. This is a genuine extraction gap — see "Gaps."

The mask-predicting SAC network conditions on the raw observation `o_t` (not on the
world-model's latent state), and a separate SAC critic `v_φ(G_t | o_t, m_t)` is used
to train it (Section III-B) — so the mask/gating decision sits alongside the world
model, taking the observation as input, rather than being fused inside the RSSM
itself.

### 4. Training procedure

**Single-phase, joint/end-to-end training — not RMA-style two-phase distillation.**
There is no "train with privileged info exposed to a teacher policy, then distill an
adaptation module separately" structure anywhere in the paper (and no adaptation
module exists to distill into, per point 2). Instead:

> "The SAC model is trained using the extrinsic reward signal from the environment,
> rt." (Section III-B)

Agents (world model + policy + mask network, for cMask) are trained together for a
fixed budget:

> "Agents are trained for 100k steps before being evaluated." (Section IV-E)

The document does not specify whether the SAC mask network and the DreamerV3 world
model/actor-critic update on the same step or in alternating passes within that
training loop — this finer-grained detail was not extractable from ar5iv.

### 5. Reported results

**Yes — got the numeric table this time.** This was the specific gap flagged in the
earlier pass. Table II (Section V-A) reports mean performance across two tracks
(ESP, GBR), each under 1-adversary and 3-adversary conditions, each further split
into in-distribution (ID) and out-of-distribution (OOD) adversary behavior, for
three methods: **DreamerV3** (context-free baseline), **cRSSM** (prior context-aware
baseline [9]), and **cMask** (this paper's proposed method). Metrics: Track
Progress (PG, higher better), Overtakes (OT, higher better), Agent-to-Agent
collisions (A2A, lower better).

ESP track:

| Method | Condition | PG↑ | OT↑ | A2A↓ |
|---|---|---|---|---|
| DreamerV3 | ID, 1 adv | 0.5515 | 0.6231 | 17.11 |
| cRSSM | ID, 1 adv | 0.5841 | 0.4867 | 9.38 |
| cMask | ID, 1 adv | 0.4300 | 0.2120 | 26.64 |
| DreamerV3 | OOD, 1 adv | 0.5255 | 0.4096 | 18.59 |
| cRSSM | OOD, 1 adv | 0.4888 | 0.3484 | 14.66 |
| cMask | OOD, 1 adv | 0.4175 | 0.2113 | 23.41 |
| DreamerV3 | ID, 3 adv | 0.5479 | 0.0667 | 18.47 |
| cRSSM | ID, 3 adv | 0.5664 | 0.0658 | 20.29 |
| cMask | ID, 3 adv | 0.6331 | 0.0867 | 14.67 |
| DreamerV3 | OOD, 3 adv | 0.4416 | 0.2038 | 19.27 |
| cRSSM | OOD, 3 adv | 0.4732 | 0.0538 | 20.35 |
| cMask | OOD, 3 adv | 0.5335 | 0.0759 | 18.41 |

GBR track:

| Method | Condition | PG↑ | OT↑ | A2A↓ |
|---|---|---|---|---|
| DreamerV3 | ID, 1 adv | 0.5057 | 0.0080 | 10.11 |
| cRSSM | ID, 1 adv | 0.4740 | 0.0342 | 10.64 |
| cMask | ID, 1 adv | 0.5993 | 0.0474 | 9.62 |
| DreamerV3 | OOD, 1 adv | 0.4446 | 0.4096 | 14.27 |
| cRSSM | OOD, 1 adv | 0.4436 | 0.0542 | 12.38 |
| cMask | OOD, 1 adv | 0.5463 | 0.0356 | 13.27 |
| DreamerV3 | ID, 3 adv | 0.6243 | 0.0529 | 15.04 |
| cRSSM | ID, 3 adv | 0.6419 | 0.0036 | 3.69 |
| cMask | ID, 3 adv | 0.5395 | 0.0387 | 12.80 |
| DreamerV3 | OOD, 3 adv | 0.5309 | 0.1217 | 19.13 |
| cRSSM | OOD, 3 adv | 0.5481 | 0.2820 | 10.54 |
| cMask | OOD, 3 adv | 0.4924 | 0.0395 | 14.89 |

(Values above are as extracted by the fetch tool from ar5iv's rendering of Table
II; I was not able to independently re-render the source LaTeX/PDF table to
byte-for-byte confirm formatting such as decimal precision beyond what's shown, so
treat trailing-digit precision as approximate even though the values themselves
come from the table.)

**PPO and TD-MPC are not in this comparison at all** — despite the earlier report's
framing that the paper was "benchmarked against Dreamer, PPO, SAC, TD-MPC." SAC
*was* tried but was dropped from the main comparison because it performed too
poorly to be a useful baseline:

> "the SAC achieves less than 25% track progress in both tracks... This render[s]
> SAC as a nonviable method." (Section V-A)

So the real comparison set is DreamerV3 vs. cRSSM vs. cMask only. cMask's results
are mixed rather than uniformly better: it wins on PG in several ID/3-adversary and
OOD/3-adversary conditions (e.g., ESP ID 3-adv: 0.6331 vs 0.5664/0.5479; GBR OOD
1-adv: 0.5463 vs 0.4436/0.4446) but underperforms both baselines on the ESP 1-adversary
conditions (PG 0.43/0.42 vs. ~0.49–0.58) and has the worst A2A collision rate in
several ESP rows. The abstract's claim that "cMask displays strong generalization
capabilities... relative to other context-aware MBRL approaches when racing against
adversaries with in-distribution behaviors" is best supported by the 3-adversary ID
rows, not universally across all rows.

## Gaps

- **The paper's actual subject does not match the earlier report's premise.** The
  earlier report characterized this paper as adding an "RMA-style privileged-context
  adaptation module" with context plausibly covering things like track curvature,
  friction, or weather, inferred online from ordinary observations. None of that is
  in this paper: the context is a 2-scalar adversary-behavior parameterization, it
  is never inferred (it's given/privileged at all times, training and eval alike),
  and there is no adaptation-module network of any kind (no MLP/RNN/transformer
  history encoder). I could not find a reading of the paper that reconciles this —
  the mismatch looks like it originates in the earlier report's summarization, not
  in something the primary source states ambiguously.
- **No RMA / "Rapid Motor Adaptation" terminology or citation anywhere in the
  paper.** Checked explicitly; also checked for "privileged information," "online
  adaptation," "domain randomization adaptation" — none present.
- **Exact fusion tensor inside the RSSM is unconfirmed.** The paper states the
  (masked) context is "concatenated with the model state" and otherwise follows
  cRSSM [9]'s equations, but neither ar5iv's rendering nor the raw PDF gave me the
  specific equation naming the encoder embedding, `h_t`, or `z_t` as the
  concatenation point. Resolving this precisely would require reading Prasanna et
  al. 2024 (arXiv:2403.10967, cRSSM) directly, which was out of scope for this
  ticket (that's the cited *prior* paper, not the one this ticket asked to extract).
- **Whether the SAC mask network and the DreamerV3 world model/actor-critic train
  simultaneously each step or in alternating passes** is not stated explicitly in
  the accessible text — only that both are trained across the same 100k-step budget
  per agent (Section IV-E) and that the SAC module uses extrinsic environment reward
  (Section III-B).
- **PDF route failed as anticipated by the ticket's fallback instructions.**
  `https://arxiv.org/pdf/2510.11501` fetched only a raw compressed binary stream via
  WebFetch, and rendering it locally with the Read tool failed because this
  environment doesn't have `pdftoppm`/poppler installed. ar5iv's HTML rendering
  (`https://ar5iv.labs.arxiv.org/html/2510.11501`) was used instead and parsed
  cleanly, per the ticket's suggested fallback — this is what makes finding #5
  possible where the earlier pass couldn't get it.
- **Did I get the numeric results table this time? Yes**, in full (Table II, both
  tracks, both adversary counts, both distribution conditions) — this was the
  specific gap the earlier report flagged. It came from ar5iv's Section V-A, not
  the PDF.

## Implications for the RMA-design ticket

This paper cannot be cited as prior art that "already does" RMA-style
privileged-context adaptation for the racing bot's Phase 5 plan, because it doesn't
contain an adaptation module, an inferred context, or a track/physics-conditioned
context vector at all — it's a contextual-MDP framing of *opponent* behavior
variation, with a learned attention/gating mechanism (cMask) over an
always-observable 2-scalar context, benchmarked against DreamerV3 and cRSSM (a
different, earlier group's context-concatenation baseline). If the project still
wants an RMA-style two-phase (privileged-teacher-then-distilled-adaptation-module)
design for track curvature/friction/weather context on top of a Dreamer-style
world model, that design has to be worked out from the original RMA robotics papers
(Kumar et al.) and adapted from scratch, rather than following this paper's
architecture — the two problems (inferring unobservable environment/physics
properties vs. having a fully-known scalar pair select an opponent's behavior
profile) are different enough that the fusion mechanics here (SAC-predicted
element-wise mask over a 2-vector) don't transfer directly to a track-lookahead or
friction-estimation use case. What *might* still be useful from this paper for
Phase 5: the general pattern of concatenating a context vector into a cRSSM-style
world model, and cRSSM's underlying paper (Prasanna et al., arXiv:2403.10967) is
probably the more relevant primary source to re-examine for actual RMA-adjacent
fusion mechanics, since this paper explicitly builds on and defers to it rather than
re-deriving it.
