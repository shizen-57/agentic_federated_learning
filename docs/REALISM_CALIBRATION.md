# Realism calibration: what this simulator is, and isn't

This document exists to answer one question honestly: **is this a real-world
experiment?** No. It is a simulation, and every result in this repository must
be reported as such. What this document tracks is a different, legitimate
question: which parts of the simulator are grounded in real, cited
measurements of actual FL deployments, versus which parts are simplifying
assumptions -- so the thesis can state precisely what "realistic" means here
and not overclaim it.

This is exactly the same posture every FL paper cited in this thesis's Related
Works takes. None of Oort, FedScale, FedProx, FedNova, DivFL, or any other
paper this thesis compares against ran their headline experiments on a real
fleet of physical phones either -- they ran trace-driven or parametrically
calibrated simulations, same as this one, and reported them as such. That is
the accepted standard in this field, not a shortcut specific to this project.

## What is calibrated against a cited real-world source, and how

### 1. Diurnal (day/night) client availability

**Before**: every client had the same flat probability of being available
every round, drawn as an independent coin flip
(`rng.random(N) < availability_p`) with no dependence on time at all.

**Real-world finding it now matches**: Hard et al., "Applied Federated
Learning: Improving Google Keyboard Query Suggestions" (2018,
[arXiv:1812.02903](https://arxiv.org/pdf/1812.02903)) -- a real, deployed FL
system (Gboard). Devices are only eligible to train when charging, idle, and
on an unmetered network, which concentrates eligible devices overnight; the
paper reports round completion is "nearly flat" around noon, with progress
much faster at night, and that the cycle's phase, shape, and amplitude differ
per device (timezone, demographics, usage pattern).

**Implementation** (`src/afl/sim.py`, `ClientEnv.availability_prob`): each
client gets its own peak-availability hour, drawn from a normal distribution
clustered around a shared overnight window (`Normal(3am, 3h)`, wrapped to
24h) rather than a uniform random hour -- matching "most devices in one
deployment charge overnight in their own local time, with modest per-client
spread," not a totally random phase (a uniform spread was tried first and
rejected: it makes individual clients' cycles cancel out in the population
average, erasing the very midday dip the source reports). Each client also
gets its own amplitude scale. A configurable `rounds_per_day` (default 96,
i.e. ~15 min/round) maps the round index to an hour-of-day.

**Verified**: population-mean availability swings from ~0.98 near the peak
hour to ~0.58 at the opposite point of the cycle (see
`tests/test_core.py::test_availability_follows_a_diurnal_cycle`), a clear,
non-flat cycle rather than noise around a constant.

### 2. Heavy-tailed device/network heterogeneity

**Real-world basis**: FedScale (Lai et al., ICML 2022,
[arXiv:2105.11367](https://arxiv.org/pdf/2105.11367)) and Oort (Lai et al.,
OSDI 2021, [arXiv:2010.06081](https://arxiv.org/pdf/2010.06081)) are the
standard trace-driven FL simulators in this research area; both calibrate
per-client compute speed and network conditions from real measured device
traces (built on AI Benchmark's smartphone inference-speed measurements)
rather than treating clients as identical.

**Implementation**: this simulator already used a 3-tier, heavy-tailed device
model (`ClientEnv.__init__`: low/mid/high speed, bandwidth, and reliability
tiers with continuous within-tier jitter) before this pass -- the change here
is documentation, not new code: the module docstring now names the real
methodological precedent (trace-driven calibration) this qualitatively
follows, rather than presenting the tiered heterogeneity as an arbitrary
choice.

## What is NOT independently calibrated (stated plainly, not hidden)

- The **exact numeric values** of the speed/bandwidth/reliability tiers (e.g.,
  the specific multipliers 0.35/1.0/2.2) are not drawn from a published trace
  file -- FedScale's actual device trace was not available to replay in this
  environment (no network access to download it). They are a heavy-tailed
  approximation *in the same qualitative shape* as what such traces show, not
  a numeric match to any specific published table.
- The dataset partition (MNIST, Dirichlet-alpha non-IID split) is an
  artificial non-IID construction, standard in the FL literature but weaker
  than a "naturally" federated dataset like LEAF's FEMNIST (partitioned by
  actual distinct human writers). Switching to FEMNIST would be a stronger
  real-world claim and is a reasonable next step, not done here.
- The acceptance/rejection mechanics (staleness-and-norm rule, upload/link
  failure rates) are simplifying assumptions consistent with the thesis's own
  formulation (Eq. 3.18), not independently validated against a measured
  failure-rate distribution.

## How to describe this in the thesis (Chapter 4/5, and the limitations section)

Accurate: *"the simulation environment's device heterogeneity and diurnal
availability pattern are qualitatively calibrated against real FL deployment
measurements (Hard et al. 2018; the FedScale/Oort trace-driven methodology),
consistent with standard practice in this research area."*

Not accurate, and must not be written: *"experiments were conducted in a
real-world setting"* or *"on real devices."* Every result in this repository
is from this calibrated simulator, and must be reported as such.
