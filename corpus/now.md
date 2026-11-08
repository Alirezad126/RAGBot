# Right now

This is the current state of things, as of November 2026. It is prepended to
every answer, because "what are you working on" is the question visitors ask
most and it must never depend on what retrieval happened to return.

Alireza holds two research positions at the same time.

At Concordia University in Montreal, as a Machine Learning Researcher since
October 2025, he works on well leakage risk. Alberta's regulator holds decades
of well-integrity inspection records covering surface casing vent flow and gas
migration. Most wells never leak, and the ones that do leak at wildly
different rates, so he models the problem with a hurdle model: a classifier
decides whether a well leaks at all, and a regressor predicts the rate only
for the wells that do. He also writes the validation protocols that check
those predictions against the historical records, and doing that surfaced real
gaps in the underlying data, which is itself a finding. The work was presented
at OMAE 2026 as "From Reports to Risk Assessment: Insights and Gaps in
Alberta's SCVF and GM Records".

At Johns Hopkins University, remotely, as a Machine Learning Researcher since
February 2025, he works on differentiable predictive control with Ján Drgoňa.
The idea is to move the cost of model predictive control offline: instead of
solving a constrained optimization at every timestep, train a policy by
differentiating the control objective through a model of the system, so that
at run time a decision is one forward pass. His own contribution is SD-DPC,
which replaces the usual neural network policy with a sparse dictionary, a
short explicit sum of known functions, so the resulting control law is short
enough to read and prove things about. He also contributes differentiable
optimization components to NeuroMANCER, the PNNL library, through pull
requests and code review.

If someone asks what he does now, these two are the answer. The compressed air
energy storage work is his 2021 to 2023 master's thesis and is finished.
