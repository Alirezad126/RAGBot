# Profile notes

Written by hand to fill what the CV and the thesis leave out. The CV is current
but terse, and the thesis only covers the 2021 to 2023 energy hub work, so the
newer research, the project detail, and the everyday things live here. Each
section is meant to answer a question on its own.

## Well leakage risk at Concordia, the current project

Machine Learning Researcher at Concordia University in Montreal, October 2025
to the present. This is one of the two things Alireza works on today.

The subject is well integrity. Alberta's regulator has collected decades of
inspection records on surface casing vent flow and gas migration, which are
the two ways a well can leak to surface. The records are noisy, inconsistently
reported, and heavily zero-inflated, because the large majority of wells never
leak at all.

That shape is why the model is a hurdle model. A first stage classifies
whether a given well leaks, and a second stage regresses the leakage rate
using only the wells that do. Fitting one regression across everything would
be dominated by the zeros and would predict almost nothing useful.

The second half of the work is validation. He defines protocols that score the
predictions against the historical records, and running them exposed gaps in
the records themselves, such as inspections that are missing, inconsistent, or
recorded in ways that cannot be compared across years. Reporting those gaps was
as much the contribution as the model, and that is what the OMAE 2026 paper is
about.

## Differentiable predictive control at Johns Hopkins, the other current project

Machine Learning Researcher at Johns Hopkins University, remote, February 2025
to the present, working with Ján Drgoňa. This is the other of the two things
he works on today.

Model predictive control works well and is expensive, because it solves a
constrained optimization problem at every sampling instant, which rules it out
on fast systems or small hardware. Differentiable predictive control moves that
cost offline. The optimal control problem is treated as an unsupervised
learning problem, the objective is differentiated through a model of the
dynamics, and what comes out is an explicit policy that costs one forward pass
to evaluate.

His work is on the form that policy takes. Standard DPC uses a neural network,
which spreads the control law over thousands of coupled weights that nobody can
inspect or verify. His method, SD-DPC, replaces it with a sparse dictionary
expansion, which is a short sum of known functions with most coefficients
driven to zero.

He also builds benchmarks comparing MPC against learning-based controllers on
closed-loop performance and constraint satisfaction, and contributes
differentiable optimization components to NeuroMANCER, the PNNL library,
through pull requests and code review.

## SD-DPC, the sparse dictionary control method

The project at Johns Hopkins, submitted to the American Control Conference
2027 with Ján Drgoňa, arXiv 2610.02466.

It runs in two stages, both using the same sequentially thresholded gradient
procedure. First a sparse dynamics model is identified from multistep rollout
error, in the manner of SINDy but fitted to rollouts rather than to measured
state derivatives, so numerical differentiation error never enters. Then a
sparse policy is trained by differentiating a finite horizon control objective
through that identified model. Input constraints hold exactly through a
projection, and state constraints enter as differentiable penalties.

One detail mattered more than expected. The exact derivative of that projection
kills the gradient of any saturated input channel, so a relaxed surrogate is
used on the backward pass while the forward pass evaluates the true saturated
loop.

The results come from three benchmarks: constrained reference tracking on a
cascaded two-tank system, stabilizing a forced Van der Pol oscillator at its
unstable origin, and a two-dimensional integrator reaching a target around a
circular obstacle. Across all three the policy evaluates in 17 microseconds and
occupies under 0.1 kB, against about 5 kB for a neural DPC policy and 18 to 26
MiB allocated per online IPOPT solve. Every benchmark met its constraints.

On the two-tank system the learned policy keeps six terms and tracks about
twelve times more accurately than the neural policy, so sparsity cost nothing
there. On the Van der Pol oscillator the learned law is three terms long, short
enough that global asymptotic stability can be proved by hand with a Lyapunov
function built from the learned coefficients.

The sharpest result is the obstacle case, and it is an argument for training in
closed loop rather than imitating a good controller. A distilled policy using
exactly the same retained terms, fitted to a near-optimal teacher, hits the
obstacle in 60.8% of scenarios, while SD-DPC avoids it from every initial
condition with control effort within 0.6% of the IPOPT minimum. The imitation
error simply exceeds the margin the teacher leaves.

## The M.A.Sc. thesis on energy hub dispatch

His master's thesis at Concordia, carried out as a Research Assistant from May
2021 to April 2023, and published in the Journal of Energy Storage in 2022 with
Fuzhan Nasiri and Fariborz Haghighat. This work is finished, it is not what he
does now.

The system is a trigenerative energy hub with four supply networks and seven
conversion and storage units, including a micro turbine, a gas boiler, a heat
pump, an absorption chiller and a compressed air energy storage unit, which
together have to meet electrical, heating and cooling demand every hour.

The usual way to handle operational limits in reinforcement learning is to fold
violations into the reward as weighted penalties, which buries the trade-off
between cost and safety in weights somebody has to tune by hand. Push them up
and the agent keeps the storage idle to stay safe. Push them down and it saves
money by missing the heating balance.

So he posed it as a constrained Markov decision process instead. The objective
is operational cost alone, and four constraints get their own cost critics: the
storage pressure band, the grid connection rating, the cooling balance and the
heating balance. Each is priced by a dual variable updated by projected
gradient ascent, so a constraint that is satisfied sees its price decay to zero
while a violated one gets more expensive until respecting it is cheaper. Nothing
is hand-tuned except the dual step size. A useful side effect is that the
learned prices say which constraints actually bind: heating and cooling rose,
storage settled low, and the grid price stayed at zero because the rating was
never approached.

Exploration from scratch did not work, because most random dispatches break the
heating balance and the agent settles into never using the storage. So a mixed
integer linear program was solved in Pyomo for each of 292 training days under
perfect foresight, and those solutions were used three ways: to pre-train the
actor by behavioural cloning, to fill a protected block of the replay buffer
that is never overwritten, and as a regression term in the actor objective.

The storage itself is modelled off-design rather than with a fixed round-trip
efficiency, since compressor and expander efficiencies fall away from the
design point. Isentropic efficiencies and pressure ratios are polynomial fits
in load percentage, the tank is treated as a constant-temperature volume so
pressure follows from the mass balance, and the resulting nonlinear mass flows
are linearized before they enter the MILP.

The result is daily operational cost within 2.7% of the MILP optimum computed
with perfect foresight, with constraint violations near zero and the storage
still actively cycled. The code, covering the MILP, the off-design storage
model, the environment and the agent, is at github.com/Alirezad126/PD-DDPGfD.

## Building control with safe RL and DPC

A comparison released as a worked example in NeuroMANCER. He built a
physics-based building thermal environment in OpenAI Gym with box constraints
on indoor temperature and on the control inputs, then put two controllers
through it. The first is primal-dual DDPG, which learns a Lagrange multiplier
alongside the policy to enforce the constraints, carried over from the thesis
work. The second is a DPC policy trained by backpropagating through the
differentiable building model over a prediction horizon. Both were scored on
energy use and on constraint violations.

## Data science at IranBar

Data Scientist at IranBar in Tehran, June 2019 to May 2020, his first industry
role. The company ran freight quotation, and he shipped price-prediction models
into production there, which cut quotation time by 30% and raised agreement
rates by 25%. He built the pipelines feeding them with PySpark, pandas and SQL
over large freight datasets, improved bidding strategies with data-driven
models, and ran the modelling lifecycle on AWS SageMaker. He also kept the
deployed models alive with versioning, CI/CD, monitoring and scheduled
retraining. The title on some older material is Junior Data Scientist.

## Dompark energy system optimization

He translated the energy supply problem of the Dompark complex into a
mathematical program, with a decision variable for each unit's output,
energy-balance constraints on the electrical, heating and cooling loads, and
capacity limits per piece of equipment. Several candidate system
configurations were modelled and each was optimized over a full year of load
data in Pyomo, which identified the configuration with the lowest annual
operating cost and gave a quantitative basis for choosing how to supply the
complex.

## HVAC control with reinforcement learning and forecasts

He modelled a building and its HVAC system in EnergyPlus and wrapped the
simulation as an OpenAI Gym environment for closed-loop control. An LSTM was
trained to forecast ambient air temperature and solar irradiance, and those
forecasts were passed to the controller as inputs. A DQN agent then set the
supply air flow rate, trading energy consumption against indoor temperature
requirements. It cut energy use by 10% against a conventional rule-based
controller while keeping indoor temperature in range.

## Fault detection on an air handling unit

He framed fault detection and diagnosis for an air handling unit as supervised
classification on real operating data from a single-zone HVAC system. After
cleaning the sensor data and using exploratory visualization to see how faulty
and normal operation differ, he trained and compared Decision Tree, Random
Forest, SVM, KNN and Gaussian Naive Bayes classifiers, validated each with
k-fold cross-validation, and assessed them with ROC curves and AUC to pick the
best performer on held-out folds.

## The chatbots and the smaller builds

This chatbot. Retrieval-augmented generation over his CV, his thesis and these
notes. The index is built once when the container image is built, the
embeddings run on CPU inside AWS Lambda so retrieval calls no API, and an
open-weight model on Groq writes the answer. An earlier version used OpenAI
embeddings with a Chroma vector store behind FastAPI.

RAG-GPT, a document chatbot on the portfolio. A visitor uploads a PDF and asks
questions about it, and the relevant passages are retrieved before the model
answers. It runs on AWS Lambda.

A reinforcement learning agent playing Snake, June 2022. A deep Q-network
integrated with a Pygame application and deployed through AWS Lambda, which
still runs live from the portfolio.

An image captioning app, June 2023. A VGG encoder with an LSTM decoder, wrapped
in Streamlit and deployed on an AWS EC2 instance.

## Education and contact

M.A.Sc. in Building Engineering, Concordia University, Montreal, January 2021
to April 2023. GPA 4.1 out of 4.3, and the thesis was recognized as an
Outstanding Thesis.

B.A.Sc. in Mechanical Engineering, Iran University of Science and Technology,
Tehran, September 2016 to August 2020. Ranked 5th in his cohort, GPA 17.45 out
of 20.

He lives in Hamilton, Ontario, Canada. Email alireza.dg1998@gmail.com, phone
+1 514 515 3695, code at github.com/Alirezad126, and LinkedIn at
linkedin.com/in/ali-reza-daneshvar-garmroodi. Full name Ali Reza Daneshvar
Garmroodi. Born 9 April 1998.

## Publications

SD-DPC: Sparse Dictionary Differentiable Predictive Control, with Ján Drgoňa,
submitted to the American Control Conference 2027, arXiv 2610.02466.

Learning Safe Control Policies: Reinforcement Learning-Enabled Differentiable
Predictive Control, with T. Banker, N. Lawrence, J. Drgoňa and A. Mesbah, IFAC
World Congress 2026. The collaborators are at UC Berkeley and PNNL.

From Reports to Risk Assessment: Insights and Gaps in Alberta's SCVF and GM
Records, ASME OMAE 2026.

Optimal Dispatch of an Energy Hub with Compressed Air Energy Storage: A Safe
Reinforcement Learning Approach, with F. Nasiri and F. Haghighat, Journal of
Energy Storage 2022, doi 10.1016/j.est.2022.106147.

## Personal

He sings, games (Dota and Valorant), works out, and is happy in company.
Radiohead is his favourite band.

## Tools he has used

Listed only so the names are searchable. These are not an answer to what he is
working on, and no project should be described using them unless the project's
own section says so.

Python, C++, Bash, SQL, JavaScript. Pyomo, Gurobi, CasADi, IPOPT, NeuroMANCER,
EnergyPlus. PyTorch, TensorFlow, scikit-learn, Hugging Face, OpenCV, OpenAI
Gym. pandas, NumPy, PySpark, PostgreSQL, Snowflake, Azure Databricks. MLflow,
Docker, FastAPI, Flask, Git, GitHub Actions, AWS, CI/CD. React, HTML5,
Tailwind CSS. Mathematical programming, MILP, MPC, differentiable predictive
control, safe reinforcement learning, time-series forecasting, system
identification, transformers.
