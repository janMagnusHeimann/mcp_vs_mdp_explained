# Markov Decision Processes (MDPs) vs. Markov Reward Processes (MRPs) in Reinforcement Learning

Markov processes are foundational to Reinforcement Learning (RL) as they provide a mathematical framework for modeling sequential decision-making problems where an agent interacts with an environment. The core idea is the **Markov Property**: the future is independent of the past given the present. This means the current state $S_t$ captures all relevant information from the history.

This document explores two types of Markov processes crucial for RL: Markov Reward Processes (MRPs) and Markov Decision Processes (MDPs).

## 1. Markov Reward Processes (MRPs)

An MRP is a simple model of a stochastic process that generates sequences of states and rewards, but without any notion of actions or control by an agent. It's essentially a Markov chain augmented with a reward function.

### Definition
An MRP is a tuple $\mathcal{M} = \langle S, P, R, \gamma \rangle$, where:

* $S$: A finite set of states.
* $P$: A state transition probability matrix, where $P_{ss'} = P(S_{t+1}=s' \mid S_t=s)$ is the probability of transitioning from state $s$ to state $s'$ at the next time step.
* $R$: A reward function. $R_s = E[R_{t+1} \mid S_t=s]$ is the expected immediate reward received after transitioning out of state $s$. (Sometimes defined as $R(s,s')$ for transitioning from $s$ to $s'$)
* $\gamma$: A discount factor, $\gamma \in [0,1]$. It determines the present value of future rewards. A $\gamma$ close to 0 leads to "myopic" evaluation, while a $\gamma$ close to 1 leads to "far-sighted" evaluation.

### Goal
The primary goal in an MRP is to evaluate the "goodness" of each state. This is quantified by the state-value function.

### State-Value Function ($V(s)$)
The state-value function $V(s)$ for an MRP is the expected cumulative discounted reward starting from state $s$.
First, let's define the **return** $G_t$ at time $t$:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

The state-value function is then:

$$V(s) = E[G_t \mid S_t=s]$$

### Bellman Equation for MRPs
The Bellman equation provides a recursive definition for the value function in an MRP. It expresses the value of a state in terms of the expected immediate reward and the discounted values of successor states.

**Derivation:**
Starting with the definition of $V(s)$:

$$V(s) = E[G_t \mid S_t=s]$$
$$V(s) = E[R_{t+1} + \gamma G_{t+1} \mid S_t=s]$$

By linearity of expectation:

$$V(s) = E[R_{t+1} \mid S_t=s] + \gamma E[G_{t+1} \mid S_t=s]$$

The first term is the definition of our reward function $R_s$. For the second term, we can condition on the next state $S_{t+1}=s'$:

$$E[G_{t+1} \mid S_t=s] = \sum_{s' \in S} P(S_{t+1}=s' \mid S_t=s) E[G_{t+1} \mid S_t=s, S_{t+1}=s']$$

Due to the Markov property, $E[G_{t+1} \mid S_t=s, S_{t+1}=s'] = E[G_{t+1} \mid S_{t+1}=s'] = V(s')$.
So,

$$E[G_{t+1} \mid S_t=s] = \sum_{s' \in S} P_{ss'} V(s')$$

Substituting back, we get the Bellman equation for an MRP:

$$V(s) = R_s + \gamma \sum_{s' \in S} P_{ss'} V(s')$$

This equation states that the value of a state $s$ is the expected immediate reward $R_s$ plus the discounted expected value of the next state $s'$, averaged over all possible next states.

**Matrix Form:**
The Bellman equation can be written in matrix form. Let $V$ be a column vector of values for all states, and $R$ be a column vector of expected rewards for all states:

$$V = R + \gamma PV$$

This is a system of linear equations. It can be solved directly if the number of states is not too large:

$$(I - \gamma P)V = R$$
$$V = (I - \gamma P)^{-1} R$$

where $I$ is the identity matrix. The inverse $(I - \gamma P)^{-1}$ exists if $\gamma < 1$ or if all states eventually lead to a terminal state.

### Role in Reinforcement Learning
MRPs are often used to evaluate a given fixed policy in an MDP. When an agent follows a specific policy, the environment dynamics (from the agent's perspective) behave like an MRP, allowing for the calculation of state values under that policy.

## 2. Markov Decision Processes (MDPs)
MDPs extend MRPs by introducing actions, allowing an agent to influence state transitions and rewards. MDPs are the standard formalism for RL problems where an agent learns to make optimal decisions.

### Definition
An MDP is a tuple $\mathcal{M} = \langle S, A, P, R, \gamma \rangle$, where:

* $S$: A finite set of states.
* $A$: A finite set of actions available to the agent. $A(s)$ may denote the set of actions available in state $s$.
* $P$: A state transition probability function (or dynamics model), $P(s' \mid s,a) = P(S_{t+1}=s' \mid S_t=s, A_t=a)$, is the probability of transitioning to state $s'$ from state $s$ after taking action $a$.
* $R$: A reward function. This can be defined in a few ways:
    * $R(s,a,s') = E[R_{t+1} \mid S_t=s, A_t=a, S_{t+1}=s']$: Expected reward for transitioning from $s$ to $s'$ via action $a$.
    * $R(s,a) = E[R_{t+1} \mid S_t=s, A_t=a] = \sum_{s' \in S} P(s' \mid s,a) R(s,a,s')$: Expected reward for taking action $a$ in state $s$.
* $\gamma$: A discount factor, $\gamma \in [0,1]$.

### Goal
The goal in an MDP is to find an optimal policy $\pi^*(a \mid s)$ that maximizes the expected cumulative discounted reward from any starting state.

### Policy ($\pi$)
A policy $\pi$ is a mapping from states to a probability distribution over actions:

$$\pi(a \mid s) = P(A_t=a \mid S_t=s)$$

It defines the agent's behavior – how it chooses actions in each state. A policy can be deterministic ($\pi(s)=a$) or stochastic.

### Value Functions in MDPs
Because the agent's actions influence outcomes, we define value functions with respect to a policy $\pi$.

#### State-Value Function ($V_\pi(s)$):
The expected return when starting in state $s$ and subsequently following policy $\pi$.

$$V_\pi(s) = E_\pi[G_t \mid S_t=s]$$

The subscript $\pi$ indicates that the expectation is taken assuming the agent follows policy $\pi$.

#### Action-Value Function ($Q_\pi(s,a)$):
The expected return when starting in state $s$, taking action $a$, and thereafter following policy $\pi$.

$$Q_\pi(s,a) = E_\pi[G_t \mid S_t=s, A_t=a]$$

Q-values are often more directly useful for decision-making, as they tell us the value of taking a specific action in a state.

### Bellman Expectation Equations for MDPs
These equations relate the value of a state (or state-action pair) to the values of subsequent states (or state-action pairs) under a given policy $\pi$.

**Derivation for $V_\pi(s)$:**

$$V_\pi(s) = E_\pi[R_{t+1} + \gamma V_\pi(S_{t+1}) \mid S_t=s]$$

To expand this, we consider the actions taken according to policy $\pi$ and the subsequent state transitions:

$$V_\pi(s) = \sum_{a \in A} \pi(a \mid s) E_\pi[R_{t+1} + \gamma V_\pi(S_{t+1}) \mid S_t=s, A_t=a]$$

The expectation $E_\pi[\cdot \mid S_t=s, A_t=a]$ means we average over possible next states $s'$ given $s$ and $a$:

$$V_\pi(s) = \sum_{a \in A} \pi(a \mid s) \sum_{s' \in S} P(s' \mid s,a) (R(s,a,s') + \gamma V_\pi(s'))$$

Here, $R(s,a,s')$ is the expected reward when transitioning from $s$ to $s'$ under action $a$.
If we use the reward function $R(s,a) = \sum_{s' \in S} P(s' \mid s,a) R(s,a,s')$, which is the expected reward for taking action $a$ in state $s$, the equation simplifies to:

$$V_\pi(s) = \sum_{a \in A} \pi(a \mid s) \left( R(s,a) + \gamma \sum_{s' \in S} P(s' \mid s,a) V_\pi(s') \right)$$

**Derivation for $Q_\pi(s,a)$:**

$$Q_\pi(s,a) = E_\pi[R_{t+1} + \gamma Q_\pi(S_{t+1}, A_{t+1}) \mid S_t=s, A_t=a]$$

After taking action $a$ in state $s$, the environment transitions to $s'$ with probability $P(s' \mid s,a)$ and gives reward (expected) $R(s,a,s')$. In state $s'$, the agent takes action $a'$ according to $\pi(a' \mid s')$.
So, we sum over possible next states $s'$ and then over possible next actions $a'$:

$$Q_\pi(s,a) = \sum_{s' \in S} P(s' \mid s,a) \left( R(s,a,s') + \gamma E_\pi[Q_\pi(S_{t+1}, A_{t+1}) \mid S_{t+1}=s'] \right)$$

The term $E_\pi[Q_\pi(S_{t+1}, A_{t+1}) \mid S_{t+1}=s']$ is the expected Q-value from state $s'$ if we follow policy $\pi$. This is equivalent to $V_\pi(s')$.
Thus:

$$Q_\pi(s,a) = \sum_{s' \in S} P(s' \mid s,a) (R(s,a,s') + \gamma V_\pi(s'))$$

If we use the expected reward $R(s,a) = \sum_{s' \in S} P(s' \mid s,a) R(s,a,s')$, this becomes:

$$Q_\pi(s,a) = R(s,a) + \gamma \sum_{s' \in S} P(s' \mid s,a) V_\pi(s')$$

Alternatively, to express $Q_\pi(s,a)$ in terms of future Q-values, recall $V_\pi(s') = \sum_{a' \in A} \pi(a' \mid s') Q_\pi(s',a')$. Substituting this into the above equation for $Q_\pi(s,a)$:

$$Q_\pi(s,a) = R(s,a) + \gamma \sum_{s' \in S} P(s' \mid s,a) \sum_{a' \in A} \pi(a' \mid s') Q_\pi(s',a')$$

**Relationship between $V_\pi$ and $Q_\pi$**:
The two value functions under policy $\pi$ are closely related:

$$V_\pi(s) = \sum_{a \in A} \pi(a \mid s) Q_\pi(s,a)$$

This means the value of a state under policy $\pi$ is the expected value of the Q-values of all actions available in that state, weighted by the policy's probability of taking each action.
And conversely (as shown in the $Q_\pi$ derivation):

$$Q_\pi(s,a) = R(s,a) + \gamma \sum_{s' \in S} P(s' \mid s,a) V_\pi(s')$$

This means the value of taking action $a$ in state $s$ and then following policy $\pi$ is the expected immediate reward $R(s,a)$ plus the discounted expected value of the next state $s'$ (where $V_\pi(s')$ is the value of following $\pi$ from $s'$).


