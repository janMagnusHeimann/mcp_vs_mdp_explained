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

Optimal Value Functions (V 
∗
 (s) and Q 
∗
 (s,a))
The ultimate goal in an MDP is to find an optimal policy π 
∗
  that achieves the highest possible expected return. An optimal policy is one that is better than or equal to all other policies. There is always at least one such policy.

Associated with an optimal policy are the optimal state-value function V 
∗
 (s) and the optimal action-value function Q 
∗
 (s,a).

Optimal State-Value Function (V 
∗
 (s)):
V 
∗
 (s) is the maximum expected return achievable from state s.

V 
∗
 (s)= 
π
max
​
 V 
π
​
 (s)
Optimal Action-Value Function (Q 
∗
 (s,a)):
Q 
∗
 (s,a) is the maximum expected return achievable by taking action a in state s and thereafter following an optimal policy.

Q 
∗
 (s,a)= 
π
max
​
 Q 
π
​
 (s,a)
The relationship between V 
∗
  and Q 
∗
  is fundamental:
If we know Q 
∗
 (s,a), we can determine V 
∗
 (s) because an optimal policy will choose the action that maximizes Q 
∗
 (s,a):

V 
(
 s,a)
And if we know V 
∗
 (s), we can express Q 
∗
 (s,a) as the expected reward for taking action a in state s plus the discounted optimal value of the next state:

Q 
(
 s 
′
 )

(assuming R(s,a) is the expected reward for taking action a in state s)

Or, using R(s,a,s 
′
 ):


Q 
(
 s 
′
 ))
Bellman Optimality Equations
The Bellman optimality equations are a system of non-linear equations that define the optimal value functions. They state that the value of a state under an optimal policy must equal the expected return for the best action from that state.

Bellman Optimality Equation for V 
∗
 (s):
This equation expresses V 
∗
 (s) in terms of the optimal values of successor states V 
∗
 (s 
′
 ). It essentially says that the optimal value of a state is obtained by choosing the action that maximizes the sum of the immediate reward and the discounted optimal value of the next state.

V^(s') \right)
Alternatively, using Q 
∗
 (s,a):


V 
(
 s,a)
Bellman Optimality Equation for Q 
∗
 (s,a):
This equation expresses Q 
∗
 (s,a) in terms of the optimal values of future state-action pairs Q 
∗
 (s 
′
 ,a 
′
 ). After taking action a in state s and moving to state s 
′
 , the agent will then choose the action a 
′
  that maximizes Q 
∗
 (s 
′
 ,a 
′
 ) in state s 
′
 .

Q 
(
 s 
′
 ,a 
′
 )
If we use V 
∗
 (s 
′
 ):


Q 
(
 s 
′
 )

where V 
(
 s 
′
 )=max 
a 
′
 ∈A(s 
′
 )
​
 Q 
(
 s 
′
 ,a 
′
 ).

Unlike the Bellman expectation equations (which are linear), the Bellman optimality equations involve the max operator, making them non-linear. There is no closed-form solution in general (like the matrix inversion for MRPs or for V 
π
​
 ). Instead, iterative solution methods are used.

Optimal Policy (π 
∗
 (s))
Once we have found the optimal value functions (V 
∗
  or Q 
∗
 ), we can easily determine an optimal policy π 
∗
 .
If we have Q 
∗
 (s,a), an optimal policy is to choose the action a that maximizes Q 
∗
 (s,a) in state s:

π 
(
 s,a)
This is a deterministic policy. If there are multiple actions that maximize Q 
∗
 (s,a), any of them can be chosen.

If we only have V 
∗
 (s), we can find an optimal policy by one-step lookahead:

\pi^(s') \right)
This is equivalent to saying:


π 
(
 s,a)
The existence of an optimal policy is guaranteed in finite MDPs. Any policy that is greedy with respect to the optimal value functions V 
∗
  or Q 
∗
  is an optimal policy.

Solving MDPs
Finding the optimal policy often involves finding the optimal value functions. Common algorithms include:

Value Iteration: Iteratively applies the Bellman optimality equation for V 
∗
 (s) (or Q 
∗
 (s,a)) to update value estimates until they converge.


V 
k+1
​
 (s)= 
a∈A(s)
max
​
 (R(s,a)+γ 
s 
′
 ∈S
∑
​
 P(s 
′
 ∣s,a)V 
k
​
 (s 
′
 ))
Policy Iteration: Alternates between two steps:

Policy Evaluation: Given a policy π, compute V 
π
​
 (s) (solve the Bellman expectation equation, e.g., by iterating V 
k+1
π
​
 (s)=∑ 
a
​
 π(a∣s)(R(s,a)+γ∑ 
s 
′
 
​
 P(s 
′
 ∣s,a)V 
k
π
​
 (s 
′
 ))).

Policy Improvement: Improve the policy by acting greedily with respect to V 
π
​
 (s):


\pi'(s) = \text{argmax}{s' \in S} P(s' \mid s,a) V_\pi(s') \right)

This process is guaranteed to converge to an optimal policy π 
∗
 .

Q-learning: A model-free RL algorithm that directly estimates Q 
∗
 (s,a) without needing the transition probabilities P or reward function R explicitly.

3. Key Differences: MRPs vs. MDPs
Feature

Markov Reward Process (MRP)

Markov Decision Process (MDP)

Definition

⟨S,P,R,γ⟩

⟨S,A,P,R,γ⟩

Agent's Role

Passive observer; no actions or control.

Active agent; chooses actions to influence transitions and rewards.

Components

States, Transition Probabilities, Reward Function, Discount Factor.

States, Actions, Transition Probabilities, Reward Function, Discount Factor.

Transition Prob.

P(S 
t+1
​
 =s 
′
 ∣S 
t
​
 =s)

P(S 
t+1
​
 =s 
′
 ∣S 
t
​
 =s,A 
t
​
 =a)

Reward Function

R 
s
​
 =E[R 
t+1
​
 ∣S 
t
​
 =s]

R(s,a) or R(s,a,s 
′
 )

Goal

Evaluate states (calculate V(s)).

Find optimal policy π 
∗
 (a∣s) to maximize rewards.

Value Functions

State-Value Function V(s).

State-Value Function V 
π
​
 (s), Action-Value Function Q 
π
​
 (s,a). Optimal versions: V 
∗
 (s), Q 
∗
 (s,a).

Bellman Equations

Bellman Equation for V(s) (linear).

Bellman Expectation Equations for V 
π
​
 (s),Q 
π
​
 (s,a) (linear). Bellman Optimality Equations for V 
(
 s),Q 
(
 s,a) (non-linear).

Policy

No concept of a policy (or a fixed, implicit one).

Agent learns a policy π(a∣s).

Primary Use in RL

Evaluating a fixed policy (policy evaluation).

Modeling and solving decision-making problems (control).

4. Conclusion and Role in Reinforcement Learning
MRPs and MDPs are fundamental building blocks for understanding and developing RL algorithms.

MRPs provide the tools to analyze the value of states when the system's dynamics (or the agent's behavior) are fixed. This is crucial for the policy evaluation step in many RL algorithms, where we want to determine how good a particular policy is.

MDPs extend this framework by incorporating actions, allowing an agent to learn how to behave optimally. They form the mathematical basis for most RL problems, providing a formal specification of the interaction between an agent and its environment. The goal in an MDP is typically to solve the control problem: finding an optimal policy.

Algorithms like Value Iteration and Policy Iteration directly solve MDPs when a model of the environment (transition probabilities and rewards) is known. Model-free RL algorithms, such as Q-learning and SARSA, aim to learn optimal policies in MDPs even when the model is unknown, by learning value functions or policies directly from experience.

Understanding the distinction and relationship between MRPs and MDPs, along with their respective value functions and Bellman equations, is essential for anyone delving into the theory and practice of Reinforcement Learning.
