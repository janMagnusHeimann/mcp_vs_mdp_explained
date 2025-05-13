Markov Decision Processes (MDPs) vs. Markov Reward Processes (MRPs) in Reinforcement Learning
Markov processes are foundational to Reinforcement Learning (RL) as they provide a mathematical framework for modeling sequential decision-making problems where an agent interacts with an environment. The core idea is the Markov Property: the future is independent of the past given the present. This means the current state S 
t
​
  captures all relevant information from the history.

This document explores two types of Markov processes crucial for RL: Markov Reward Processes (MRPs) and Markov Decision Processes (MDPs).

1. Markov Reward Processes (MRPs)
An MRP is a simple model of a stochastic process that generates sequences of states and rewards, but without any notion of actions or control by an agent. It's essentially a Markov chain augmented with a reward function.

Definition
An MRP is a tuple M=⟨S,P,R,γ⟩, where:

S: A finite set of states.

P: A state transition probability matrix, where P 
ss 
′
 
​
 =P(S 
t+1
​
 =s 
′
 ∣S 
t
​
 =s) is the probability of transitioning from state s to state s 
′
  at the next time step.

R: A reward function. R 
s
​
 =E[R 
t+1
​
 ∣S 
t
​
 =s] is the expected immediate reward received after transitioning out of state s. (Sometimes defined as R(s,s 
′
 ) for transitioning from s to s 
′
 )

γ: A discount factor, γ∈[0,1]. It determines the present value of future rewards. A γ close to 0 leads to "myopic" evaluation, while a γ close to 1 leads to "far-sighted" evaluation.

Goal
The primary goal in an MRP is to evaluate the "goodness" of each state. This is quantified by the state-value function.

State-Value Function (V(s))
The state-value function V(s) for an MRP is the expected cumulative discounted reward starting from state s.
First, let's define the return G 
t
​
  at time t:

G 
t
​
 =R 
t+1
​
 +γR 
t+2
​
 +γ 
2
 R 
t+3
​
 +⋯= 
k=0
∑
∞
​
 γ 
k
 R 
t+k+1
​
 
The state-value function is then:

V(s)=E[G 
t
​
 ∣S 
t
​
 =s]
Bellman Equation for MRPs
The Bellman equation provides a recursive definition for the value function in an MRP. It expresses the value of a state in terms of the expected immediate reward and the discounted values of successor states.

Derivation:
Starting with the definition of V(s):

V(s)=E[G 
t
​
 ∣S 
t
​
 =s]
V(s)=E[R 
t+1
​
 +γG 
t+1
​
 ∣S 
t
​
 =s]
By linearity of expectation:

V(s)=E[R 
t+1
​
 ∣S 
t
​
 =s]+γE[G 
t+1
​
 ∣S 
t
​
 =s]
The first term is the definition of our reward function R 
s
​
 . For the second term, we can condition on the next state S 
t+1
​
 =s 
′
 :

E[G 
t+1
​
 ∣S 
t
​
 =s]= 
s 
′
 ∈S
∑
​
 P(S 
t+1
​
 =s 
′
 ∣S 
t
​
 =s)E[G 
t+1
​
 ∣S 
t
​
 =s,S 
t+1
​
 =s 
′
 ]
Due to the Markov property, E[G 
t+1
​
 ∣S 
t
​
 =s,S 
t+1
​
 =s 
′
 ]=E[G 
t+1
​
 ∣S 
t+1
​
 =s 
′
 ]=V(s 
′
 ).
So,

E[G 
t+1
​
 ∣S 
t
​
 =s]= 
s 
′
 ∈S
∑
​
 P 
ss 
′
 
​
 V(s 
′
 )
Substituting back, we get the Bellman equation for an MRP:

V(s)=R 
s
​
 +γ 
s 
′
 ∈S
∑
​
 P 
ss 
′
 
​
 V(s 
′
 )
This equation states that the value of a state s is the expected immediate reward R 
s
​
  plus the discounted expected value of the next state s 
′
 , averaged over all possible next states.

Matrix Form:
The Bellman equation can be written in matrix form. Let V be a column vector of values for all states, and R be a column vector of expected rewards for all states:

V=R+γPV
This is a system of linear equations. It can be solved directly if the number of states is not too large:

(I−γP)V=R
V=(I−γP) 
−1
 R
where I is the identity matrix. The inverse (I−γP) 
−1
  exists if γ<1 or if all states eventually lead to a terminal state.

Role in Reinforcement Learning
MRPs are often used to evaluate a given fixed policy in an MDP. When an agent follows a specific policy, the environment dynamics (from the agent's perspective) behave like an MRP, allowing for the calculation of state values under that policy.

2. Markov Decision Processes (MDPs)
MDPs extend MRPs by introducing actions, allowing an agent to influence state transitions and rewards. MDPs are the standard formalism for RL problems where an agent learns to make optimal decisions.

Definition
An MDP is a tuple M=⟨S,A,P,R,γ⟩, where:

S: A finite set of states.

A: A finite set of actions available to the agent. A(s) may denote the set of actions available in state s.

P: A state transition probability function (or dynamics model), P(s 
′
 ∣s,a)=P(S 
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
 =a), is the probability of transitioning to state s 
′
  from state s after taking action a.

R: A reward function. This can be defined in a few ways:

R(s,a,s 
′
 )=E[R 
t+1
​
 ∣S 
t
​
 =s,A 
t
​
 =a,S 
t+1
​
 =s 
′
 ]: Expected reward for transitioning from s to s 
′
  via action a.

R(s,a)=E[R 
t+1
​
 ∣S 
t
​
 =s,A 
t
​
 =a]=∑ 
s 
′
 ∈S
​
 P(s 
′
 ∣s,a)R(s,a,s 
′
 ): Expected reward for taking action a in state s.

γ: A discount factor, γ∈[0,1].

Goal
The goal in an MDP is to find an optimal policy π 
∗
 (a∣s) that maximizes the expected cumulative discounted reward from any starting state.

Policy (π)
A policy π is a mapping from states to a probability distribution over actions:

π(a∣s)=P(A 
t
​
 =a∣S 
t
​
 =s)
It defines the agent's behavior – how it chooses actions in each state. A policy can be deterministic (π(s)=a) or stochastic.

Value Functions in MDPs
Because the agent's actions influence outcomes, we define value functions with respect to a policy π.

State-Value Function (V 
π
 (s)):
The expected return when starting in state s and subsequently following policy π.

V 
π
 (s)=E 
π
​
 [G 
t
​
 ∣S 
t
​
 =s]
The subscript π indicates that the expectation is taken assuming the agent follows policy π.

Action-Value Function (Q 
π
 (s,a)):
The expected return when starting in state s, taking action a, and thereafter following policy π.

Q 
π
 (s,a)=E 
π
​
 [G 
t
​
 ∣S 
t
​
 =s,A 
t
​
 =a]
Q-values are often more directly useful for decision-making, as they tell us the value of taking a specific action in a state.

Bellman Expectation Equations for MDPs
These equations relate the value of a state (or state-action pair) to the values of subsequent states (or state-action pairs) under a given policy π.

Derivation for V 
π
 (s):

V 
π
 (s)=E 
π
​
 [R 
t+1
​
 +γV 
π
 (S 
t+1
​
 )∣S 
t
​
 =s]
To expand this, we consider the actions taken according to policy π and the subsequent state transitions:

V 
π
 (s)= 
a∈A
∑
​
 π(a∣s)E 
π
​
 [R 
t+1
​
 +γV 
π
 (S 
t+1
​
 )∣S 
t
​
 =s,A 
t
​
 =a]
The expectation E 
π
​
 [⋅∣S 
t
​
 =s,A 
t
​
 =a] means we average over possible next states s 
′
  given s and a:

V 
π
 (s)= 
a∈A
∑
​
 π(a∣s) 
s 
′
 ∈S
∑
​
 P(s 
′
 ∣s,a)(R(s,a,s 
′
 )+γV 
π
 (s 
′
 ))
Here, R(s,a,s 
′
 ) is the expected reward when transitioning from s to s 
′
  under action a.
If we use the reward function R(s,a)=∑ 
s 
′
 ∈S
​
 P(s 
′
 ∣s,a)R(s,a,s 
′
 ), which is the expected reward for taking action a in state s, the equation simplifies to:

V 
π
 (s)= 
a∈A
∑
​
 π(a∣s)(R(s,a)+γ 
s 
′
 ∈S
∑
​
 P(s 
′
 ∣s,a)V 
π
 (s 
′
 ))
Derivation for Q 
π
 (s,a):

Q 
π
 (s,a)=E 
π
​
 [R 
t+1
​
 +γQ 
π
 (S 
t+1
​
 ,A 
t+1
​
 )∣S 
t
​
 =s,A 
t
​
 =a]
After taking action a in state s, the environment transitions to s 
′
  with probability P(s 
′
 ∣s,a) and gives reward (expected) R(s,a,s 
′
 ). In state s 
′
 , the agent takes action a 
′
  according to π(a 
′
 ∣s 
′
 ).
So, we sum over possible next states s 
′
  and then over possible next actions a 
′
 :

Q 
π
 (s,a)= 
s 
′
 ∈S
∑
​
 P(s 
′
 ∣s,a)(R(s,a,s 
′
 )+γE 
π
​
 [Q 
π
 (S 
t+1
​
 ,A 
t+1
​
 )∣S 
t+1
​
 =s 
′
 ])
The term E 
π
​
 [Q 
π
 (S 
t+1
​
 ,A 
t+1
​
 )∣S 
t+1
​
 =s 
′
 ] is the expected Q-value from state s 
′
  if we follow policy π. This is equivalent to V 
π
 (s 
′
 ).
Thus:

Q 
π
 (s,a)= 
s 
′
 ∈S
∑
​
 P(s 
′
 ∣s,a)(R(s,a,s 
′
 )+γV 
π
 (s 
′
 ))
If we use the expected reward R(s,a)=∑ 
s 
′
 ∈S
​
 P(s 
′
 ∣s,a)R(s,a,s 
′
 ), this becomes:

Q 
π
 (s,a)=R(s,a)+γ 
s 
′
 ∈S
∑
​
 P(s 
′
 ∣s,a)V 
π
 (s 
′
 )
Alternatively, to express Q 
π
 (s,a) in terms of future Q-values, recall V 
π
 (s 
′
 )=∑ 
a 
′
 ∈A
​
 π(a 
′
 ∣s 
′
 )Q 
π
 (s 
′
 ,a 
′
 ). Substituting this into the above equation for Q 
π
 (s,a):

Q 
π
 (s,a)=R(s,a)+γ 
s 
′
 ∈S
∑
​
 P(s 
′
 ∣s,a) 
a 
′
 ∈A
∑
​
 π(a 
′
 ∣s 
′
 )Q 
π
 (s 
′
 ,a 
′
 )
Relationship between V 
π
  and Q 
π
 :
The two value functions under policy π are closely related:

V 
π
 (s)= 
a∈A
∑
​
 π(a∣s)Q 
π
 (s,a)
This means the value of a state under policy π is the expected value of the Q-values of all actions available in that state, weighted by the policy's probability of taking each action.

And conversely (as shown in the Q 
π
  derivation):

Q 
π
 (s,a)=R(s,a)+γ 
s 
′
 ∈S
∑
​
 P(s 
′
 ∣s,a)V 
π
 (s 
′
 )
This means the value of taking action a in state s and then following policy π is the expected immediate reward R(s,a) plus the discounted expected value of the next state s 
′
  (where V 
π
 (s 
′
 ) is the value of following π from s 
′
 ).

Bellman Optimality Equations
The goal of RL is often to find an optimal policy π 
∗
  that achieves the highest possible expected return. An optimal policy is one for which V 
π 
∗
 
 (s)≥V 
π
 (s) for all s∈S and all policies π.
The value functions for this optimal policy are denoted V 
∗
 (s) and Q 
∗
 (s,a).

Optimal state-value function: V 
∗
 (s)=max 
π
​
 V 
π
 (s)

Optimal action-value function: Q 
∗
 (s,a)=max 
π
​
 Q 
π
 (s,a)

The Bellman optimality equation for V 
∗
 (s) is derived by considering that an optimal policy must select the action that maximizes the expected return:

V 
(
 s,a)
Substituting the expression for Q 
∗
 (s,a) (which is R(s,a)+γ∑ 
s 
′
 ∈S
​
 P(s 
′
 ∣s,a)V 
∗
 (s 
′
 ) because if we take action a and then act optimally, the value of the next state is V 
∗
 (s 
′
 )):

V^(s') \right)
This equation states that the value of a state under an optimal policy must equal the expected return for the best action from that state, followed by acting optimally thereafter.

The Bellman optimality equation for Q 
∗
 (s,a) is:

Q 
(
 s 
′
 )
Since V 
(
 s 
′
 )=max 
a 
′
 ∈A
​
 Q 
(
 s 
′
 ,a 
′
 ), we can write:

Q 
(
 s 
′
 ,a 
′
 )
This equation states that the value of taking action a in state s and then following an optimal policy is the expected immediate reward plus the discounted expected value of the optimal action-value from the next state s 
′
 , where the best action a 
′
  is chosen in s 
′
 .

These equations are non-linear (due to the max operator) and typically do not have a closed-form solution like the Bellman expectation equations. They are solved using iterative methods like Value Iteration or Policy Iteration, or approximated by various model-free RL algorithms (e.g., Q-learning, SARSA).

Once Q 
∗
 (s,a) is found, the optimal policy π 
∗
  is deterministic and greedy with respect to Q 
∗
 (s,a):

π 
(
 s,a)
Role in Reinforcement Learning
MDPs provide the standard mathematical framework for most RL problems. They allow us to formally define the interaction between an agent and its environment, the goal of maximizing cumulative reward, and to develop algorithms for learning optimal policies. They are essential for problems involving control and decision-making under uncertainty.
