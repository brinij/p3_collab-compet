[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 2: Continuous Control

### Learning Algorithm : DDPG

Deep Deterministic Policy Gradient ([DDPG](https://arxiv.org/pdf/1509.02971.pdf)) is the RL algorithm that adapts the success ideas (replay buffer and target Q network) of Deep Q-Learning (DQN) to the continuous action domain. It is also an actor-critic, off-policy, model-free algorithm where two deep neural networks are used. The actor network is used to approximate the optimal policy deterministically. That means it always outputs the best believed action for any given state. The critic learns to evaluate the optimal action value function by using the actors best believed action.

Here are solutions proposed for the issues that arise in Deep RL algorithms:

-  **Replay Buffer** is a finite sized cache where sampled transitions from the environment are stored. When the replay buffer is full, the oldest samples are discarded. At each timestep the actor and critic are updated by sampling a minibatch uniformly from the buffer, which solves the assumption of most NN optimization algorthms that the samples are independently and indetically distributed. Because DDPG is an off-policy algorithm, the replay buffer can be large, allowing
the algorithm to benefit from learning across a set of uncorrelated transitions.

- **Soft-update** is used instead of directly copying the weights to the target network. First, the copy of actor and critic networks, that are used for calculating the target values, are created. Then, the weights of these target networks are updated by having them slowly track the learned networks: `θ' ← τθ + (1 − τ)θ'` with `τ <= 1`. This means that the target values are constrained to change slowly, greatly improving the stability of learning.

- **Batch Normalization** is a technique that normalizes each dimension across the samples in a minibatch to have unit mean and variance. It solves the problem when the input states are low dimensional feature vector of observations which contains different components with different physical units (for example, position in centimeters and angular velocity in radians) where ranges may vary.

- **Noise** is added to actor policy in order to increse the exploration which is the major challenge of learning in continuous action spaces. An advantage of off-policies algorithms such as DDPG is that the problem of exploration can be treated independently from the learning algorithm. For this project, an Ornstein-Uhlenbeck process (Uhlenbeck & Ornstein, 1930) with `θ = 0.15` and `σ = 0.1` is used, which models the velocity of a Brownian particle with friction, which results in temporally correlated values centered around 0. It performs well in physical environments that have momentum.

The pseudocode of [DDPG](https://arxiv.org/pdf/1509.02971.pdf) algorithm is shown below:

<p align="center">
<img src="https://github.com/brinij/p2_continuous-control/blob/master/DDPG_algorithm.png" width="600">
</p>

### Model Architecture
This is the architecture of the **Actor Network** :
<p align="center">
<img src="https://github.com/brinij/p2_continuous-control/blob/master/ActorNetwork.png" width="600">
</p>

This is the architecture of the **Critic Network** :
<p align="center">
<img src="https://github.com/brinij/p2_continuous-control/blob/master/CriticNetwork.png" width="600">
</p>


### Implementation Details

In the project file there are two Python files defining five classes. 
- In `model.py` are Python classes `Actor` and `Critic` who define the structures of Neural Networks used in this project for solving the Reacher environment. Model architecure is shown above.
- In `dqn_agent.py` are three Python classes: `ReplayBuffer` which implements the functionality of adding samples to a buffer and sampling from it,`OUNoise` which implements noise adding important for exploration, and `Agent` with methods `step` and `act` so that Agent can interact with the environment and add the experience to the memory, and method `learn` which is called every 5. step taken. 

### Hyperparameters
Parameters that showed the best results are:
- `BUFFER_SIZE` = 1e6 (1 milion), recommended in the ddpg paper
- `BATCH_SIZE`  = 128 , bigger is better, it is limited by RAM memory of the maschine where you run the learning
- `GAMMA`       = 0.99 , discount factor
- `TAU`         = 1e-3 , parameter for soft update of target parameters
- `LR_ACTOR`    = 3e-4 , learning rate of the actor
- `LR_CRITIC`   = 3e-4 , learning rate of the critic
- `UPDATE_EVERY`= 5, how often to update the network
- `WEIGHT_DECAY` = 0 ,  L2 weight decay

### Result

The Environment has been solved in 369 learning episodes where each of them lasted 1000 steps. The environment is considered solved when in the last 100 episodes average reward is 30. The graph of rewards during the learning period is shown in the image below:

<p align="center">
<img src="https://github.com/brinij/p2_continuous-control/blob/master/p2_rewards.png" width="400">
</p>

```python
Episode 100	Average Score: 1.30	Score: 1.76
Episode 200	Average Score: 7.07	Score: 12.98
Episode 300	Average Score: 17.27	Score: 28.50
Episode 400	Average Score: 25.34	Score: 13.10
Episode 469	Average Score: 30.04	Score: 23.75
Environment solved in 369 episodes!	Average Score: 30.04
Episode 500	Average Score: 31.14	Score: 27.01
Episode 600	Average Score: 31.96	Score: 34.27
```

And here is gif of the agent performing in the environment after training:

<p align="center">
<img src="https://github.com/brinij/p2_continuous-control/blob/master/reacher_trained_solution.gif" width="600">
</p>

## Improvements
In order to get the best possible results in continuous domain environment with multiple agents, it would be necessary to implement [D4PG](https://openreview.net/forum?id=SyZipzbCb) algorithm which showed amazing results across many different environments by using **Prioritized Experience Replay** and **N-step returns** techniques.
