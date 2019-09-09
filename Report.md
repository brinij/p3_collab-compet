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

The pseudocode of ([DDPG](https://arxiv.org/pdf/1509.02971.pdf)) algorithm is shown below:

<p align="center">
<img src="https://github.com/brinij/p2_continuous-control/blob/master/DDPG_algorithm.png" width="600">
</p>


### Implementation Details

In the project file there are two Python files defining three classes. 
- In `model.py` is Python class `QNetwork` which defines the structure of Neural Network used in this project for solving the Banana environment. It has three linear layers where first two are followed by ReLu activation functions and the last one is linear. First two hidden layers have 64 nodes and the last one has the size of the action space which is 4.
- In `dqn_agent.py` are two Python classes: `ReplayBuffer` which implements the functionality of adding samples to a buffer and sampling from it, and `Agent` with methods `step` and `act` so that Agent can interact with the environment and add the experience to the memory, and method `learn` which is called every 4. step taken. 

### Hyperparameters
Parameters that showed the best results are:
- `BUFFER_SIZE` = 1e6 (1 milion), recommended in the dqn paper
- `BATCH_SIZE`  = 64 , bigger is better, it is limited by RAM memory of the maschine where you run the learning
- `GAMMA`       = 0.99 , discount factor
- `TAU`         = 1e-3 , parameter for soft update of target parameters
- `LR`          = 5e-4 , learning rate
- `UPDATE_EVERY`= 4, how often to update the network
- `EPS_DECAY` = 0.995, how much to decay epsilon from 1.0 to 0.1, for epsilon-greedy action selection

### Result

Environment has been solved in 419 learning episodes where each of them lasted 300 steps. The environment is considered solved when in the last 100 episodes average reward is 13. The graph of rewards during the learning period is shown in the image below:

<p align="center">
<img src="https://github.com/brinij/p2_continuous-control/blob/master/p2_rewards.png" width="400">
</p>

## Improvements

- Since Deep Q-Learning tends to overestimate action values, **Double DQN** has been shown as a good solution for that. 
Instead of blindly trusting the max value of Q-values that are completely random at the beginning of learning, we select the best action using one set of parameters w and evaluete it using a different set of parameters w'. It is like having two seperate function approximators that must agree on the best action. The second set of parameters are the ones from target network that are frozen for some time and so differenet enough to be reused for this purpose. In the long run this prevents the algorithm from propagating incidental high rewards taht may have been obtained by chance, and do not reflect long-term returns. 
- Another improvement can be **Prioritized Experience Replay** which is based on the idea that the agent can learn more effectively from some transitions than from others, and the more important transitions should be sampled with higher probability. Importance of an sample is measured with the TD error, where the bigger the error, the more we expect to learn from that tuple. 
