[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 3: Collaboration and Competition

### Learning Algorithm : DDPG

Deep Deterministic Policy Gradient ([DDPG](https://arxiv.org/pdf/1509.02971.pdf)) is adapted to work with two agents. Since each agent receives its own, local observation, the code from DRLND Project 2 (Continuous Control) was easily adapted to train both agents through self-play. Each agent has its own actor network (both have the same architecture) and they share critic network. Also both agents are adding and sampling from the same shared replay buffer. ([MADDPG](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf))

### Model Architecture
This is the architecture of the **Actor Networks** :
<p align="center">
<img src="https://github.com/brinij/p3_collab-compet/blob/master/maddpg_actor.png" width="600">
</p>

This is the architecture of the **Critic Network** :
<p align="center">
<img src="https://github.com/brinij/p3_collab-compet/blob/master/maddpg_critic.png" width="600">
</p>


### Implementation Details

In the project file there are two Python files defining five classes. 
- In `model.py` are Python classes `Actor` and `Critic` who define the structures of Neural Networks used in this project for solving the Reacher environment. Model architecure is shown above.
- In `dqn_agent.py` are three Python classes: `ReplayBuffer` which implements the functionality of adding samples to a buffer and sampling from it,`OUNoise` which implements noise adding important for exploration, and `Agent` with methods `step` and `act` so that Agent can interact with the environment and add the experience to the memory, and method `learn` which is called every 5. step taken. 

### Hyperparameters
Parameters that showed the best results are:
- `BUFFER_SIZE` = 1e6 (1 milion), recommended in the ddpg paper
- `BATCH_SIZE`  = 128 , minibatch size
- `GAMMA`       = 0.99 , discount factor
- `TAU`         = 2e-1 , parameter for soft update of target parameters
- `LR_ACTOR`    = 1e-4 , learning rate of the actor
- `LR_CRITIC`   = 3e-4 , learning rate of the critic
- `WEIGHT_DECAY` = 0 ,  L2 weight decay

### Result

The Environment has been solved in 208 learning episodes where each of them lasted until done is returned from the environment. The environment is considered solved when in the last 100 episodes average reward is 0.5. The graph of rewards during the learning period is shown in the image below:

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
