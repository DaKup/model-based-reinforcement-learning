# Model-Based-Reinforcement-Learning

- deprecated
  - https://github.com/hejia-zhang/awesome-model-based-reinforcement-learning
- https://towardsdatascience.com/the-complete-reinforcement-learning-dictionary-e16230b7d24e
- http://rail.eecs.berkeley.edu/deeprlcourse/

## Algorithms

### Model-Free

- SAC
  - Soft Actor-Critic
- Value Iteration
- Policy Iteration
- R-Learning
- MC
  - Monte Carlo [Offline]
- SARSA
  - State-Action-Reward-State-Action
  - On-Policy
  - Value-Based
- Q-Learning
  - Off-Policy
- Eligibility Traces TD
- Actor Critic
- Q-P Learning
- Policy Gradient
- A2C/A3C
  - Asynchronous Advantage Actor-Critic
- PPO
  - Proximal Policy Optimization
- TRPO
  - Trust Region Policy Optimization
- DQN
  - Deep Q-Network
- C51
- QR-DQN
- HER
- DDPG
  - Deep Deterministic Policy Gradient
- TD3
- SAC

### Model-Based

- Dyna Q
- Dyna Q+
- Deep Dyna Q
- Prioritized Sweeping
  - 1993
- Model-Based Q-Learning
- Model-Based Relative Q-Learning
- World Models
- I2A
  - Imagination-Augmented Agents
  - 2018
- MBMF
  - Model-Based Priors for Model-Free Reinforcement Learning
  - 2017
- MBVE
  - Model-Based Value Estimation for Efficient Model-Free Reinforcement Learning
  - 2018
- AlphaZero
  - 2017
- MuZero
  - 2019
- SimPle
  - Simulated Policy Learning
  - 2019

## Labels

- Model-Free vs Model-Based
- Average Reward Algorithms vs Discounted Reward Algorithms
- Value Iteration vs Online-Temporal Difference (TD)
- Temporal Difference Learning
- Policy Iteration (Q-P Learning)
- Policy Optimization
- Q-Learning
- Value-based Methods
- Policy-based Methods
- Off-Policy vs On-Policy
- Learn the Model vs Given the Model
- Inverse Reinforcement Learning
  - Learn what the goal was
  - Learn the reward function
- Deep Reinforcement Learning
- Apprecenticeship Learning
  - Learn from observing an expert
- Actor-Critic methods
  - think of actor-critic algorithms as value and policy-based because they use both a value and policy functions.


## Unordered

- PILCO
  - Probabilistic Inference for Learning COntrol
  - A model-Based and Data-Efficient Approach to Policy Search
- GMM Gaussian Mixture Model
- MPC Model Predictive Control


## Notes

- train: model (transition probability of state and action) and policy
  - warm-up phase
  - alternating training
- model training: seek for unexpected observations (curiosity)
  - another model(?) to predict if action will lead to new observations
- replay memory
  - epsilon? stochastic replay won't always lead to the same result due to uncertainty
  - not a perfect MC environment
  - observation (incomplete) vs state (complete)
- 