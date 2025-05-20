# Portfolio Optimization using Deep Reinforcement Learning

## Authors

- **Akshitha Peguda**  
- **Shatrughna Chaurasia**  
- **Ishwar Govind**  
- **Jerry Thomas**

**Mentors**:  
- Prof. Shashi Shekhar Jha  
- Prof. Chandrashekar Lakshminarayan

This project explores the use of **Deep Reinforcement Learning (DRL)** for **stable, risk-sensitive portfolio optimization** by incorporating realistic market environments and custom reward functions like the Sortino Ratio.

## Repository Structure


## Key Features

- Risk-sensitive Actor-Critic (RAC) Agent
- Transformer-encoded PPO agent
- Custom reward: **Sortino Ratio** (focuses on downside risk)
- Market simulation using **BEKK** and other realistic models
- Comparative analysis with traditional DRL models: A2C, DDPG, PPO, TRPO
- Evaluation metrics include:
  - Sharpe Ratio
  - Sortino Ratio
  - Max Drawdown
  - Annual Returns & Volatility
  - Value at Risk (VaR)

## Highlights from Results

- **TRPO with Sharpe Ratio** achieved the best performance in terms of cumulative returns and stability.
- **Sortino Ratio** reward provided better risk-sensitive learning, especially when used with PPO.
- Stable training and more realistic simulations resulted in significantly better drawdown control.
