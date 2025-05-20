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

```text
├── /ipynb # Notebooks for RAC and Transformer-Encoded PPO Agents
│   ├── Compare_RAC.ipynb/
│   ├── RiskSensitive_AC_stock.ipynb/
│   ├── portfolio_optimization_new.ipynb/
│   └── transformer_ppo_portfolio.ipynb/
├── /src # Environments and Agents (Ray[RLLib]-based implementations)
├── Final_Stable and Realistic Deep Reinforcement Learning in Portfolio Management..pdf
└── README.md # This file
```

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

## 📚 Credits

This project is heavily inspired by the work:

**Paper**: [Deployability of Deep Reinforcement Learning in Portfolio Management](https://github.com/ishwargov/PortfolioOptimization/blob/main/Report.pdf)  
**Authors**: Ishwar Govind, Jerry Thomas, Prof. Chandrashekar Lakshminarayan.  
**Repository**: [Google Research – deep_rl_for_portfolio](https://github.com/ishwargov/PortfolioOptimization)

We have extended the original work with new agents (e.g., RAC, Transformer-PPO), reward functions (Sortino Ratio), and more realistic simulation environments.  
Please cite or credit the original authors if you reuse or modify any parts of their research or code.
