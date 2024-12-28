# Machine Learning Types: Fundamentals and Implementation

## Learning Objectives
- Understand the core types of machine learning
- Identify when to use each type of learning
- Implement basic examples of each approach

## 1. Overview of Machine Learning Types
### What is Machine Learning?
Machine learning enables computers to learn from data without being explicitly programmed. Unlike traditional programming where we write specific rules, ML algorithms find patterns in data to make decisions or predictions.

### Types of Machine Learning
```python
# Visual representation of ML types using Python and Matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_types():
    """
    Create a visual representation of different learning types
    using synthetic data
    """
    # Create example data for each type
    # Supervised Learning Example
    X_supervised = np.random.randn(50, 2)
    y_supervised = X_supervised[:, 0] + X_supervised[:, 1] > 0
    
    # Unsupervised Learning Example
    X_unsupervised = np.concatenate([
        np.random.randn(25, 2) + np.array([2, 2]),
        np.random.randn(25, 2) + np.array([-2, -2])
    ])
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot supervised learning
    ax1.scatter(X_supervised[y_supervised, 0], X_supervised[y_supervised, 1], 
                c='blue', label='Class 1')
    ax1.scatter(X_supervised[~y_supervised, 0], X_supervised[~y_supervised, 1], 
                c='red', label='Class 2')
    ax1.set_title('Supervised Learning\n(Classification)')
    ax1.legend()
    
    # Plot unsupervised learning
    ax2.scatter(X_unsupervised[:, 0], X_unsupervised[:, 1], c='gray')
    ax2.set_title('Unsupervised Learning\n(Clustering)')
    
    plt.tight_layout()
    plt.show()

# Execute the visualization
plot_learning_types()
```

## 2. Supervised Learning Example
```python
# Simple linear regression example
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate sample data
X = np.random.randn(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.1

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

print(f"Model coefficient: {model.coef_[0][0]:.2f}")
print(f"Model intercept: {model.intercept_[0]:.2f}")
```

## 3. Reinforcement Learning Example
Reinforcement learning is like training a pet - the agent learns through trial and error, receiving rewards for good actions and penalties for bad ones.

### Key Components
- **Agent**: The learner/decision maker
- **Environment**: The world the agent interacts with
- **Actions**: What the agent can do
- **Rewards**: Feedback from the environment
- **State**: Current situation of the agent

### Simple Grid World Example
Here's a Q-learning implementation where an agent learns to navigate a grid:

```python
import numpy as np
from collections import defaultdict

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.state = 0  # Start state
        self.goal = size * size - 1  # Goal state
        self.actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left
        
    def step(self, action):
        # Apply action and get reward
        # Returns: next_state, reward, done
        pass  # Implementation details omitted for brevity

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1):
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.lr = learning_rate
        
    def get_action(self, state):
        # Choose action using epsilon-greedy policy
        pass  # Implementation details omitted for brevity
        
    def learn(self, state, action, reward, next_state):
        # Update Q-values based on experience
        pass  # Implementation details omitted for brevity
```

Key Points about Reinforcement Learning:
- Learning through interaction with environment
- Balance between exploration and exploitation
- No direct supervision, only rewards
- Agent must discover good actions through trial and error

## 4. Implementation Notes
- Each type of learning suits different problems
- Supervised learning: When you have labeled data
- Unsupervised learning: When finding patterns in unlabeled data
- Reinforcement learning: When learning from interaction/feedback

## 5. Resources and References
- Course materials
- Additional reading
- Helpful tutorials
- Code examples

## 6. Next Steps
- Areas for deeper study
- Practice exercises
- Project ideas
  - Build a supervised learning classifier
  - Create a clustering algorithm
  - Implement a simple reinforcement learning game
