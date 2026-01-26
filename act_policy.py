'''
File: act.py

Author: Eli Yale

Description: The action chunking transformer model implemented in pytorch
'''
def train_policy(obs, acs, nn_policy, num_train_iters):
    nn_policy.train()

    optimizer = Adam(nn_policy.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_train_iters):
        # Forward pass
        logits = nn_policy(obs)

        # Compute supervised BC loss
        loss = criterion(logits, acs)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class PolicyNetwork(nn.Module):
    '''
        Simple neural network with two layers that maps a 2-d state to a prediction
        over which of the three discrete actions should be taken.
        The three outputs corresponding to the logits for a 3-way classification problem.
    '''
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 3)
        

    def forward(self, x):
        logits = self.fc2(self.relu(self.fc1(x)))
        return logits
    

#evaluate learned policy
def evaluate_policy(pi, num_evals, human_render=True):
    if human_render:
        env = gym.make("MountainCar-v0",render_mode='human') 
    else:
        env = gym.make("MountainCar-v0") 

    policy_returns = []
    for i in range(num_evals):
        done = False
        total_reward = 0
        obs = env.reset()
        while not done:
            #take the action that the network assigns the highest logit value to
            #Note that first we convert from numpy to tensor and then we get the value of the 
            #argmax using .item() and feed that into the environment
            action = torch.argmax(pi(torch.from_numpy(obs).unsqueeze(0))).item()
            obs, rew, done, info = env.step(action)
            total_reward += rew
        print("reward for evaluation", i, total_reward)
        policy_returns.append(total_reward)

    print("average policy return", np.mean(policy_returns))
    print("min policy return", np.min(policy_returns))
    print("max policy return", np.max(policy_returns))

