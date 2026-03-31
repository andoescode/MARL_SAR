

# import imageio

class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        # Create two fully connected layers
        # h_size = hidden size
        self.fc1 = nn.Linear(s_size, h_size) # state -> h_size hidden
        self.fc2 = nn.Linear(h_size, a_size) # h_size hidden -> action


    def forward(self, x):
        # Define the forward pass
        # state goes to fc1 then we apply ReLU activation function
        x = F.relu(self.fc1(x))
        # fc1 outputs goes to fc2
        x = self.fc2(x)

        # We output the softmax
        return F.softmax(x, dim=1)
    
    def act(self, state):
        """
        Given a state, take action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)