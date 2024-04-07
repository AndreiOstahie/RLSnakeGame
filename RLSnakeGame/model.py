import torch
import torch.nn as nn
import torch.nn.functional as nnFunc
import torch.optim as optim
import os

class Linear_Qnet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):  # feed-forward neural network with 3 layers: input, hidden, output
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    # "prediction" function
    def forward(self, x):
        x = nnFunc.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()  # loss function -> mean squared error

    def train_step(self, state, action, reward, next_state, game_over):
        # data is in correct shape -> (n, x)
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)



        if len(state.shape) == 1:  # train model with a single instance (train_short_memory in agent)
            # data is not in correct shape -> convert data from format x to format (1, x)
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            game_over = (game_over, )  # tuple with single value

        # predicted Q values with current state
        prediction = self.model(state)

        # predict new Q value: Q_new = r + y * max(next_predicted_Q_value) -> only do this if not game_over
        # predictionss[argmax(action)] = Q_new

        target = prediction.clone()
        for i in range(len(game_over)):
            q_new = reward[i]
            if not game_over[i]:
                q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))

            target[i][torch.argmax(action).item()] = q_new

        self.optimizer.zero_grad()  # empty gradient
        loss = self.criterion(target, prediction)
        loss.backward()

        self.optimizer.step()
