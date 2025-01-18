from tqdm import tqdm
import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
import gym
import minerl
from buffered_batch_iter import BufferedBatchIter
from data_pipeline import DataPipeline
import os


number_of_actions = 7
"""
Your task: Implement behavioural cloning for MineRLTreechop-v0.

Behavioural cloning is perhaps the simplest way of using a dataset of demonstrations to train an agent:
learn to predict what actions they would take, and take those actions.
In other machine learning terms, this is almost like building a classifier to classify observations to
different actions, and taking those actions.

For simplicity, we build a limited set of actions ("agent actions"), map dataset actions to these actions
and train on the agent actions. During evaluation, we transform these agent actions (integerse) back into
MineRL actions (dictionaries).

To do this task, fill in the "TODO"s and remove `raise NotImplementedError`s.

Note: For this task you need to download the "MineRLTreechop-v0" dataset. See here:
https://minerl.readthedocs.io/en/latest/tutorials/data_sampling.html#downloading-the-minerl-dataset-with-minerl-data-download
"""

class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x, inplace=True)
        return x

class ConvNet(nn.Module):
    """
    :param input_shape: A three-item tuple telling image dimensions in (C, H, W)
    :param output_dim: Dimensionality of the output vector
    """

    def __init__(self, input_shape, output_dim):
        super().__init__()

        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=6, stride=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256),
            nn.AdaptiveAvgPool2d((1,1))
        )

        # Calculate output size
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[1], 6, 3), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[2], 8, 3), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.fc1 = nn.Linear(256, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, output_dim)

        self.dropout = nn.Dropout(0.4)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        # TODO Create a torch neural network here to turn images (of shape `input_shape`) into
        #      a vector of shape `output_dim`. This output_dim matches number of available actions.
        #      See examples of doing CNN networks here https://pytorch.org/tutorials/beginner/nn_tutorial.html#switch-to-cnn
        # raise NotImplementedError("TODO implement a simple convolutional neural network here")

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten feature maps

        # Fully connected layers with residuals
        x = F.relu(self.bn1(self.fc1(x)), inplace=True)
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)), inplace=True)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def agent_action_to_environment(noop_action, agent_action):
    noop_action['attack']=1
    match agent_action:
        case 0:
            noop_action['forward']=1
        case 1:
            noop_action['jump']=1
        case 2:
            noop_action['camera'][1]=15
        case 3:
            noop_action['camera'][1]=-15
        case 4:
            noop_action['camera'][0]=-15
        case 5:
            noop_action['camera'][0]=15
        case 6:
            noop_action['attack']=1
    return noop_action

    """
    Turn an agent action (an integer) into an environment action.
    This should match `environment_action_batch_to_agent_actions`,
    e.g. if attack=1 action was mapped to agent_action=0, then agent_action=0
    should be mapped back to attack=1.

    noop_action is a MineRL action that does nothing. You may want to
    use this as a template for the action you return.
    """
    # raise NotImplementedError("TODO implement agent_action_to_environment (see docstring)")


def environment_action_batch_to_agent_actions(dataset_actions):

    """
    Turn a batch of actions from environment (from BufferedBatchIterator) to a numpy
    array of agent actions.

    Agent actions _have to_ start from 0 and go up from there!

    For MineRLTreechop, you want to have actions for the following at the very least:
    - Forward movement
    - Jumping
    - Turning camera left, right, up and down
    - Attack

    For example, you could have seven agent actions that mean following:
    0 = forward
    1 = jump
    2 = turn camera left
    3 = turn camera right
    4 = turn camera up
    5 = turn camera down
    6 = attack

    This should match `agent_action_to_environment`, by converting dictionary
    actions into individual integeres.

    If dataset action (dict) does not have a mapping to agent action (int),
    then set it "-1"
    """
    # There are dummy dimensions of shape one
    batch_size = len(dataset_actions["camera"])
    actions = np.zeros((batch_size,), dtype=np.int32)

    for i in range(batch_size):
        # TODO this will make all actions invalid. Replace with something
        # more clever
        if(dataset_actions['camera'][i][0]==0 and dataset_actions['camera'][i][1]==0):
            if(dataset_actions['forward'][i]==1): actions[i] = 0
            elif(dataset_actions['jump'][i]==1): actions[i] = 1
            elif(dataset_actions['attack'][i]==1): actions[i] = 6
            else: actions[i] = -1
        else:
            pitch = dataset_actions['camera'][i][0]
            yaw = dataset_actions['camera'][i][1]
            if max(abs(pitch), abs(yaw)<=15):
                if abs(pitch) > abs(yaw):  # Vertical movement dominates
                    if pitch > 0:
                        actions[i] = 5
                    else:
                        actions[i] = 4
                else:  # Horizontal movement dominates
                    if yaw > 0:
                        actions[i] = 2
                    else:
                        actions[i] = 3
            else: actions[i] = -1            
    return actions


def train():
    # Path to where MineRL dataset resides (should contain "MineRLTreechop-v0f" directory)
    DATA_DIR = "."
    # How many times we train over dataset and how large batches we use.
    # Larger batch size takes more memory but generally provides stabler learning.
    EPOCHS = 1
    BATCH_SIZE = 32
    learning_rate = 3e-4

    data = DataPipeline(os.path.join('/home/dodo/byop', 'MineRLTreechop-v0'), 'MineRLTreechop-v0', 4, 32, 32)

    iterator = BufferedBatchIter(data)

    number_of_actions = 7

    network = ConvNet((3, 64, 64), number_of_actions)
    # removed .cuda() at end

    optimizer = th.optim.AdamW(
        network.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    # optimizer = th.optim.SGD(network.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)

    iter_count = 0
    losses = []
    for dataset_obs, dataset_actions, _, _, _ in tqdm(iterator.buffered_batch_iter(num_epochs=EPOCHS, batch_size=BATCH_SIZE)):
        # We only use camera observations here
        obs = dataset_obs["pov"].astype(np.float32)
        # Transpose observations to be channel-first (BCHW instead of BHWC)
        obs = obs.transpose(0, 3, 1, 2)
        # Normalize observations, otherwise the neural network will get spooked
        obs /= 255.0

        # Turn datasenv = gym.make('MineRLTreechop-v0')et actions into agent actions
        actions = environment_action_batch_to_agent_actions(dataset_actions) 
        assert actions.shape == (obs.shape[0],), "Array from environment_action_batch_to_agent_actions should be of shape {}".format((obs.shape[0],))

        # Remove samples that had no corresponding action
        mask = actions != -1
        obs = obs[mask]
        actions = actions[mask]

        # TODO perform optimization step:
        # - Predict actions using the neural network (input is `obs`)
        # - Compute loss with the predictions and true actions. Store loss into variable `loss`
        # - Use optimizer to do a single update step
        # See https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html 
        # for a tutorial
        # NOTE: Variables `obs` and `actions` are numpy arrays. You need to convert them into torch tensors.

        pred = network(th.from_numpy(obs))
        loss = loss_function(pred, th.from_numpy(actions).long())

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Keep track of how training is going by printing out the loss
        iter_count += 1
        losses.append(loss.item())
        if (iter_count % 1000) == 0:
            mean_loss = sum(losses) / len(losses)
            tqdm.write("Iteration {}. Loss {:<10.3f}".format(iter_count, mean_loss))
            losses.clear()
            th.save(network, f"behavioural_cloning{iter_count/500}.pth")

    # Store the network
    th.save(network, "behavioural_cloning.pth")


def enjoy():
    # Load up the trained network
    network = th.load("behavioural_cloning15.0.pth")

    env = gym.make('MineRLTreechop-v0')

    # Play 10 games with the model
    for game_i in range(10):
        obs = env.reset()
        done = False
        reward_sum = 0
        while not done:
            obs = obs["pov"].astype(np.float32)
            obs = np.expand_dims(obs, axis=0)
            obs = obs.transpose(0, 3, 1, 2)
            obs /= 255.0
            
            logits = network(obs)
            # Turn logits into probabilities
            probabilities = th.softmax(logits, dim=1)[0]
            # Into numpy
            probabilities = probabilities.detach().cpu().numpy()
            # TODO Pick an action based from the probabilities above.
            # The `probabilities` vector tells the probability of choosing one of the agent actions.
            # You have two options:
            # 1) Pick action with the highest probability
            # 2) Sample action based on probabilities
            # Option 2 works better emperically.

            agent_action = np.random.Generator.choice(number_of_actions, p=probabilities)

            noop_action = env.action_space.noop()
            environment_action = agent_action_to_environment(noop_action, agent_action)

            obs, reward, done, info = env.step(environment_action)
            reward_sum += reward
        print("Game {}, total reward {}".format(game_i, reward_sum))

    env.close()

if __name__ == "__main__":
    # First train the model...
    # train()
    # ... then play it on the environment to see how it does
    enjoy()



