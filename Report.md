[//]: # (Image References)

[image1]: images/q_formula.png "Action-value function"
[image2]: images/n1.png "Q-Network"
[image3]: images/n2.png "Dueling Q-Network"
[image4]: images/bellman.png "Bellman equation"
[image5]: images/eq2.png "Bellman equation"
[image6]: images/loss.png "loss"
[image7]: images/algorithm.png "Algorithm"

#  Navigation

### Goal

The goal of the project is to train agent to Solve "Bananas" environemnt. You can found detailed description of the environemnt following the [link](README.md) 


### Solution Summary

To solve the environment we are going to use Deep Q-learning with experience replay algorithm published in the [the Deepmind paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). We need to calculate optimal action-value function Q:

![Action-value function][image1]

The problem is that our state space is continius  with 37 dimensions so we can not use traditional temporal-difference method like SARSA or [Q-learning](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.80.7501&rep=rep1&type=pdf). Of course this task can be solved with descritisations techniques like: Tile Coding or Course Coding. However as described in the paper the better results can be archived with function approimation aproach and using of Neural Network as a universal function aproximator for this purpose.  So we approaximating true action-value function q(S,a) with function q(S,a,w). Our goal is to optimize parameters w to make approximatein as good as possible.

It was a known fact that reinfercement learning is unstable when a Q function is represented with newral network. Autwors introduced two additional ideas:
    - Expirience replay - the mechanizm to store observed tuples of (state, action, next_state, is_terminal) in the special buffer and randomly sample these tuples during learning process. Fistly, it allows to resue observed tuples for training over and over agin, secondly it breaks correlation with latest observed sequence.
    - Target values are stored in the separate network with same archtiecture and only periodicaly updated reducing correlation with latest target values.

As any other reinforcement learning algorithms the action-value function is estimated by using the Bellman equation as an iterative update: ![Action-value function][image4]

The issue is that we do not have actual target values - so we estimate them from: ![Action-value function][image5]. Useing weights from previous target network which was fixed on some previous iterations.

This leads to optimization of the loss function:

![Action-value function][image6]


So the final algorithm is: [(from original paper)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

![network architecture][image7]


### Network archtiecture

We have different network architecture then described in the paper. Archtiecture is designed there to capture featrues from the eaw pixel data - so Convolutional network is natural fit there. WE have simplified state vector instead so pure Fully Connected network must work relatively well.

![network architecture][image2]


### Variations 

1. The Double DQN

2. Dueling DQN: 

![ Dueling DQN][image3]

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Install Anaconda for Python 3: https://www.anaconda.com/download/
3. Setup Anaconda environment:
    - conda create --name bananas python=3.6 
    - conda activate bananas 
    - python -m ipykernel install --user --name bananas --display-name "bananas"
4. Install required packages:
    - pip install numpy 
    - pip install unityagents
    - conda install pytorch torchvision cuda90 -c pytorch    

### Instructions

Run Notebook using command "jupyter notebook" and follow the instructions in `Navigation.ipynb` to get started with training the agent!  
