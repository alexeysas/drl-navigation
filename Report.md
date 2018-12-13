[//]: # (Image References)

[image1]: images/q_formula.png "Action-value function"
[image2]: images/n1.png "Q-Network"
[image3]: images/n2.png "Dueling Q-Network"
[image4]: images/bellman.png "Bellman equation"
[image5]: images/eq2.png "Bellman equation"
[image6]: images/loss.png "loss"
[image7]: images/algorithm.png "Algorithm"
[image8]: images/dqn.png "Dqn"

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

The overall architecture which provides good results for the enviroments is presented on the image below:


![network architecture][image2]


Relu is used as activation function and Dropout as regularization technique.

### Training 

Network was trained on the 1500 iterations. And result can be seen below.

![network architecture][image8]

As you can see enviroment was soleved around 600 episode. And maximum average score over 100 episeods is 17.01. Which is prety good result. 

### Variations 

Additinaly couple additional improvements was tried for this environment desacribed

1. The Double DQN: the Deepmind paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

2. Dueling DQN with following architecture below: 

![ Dueling DQN][image3]

3. Combination of Dueling and  Double DQN


### Next Step

There are possible additional improvements might be tried described in the Rainbow paiper: https://arxiv.org/pdf/1710.02298.pdf

1. Prioritized expiriebce relay https://arxiv.org/pdf/1710.02298.pdf
2. Distributional RL
3. Noisy nets https://arxiv.org/pdf/1710.02298.pdf

to improve results

Additinal improvements might be related to environment state itself.  The first idea is relatede to sobserved behavior: it might be benefitial to add additional value to state vector which idicates how much time left to complete the episode. The intuition behind that is that same states in the beginingof episode and at the end of episode have different values. THe most vivid example is when you surroundede by black bananas - in the begining of the episode it makes sense just go and collect banana to geout of at cost of -1 points and compensate later as at the end of the episode it might have more sense to just stay and wait while episode completed.

And final step was to try to learn agent from he raw pixl data using Convolutional netwrok as originaly proposed in the paper.  Also ther might be benefitial to use Recurrent or long sequence of fremaes to  "remember" what agenmt seen couple of seconds ago - expecial usefl when agent is rotated at 180% aand this information is lost. 




