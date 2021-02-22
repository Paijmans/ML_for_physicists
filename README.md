# Machine learning for physicists course

Below you find my notes from the Machine Learning for Physicists course by prof.  Marquardt from the University of Erlangen:

[Machine Learning for Physicists website](https://machine-learning-for-physicists.org/)

## Lecture 1 - Introduction

### Concepts of the course

#### Outline

1. Supervised learning (Learning by ground truth)
2. Unsupervised learning (Excersize with only data, no labels or ground truth)
3. Reinforcement learning (Learn strategy yourself, given fitness function)
4. Recurrent networks (Net with memory)
5. Bolzmann Machines: Learning a probablility distribution. (Eg. over images, generate new images with same statistics.)

#### Software we will use

- Python
- Keras and Tensor flow.

### Neural network definition

> Non linear function $f$ with many variables and parameters

$$\vec y = f(\vec X)$$

#### Structure of neural net

- Input layer
- Hidden layers
- Output layer

#### Structure of a single neuron

Output of the neuron is non-linear function of weighted sum of inputs.

Take $z$, which results from a linear function on input layers. Define non-linear function $f$ on $z$ which will give the neuron output.
Populair exacples for $f$:

- Sigmoidal function.
- Piecewise linear function or relu (zero, linear dependence)

Repeatedly applying neural net to input, creates a very complex function on the input. (Cool picture!)

## Lecture 2 - Neural networks

Efficient bash processing: concatonate all input samples in a matrix x[N_samples][N_data]:

$z = w x + b$

A non-linear neural net can approximate a 1d function arbitrarily wel. This also applies to a function with N input argumets and outputof vector with size M. It can be done with a single layer of neurons.

Train a network by minimizing the cost function through varying the values of the weights of the connected neurons. Minimization is by doing a gradient decent on the cost function. Only using a subset of input values at each step makes algorithm more stochastic on supposedly faster.

Since differentiating cost function of specific weight $w_i$ propagates through the whole network, calculation of the gradient wrt $w$ is expensive. However, we can use the chain rule!

### Backpropagation

To minimize the objective function, we perform a stochastic gradient descent (because each step, we only use a random sample of the data) in the space of the neural net weights (parameters).

We can calulcate this gradient numerically for all weights, but this is very expensive.

However, we can propagate the effect of chaning a weight through the neural net (from the top (output), till the layer where the weight enters the net) using the chain rule and matrix multiplication.

## Lecture 3

Hyper parameters: Number of layers, size of each layer, what non-linear function.

Understand what the network has learned:
Show output of each single neuron of the last layer.

Use Keras to speed up using neural nets.

Attempt to make a handwritten digits classifier:

28x28 pixels -> 10 classes. Output of net is preferrably probability: $\sum p_i = 1$.

How to enforce that output layer gives proper probability?
Softmax activation function: Introduce a non-linear function that depends
on all the values in the layer simulataneously. Use bolzmann weights.

What is a good cost function when you have output probabilities? Motivated by entropy: $S = \sum p_i log(p_i)$: $C = -\sum y^{\mathrm{target}}_j \mathrm{ln}(y^{\mathrm{out}}_j)$

Problem of overfitting: Always check your network on data it has *not* been trianed on.

## Lecture 4

How to prevent overfitting?

- Use validation set to varify accuracy of network.
- Stop after reaching maximum.
- Generate fresh training data by disturbing original data with noise / transate / scale / rotations etc,
- Dropouts: Randomnly set the nodes to zero during the training. ~10% This prevents overfitting.

### Convolutional neural networks

Great in case of translational invariance:

$$ F^{\mathrm{new}}(x) = \int K(x-x') F(x') dx'$$

New function in x depends on all the values of F in $x'$, weigted by the kernel $K$. Translational invariants required due to kernel argument $x-x'$.

Translated to the language of neural nets, this means that the value of the node in the next layer only depends on the weights of the neighbouring nodes in the grid (representing the pixels of the image.) Because of the translational invariance, the weights of the connections enetering a node in the next layer do not change from one node to the next.

### Unsupervised learning

Extracting crucial features without any guidance.

Example: autoencoder: input and output are same images of same size, while going through a small bottleneck: Compression!
Input image -> encodes -> bottleneck -> decoder -> output image.
Training method: gradually increase number of hidden layers, keepen previous layers fixed and training new layers, making bottleneck thinner.
First create optimal encoder-decoder, then take encoder and use it as the input for a dense soft-max weighted network for a classifier. And train dense layer with a supervised task.

In case of a single hidden layer, smaller than the input and without a non-linear function, how can I maximaize the information about the input layer in a smaller space?
In this case, the value of every neuron in the middle layer is the scalar product of a weight vector with the input vector. This amounts to a principle component analysis of the input data (PCA).

## Lecture 5

### PCA analysis

Input vector $|\psi>$ with components $\psi_m$. What is the statistics of the input vectors? Mean $<\psi_m>$ not interesting.
Second mode is intereting: $\rho_{mn} = <\psi_m \psi_n>$, averaged over all inout vectors $|\psi>$.

Claim: Second mode of input vectors is enough to construct optimal encoder. Matrix $\rho_{mn}$ is hermitian -> Diagonalize. Eigenvectors with largest eigenvalues build the subspace with the largest variance in $\psi_m$.

EXCESISE: Decompose pictures (digits) in eigenvectors (MNIST. Express pictures in first N components of the decomposition -> Do digits look similar to original?

### Visualisation and clustering of high dimensionnal data

t-SNE (Stochastic Neighbour embedding), clustering the images of digits. Using PCA there is no clusering (10 clusters). Can we do better?

Try to minimize $C$, where $x_i$ is vector in high dim space and $y_i$ on the lower dimensional space. Define cost function
$$C = \sum_{i\neq j} F(x_i - x_j,y_i - y_j) $$

To find $F$, define probability distributions $p_{ij}$ (high-dim) and $q_{ij}$ (low-dim) for picking two points labeled $i$ and $j$, resp. Defined such that picking points closer together is larger. Disr are normalized. We want $p$ and $q$ to be as similar as possible. We use the 'Kullback-Leibler divergence'.
$$C = KL(P||Q) = \sum_{i,j} p_{ij} \mathrm{log}\left( \frac{p_{ij}}{q_{ij}} \right)$$
We want to panalize larger distances in $p$ stronger than in $q$. We choose for $p$ a gaussian penalty on distance
$$p_{j|i} = \frac{\mathrm{exp}\left(-||x_i-x_j||^2/2\sigma_i^2 \right)}{\mathrm{normalization}} $$
and a slower quadratic decay is the low dim space.

Now, $\frac{C}{\delta y_i}$ describes a force in the low dim space wich can be used to move the points untill they minimize $C$.
These clusters canbe meaningfull. Results are different each time because stochastic.
You can use t-SNE to cluster neuron values in hidden layers of neural net that is trianed to classify images. Similar images can cluster.

### Improve the gradient decent for minimizing work function

Adaptive learning rate.
If you don't know anything else: You use adam!

### Neural net on timeseries data with Recurrant Neural Networks (RNN)

Convolutional net: Use time window with size K to find characterestic features in the data.
Long memories in the conv net are challenging. Very large filter sizes required. Trick: subsampling.

Setup: Many similar neural nets connected in sequence, output of each in turn determines a final output. Each subnet is perhaps used at diiferent points in time.
This results in many layers of the net. Backpropagation to trian network becomes problematic since subnets that are at early times have little effect on cost function. Very hard to optimize.

Solution: Long short-term memory. Long term memory of the net is in the weights on the net. Short term memory used after training.

Solution: RNN
Output at time $t$, does not only depend on the input of all the times before but also on the previous outputs: memory!

## Lecture 6: Recurrant Neural Networks

Network acting on time sequenced data: Need short term memory in network which depends on last input.
Network is the same for each time step, weights are the same. Optimize how they are connected through time.

Short term memory acts on the neural net, using delete, write and output.

### Delete of memory cell

Use that input vec is $\vec x_t$, and memory cells $\vec c_t$ and $\vec c_{t-1}$, and $\vec f = W\cdot x_t + \vec b$,
such that
$$\vec c_t = \vec f \cdot \vec c_t,$$
where we now multiply values of different neurons to get the value of the next neuron in time. This has implications for the back-propagation: product rule. Split in paths of the back propagation.

### Write to memory cell

$$\vec c_t = \vec f \cdot \vec c_t-1 + \vec i \cdot \vec c^*_t$$

First part is removing old memory, second is writing new data to memory, where $\vec c^*_t = \mathrm{tanh}(\vec W_c \cdot \vec x_t + \vec b_c)$.

### Output memory cell

The output of a memory cell $\vec c_t$ is
$$\vec h = \vec \cdot \mathrm{tanh}( \vec c_t )$$

### Examples of RNN

- Network trained on text predicts next character based on current and previous characters.
- Network that given two numbers and math operation symbol, learns the outcome.

### Word vectors

Not individual characters but act on whole words.

Naive approach: One hot encoding for input vectos (Given I dict of 1000 words, vector has size 1000.)
These vectos become meaningless and too long for real dictionary.

Solution: word vectors. Shorter real vectors where every word is a point in this space where words that are 'similar' are close together.
Words are related if they are are close together in a text.
Idea: Given a word at position $t$, $w(t)$ (one-hot encoding of dictionary of dim D), what is the probability of the next word $y(t)$? Use encoding $s[w(t)]$ that easiest lets you predict next word.
Still problem with very high dimensional vecors of dim D! Therefore, define $P_{\theta}(w_t|h)$: the probability of finding the word $w_t$ given the context words (near in the text) h with the parameters $\theta$. Find parameters by minimizing

$$ C = -\left( \mathrm{ln}(P_{\theta}(w_t|h)) + \sum_{\tilde w} \mathrm{ln}(P_{\theta}(\tilde w|h)) \right)$$

where $\tilde w$ are a set of random wrong words given the context.

GloVe word embeddings: Encoding 400k words using 100 dim real word vectors found through training on a 800MB sized English wikipedia.

## Lecture 7:Reinforcement learning

No training data. Only know correct outcome.
Interaction and feedback between agent and its environment. Response of environemnt to agents actions and resulting reward are unknown a-priori: Model free reinforcement learning! When we can not anwnser the queastion: what would the enviroment have done when we had chosen a different action.

Define probability distribution $\pi_{\theta}(a_t|s_t)$: Probability of taking an action $a$, given the state of the environment $s$ at time $t$, with parameters $\theta$.

How do we take changes in the state of the enivorment into account? Define transition function $P(s'|s,a)$, the probability that the enivorment is in state $s'$, given that the environment is now in state $s$ and I take action $a$. Could be deterministic.

We can now define the propability of a certain trajectory $\tau$ of system states and actions $(\mathbf{s}, \mathbf{a})$:

$$P_{\theta}(\tau) = \Pi_t P(s_{t+1}|t_t, a_t)\pi_{\theta}(a_t|s_t)$$

The expected return averaged over all trajectories

$$\bar R = E[R] = \sum_{\tau} P_{\theta}(\tau)R(\tau)$$

We now want to maximize $\bar R$ with respect to system parameters $\theta$,
$\partial \bar R/\partial \theta = ?$ Some standard math trics, we get

$$\frac{\partial\bar R}{\partial \theta} = \sum_t E\left[ R \frac{\partial \mathrm{ln} \pi_{\theta}(a_t|s_t)}{\partial \theta} \right]$$

And then perform the usual gradient decent

$$\Delta\theta = \eta \frac{\partial \bar R}{\partial \theta}$$

(What is unsupervised learning?)
Where do the word vectprs cp,me from.

## Lecture 8: Reinforement learning

*Example*: 1D Walker on the look for a specific site.

- Input: (Local) System state $s_t$ (0,1 = on target site)
- Output: Action $a_t$: (0 = stay,1 = move) Softmax

Policy gradient: all the steps:

1. Execute action, record new state $s_t$.
2. Apply NN to $s_t \to \pi(a_t|s_t)$.
3. From $\pi$, obtain $a_t$.

For each trajectory:

1. Execute one (batch of) trajectory.
2. Obtain R for each trajectory.
3. Apply policy gradient training

Explaining simple net and AlphaGo

### Q-learning (alt. to policy grad.)

Quality function, $Q(a, s_t)$: Predicts future reward given current state state and action.
Ex.: Gives the value of every state given the action in that state.

Value function $V(s)$: assigns value to a given state $s$.

$$Q(s_t, a_t) = E[R_t|s_t, a_t]$$

Expected return in the future, given the current state $s$ and action $a$, assuming actions $a$ in the future are taken that maximize the expected return/ largest Q.

- $R_t$ all the weighted future rewards after $t$: $R_t = \sum_{t'} r_{t'} \gamma^{t'-t}$, discounting future rewards. Greedy algorithm.
- $V(s) = \mathrm{max}_a Q(s,a)$

How to find $Q$? The Bellmann equations:
$$ Q(s_t, a_t) = E[r_t + \gamma \mathrm{max}_a Q(s_{t+1}, a)|s_t, a_t]$$
Still have no explicit expression for Q. Itteratively obtain it through initial guess $Q^{\mathrm{old}}$

$$Q^{\mathrm{new}}(s_t, a_t) = Q^{\mathrm{old}}(s_t, a_t) + \alpha\left( r_t + \gamma \mathrm{max}_a Q^{\mathrm{old}}(s_{t+1}, a) - Q^{\mathrm{old}}(s_t, a_t) \right)$$

## Lecture 9: Connection with physics / Bolzmann machines

Bolzmann machine: use to model probability distributions. Use NN to produce previously unseen examples, according to the probability distribution of the training samples.

Example of Mone Carlo Sampeling of the states $s$ with transition propensities $P(s'|s)$. If the trnasition propensities are correct, the system can reach equilibrium.
This means there exists a probability distribution $P(s)$ such that

$$\frac{P(s|s')}{P(s'|s)} = \frac{P(s)}{P(s')} $$

and thus detailed balance is maintained.

In (non-kinetic) Mone Carlo simulation, choose forward rate 1 and backward rate $exp(\Delta E / (kT))$. This way, detailed balance is automatically maintained in the model. There is however no kinetic
but sampling is done via bolzmann statistics.

## Restricted Bolzmann machine (RBM)

- Type of Neural Network: Value of the neurons are discrete (0 or 1).
- No connections between neurons in the same layer.
- Visisble units $\vec v$ (input layer) and hidden layers $\vec h$.
- Interaction beween units via Ising model with coupling $w$ and external field $a$ and $b$:

$$E(\vec v, \vec h) = -\sum_i v_i a_i - \sum_j h_j b_j - \sum_{i,j} W_{ij} v_i h_j $$

Where we want $P(h,v) = \frac{\mathrm{exp}(-E(v,h))}{Z}$ and $P(v) = \sum_h P(v,h)$.

Choose weights and biases such that $P(v) \sim P_0(v)$, where $P_0$ is the distr of the sample data.

Hidden units are a higher level repesentation of the visbile units (color, race, angle, light....) -> picture of dog.

## Lecture 10

### Neural networks applied to scientific tasks: some examples

## Lecture 11: Research I do in NN

## Useful Markdown code

_Hallo_

**Hallo**

`Hello world`

```python
def my_function(my_var):
     a = my_var + 1
     return a
```

```c++
void main(){
    double a,b;
    a = b + 1;
}
``` 
