# Perceptron

- Perceptron is a type of artificial neuron that simulates human intelligence by mimicking the complex web of neurons.
- The perceptron model multiplies each input feature by a learnable weight and sums them up; if the sum exceeds a threshold, the perceptron gets "excited" and outputs a signal.
- The perceptron learning rule updates weight coefficients automatically for decision-making, based on the difference between the actual and predicted outputs.
- The perceptron's ability to classify inputs correctly improves by adjusting its weights based on the training data.
- Adaline (ADAptive LInear NEuron) is another type of single-layer neural network that minimizes errors through a loss/objective function.
- The main difference is Adaline uses a linear activation function, enabling gradient descent training by making the loss function differentiable.
- Proper selection of the learning rate is crucial for the successful training of both the perceptron and Adaline models.
- Scale standardization, such as Z-score normalization, boosts model performance by balancing input feature contributions.
- The perceptron model may perform suboptimally when classifying data with features of varying scales, as this can affect the model's ability to learn effectively.
- Z-score normalization standardizes input data features to a mean of 0 and a standard deviation of 1, enhancing model performance.

### Example Questions
1. What is the primary difference between the perceptron and Adaline models?
   - [ ] Perceptron uses a linear activation function, while Adaline uses a step function.
   - [ ] Perceptron uses a step function as activation, while Adaline uses a linear function.
   - [ ] Perceptron cannot be trained using gradient descent, while Adaline can.
   - [ ] There is no significant difference; both models are the same.
   > Answer: Perceptron uses a step function as activation, while Adaline uses a linear function.

2. Why is data scaling important in the context of perceptron and Adaline models?
   - [ ] It increases the computational complexity of the models.
   - [ ] It decreases the accuracy of the models.
   - [ ] It ensures that all input features contribute equally to the prediction.
   - [ ] It is not important; models perform the same with or without scaling.
   > Answer: It ensures that all input features contribute equally to the prediction regardless of their scale.

3. Which of the following statements about the learning rate during the training of perceptron and Adaline models is true?
   - [ ] A high learning rate guarantees faster convergence to the optimal weights.
   - [ ] An improperly chosen learning rate can cause the model to fail to converge or to converge too slowly.
   - [ ] The learning rate has no significant impact on the training process.
   - [ ] The learning rate should be set as high as possible to minimize training time.
   > Answer: An improperly chosen learning rate can cause the model to fail to converge or to converge too slowly.

# Logistic Regression

- Logistic regression is a classification model that computes the probability of a sample belonging to a positive class using a logistic sigmoid function.
- The logistic sigmoid function is characterized by its S-shaped curve, which outputs a probability value between 0 and 1.
- Logistic regression's probability is given by \(p = \frac{1}{1 + \exp(-z)}\), where \(z\) is the weighted sum of input variables plus a bias term.
- Maximum Likelihood Estimation (MLE) is used in logistic regression to find the parameters that most likely explain the given data.
- The log-likelihood function, used in MLE, is concave, ensuring a unique maximum for the best parameters.
- Gradient descent algorithm is employed to optimize the log-likelihood function, updating weights to maximize the likelihood of observing the given data.
- Regularization, such as L2 regularization, is applied to prevent overfitting by penalizing large weights in the model.
- The weight update rule in logistic regression with L2 regularization includes a term to penalize large weights, ensuring the model does not overly depend on any single feature of the data.
- Dataset imbalance can degrade model performance by favoring the majority class.
- Using stratification in train-test splits maintains consistent class ratios, mitigating imbalance.
- Increasing regularization strength in Logistic Regression helps avoid overfitting by penalizing larger weights.
- L2 regularization simplifies the model by penalizing weight squares, aiding in overfitting prevention.


### Example Questions
1. What is the primary purpose of using a logistic sigmoid function in logistic regression?
   - [ ] To convert the regression output into a binary classification
   - [ ] To ensure the output is a probability value between 0 and 1
   - [ ] To increase the computational complexity of the model
   - [ ] To reduce the likelihood of overfitting
   > Answer: To ensure the output is a probability value between 0 and 1

2. Why is regularization used in logistic regression?
   - [ ] To increase the model's accuracy on the training data
   - [ ] To penalize large weights and prevent overfitting
   - [ ] To speed up the gradient descent optimization process
   - [ ] To convert the regression model into a classification model
   > Answer: To penalize large weights and prevent overfitting

3. What is the role of the gradient descent algorithm in logistic regression?
   - [ ] To calculate the probability of the positive class
   - [ ] To minimize the model's bias
   - [ ] To optimize the log-likelihood function and find the best parameters
   - [ ] To transform the input features using the logistic sigmoid function
   > Answer: To optimize the log-likelihood function and find the best parameters

# Dimensionality Reduction

- Dimensionality reduction aids in data compression and accelerates computation times.
- Facilitates data visualization by reducing dimensions.
- Minimizes the risk of model overfitting.
- Linear projection is akin to taking a photo, projecting a 3D object onto a 2D screen.
- Principal Component Analysis (PCA) maximizes data variation.
- Linear Discriminant Analysis (LDA) maximizes between-class variations while minimizing within-class variations.
- Non-linear dimensionality reduction techniques capture complex, non-linear structures in the data.
- Multidimensional Scaling (MDS) preserves global distances between data points.
- Isomap focuses on preserving local structure, minimizing the Kullback-Leibler divergence between distributions.
- t-SNE and UMAP approximates the manifold structure in a low-dimensional space, preserving both local and global data structure.
- The choice of dimensionality reduction technique depends on the desired preservation of data structure (global vs. local).


### Example Questions
1. What is the primary goal of dimensionality reduction?
   - [ ] To increase the computational complexity of models
   - [ ] To aid in data compression and accelerate computation times
   - [ ] To increase the number of features in a dataset
   - [ ] To enhance the non-linear separability of the data
   > Answer: To aid in data compression and accelerate computation times

2. Which technique focuses on preserving the global structure of the data?
   - [ ] t-SNE
   - [ ] PCA
   - [ ] LDA
   - [ ] UMAP
   > Answer: PCA

3. What is the main difference between PCA and LDA?
   - [ ] PCA is supervised while LDA is unsupervised
   - [ ] LDA focuses on maximizing between-class variations, PCA does not
   - [ ] PCA can be used for non-linear dimensionality reduction
   - [ ] LDA is used for data compression, PCA is not
   > Answer: LDA focuses on maximizing between-class variations, PCA does not

4. Which of the following is true about MDS, t-SNE and UMAP?
   - [ ] All are linear dimensionality reduction techniques
   - [ ] t-SNE preserves global distances, while UMAP focuses on local distances
   - [ ] t-SNE and UMAP preserves both global and local structures
   - [ ] MDS preserves global distances
   > Answer: t-SNE and UMAP preserves both global and local structures
   > Answer: MDS preserves global distances

# Best practices

- Cross-validation is used to estimate model performance for unseen data by dividing data into multiple folds.
- K-Fold cross-validation involves splitting the dataset into K folds, training on K-1 folds, and testing on the remaining fold.
- Model performance in cross-validation is determined by averaging scores across all iterations.
- Precision measures the proportion of true positive predictions in all positive predictions.
- Recall measures the proportion of true positive predictions out of all actual positives.
- When is recall preferred over precision?
- F1-score is the harmonic mean of precision and recall, providing a balance between them.
- Accuracy calculates the proportion of true predictions (both true positives and true negatives) out of all predictions.
- AUC-ROC curve represents model performance across different thresholds, with an area of 1 indicating perfect prediction.
- Average Precision (AP) evaluates the precision-recall balance across different thresholds, useful in imbalanced datasets.
- Replicability ensures the same results using the same code and data, while reproducibility involves generating results independently.
- Data provenance documents the history of data from creation to its current version, ensuring trust and reproducibility.
- Containers like Docker and Singularity help in achieving consistent environments, enhancing reproducibility.
- Explicit typing of data during loading is crucial to prevent type mismatches, ensuring data integrity and analysis accuracy.
- Nominal data is a type of data that categorizes without implying order, used for classification or grouping.
- Ordinal data is a type of data that categorizes with an order, used for ranking or ordering.


### Example Questions
1. What is the primary purpose of cross-validation?
   - [ ] To increase the size of the dataset
   - [ ] To estimate model performance on unseen data
   - [ ] To reduce the computational complexity
   - [ ] To enhance the model's accuracy on the training set
   > Answer: To estimate model performance on unseen data

2. Which metric is the harmonic mean of precision and recall?
   - [ ] Accuracy
   - [ ] F1-score
   - [ ] AUC-ROC
   - [ ] Average Precision
   > Answer: F1-score

3. What does AUC-ROC stand for?
   - [ ] Area Under the Curve - Receiver Operating Characteristic
   - [ ] Average User Calculation - Rate of Change
   - [ ] Area Under the Curve - Rate of Change
   - [ ] Average User Calculation - Receiver Operating Characteristic
   > Answer: Area Under the Curve - Receiver Operating Characteristic

4. What is the difference between replicability and reproducibility?
   - [ ] Replicability involves generating results independently, while reproducibility uses the same code and data.
   - [ ] Replicability and reproducibility are the same.
   - [ ] Replicability uses the same code and data for the same results, while reproducibility involves generating results independently.
   - [ ] There is no difference; both terms are outdated.
   > Answer: Replicability uses the same code and data for the same results, while reproducibility involves generating results independently.

5. What is data provenance?
   - [ ] The history of data from creation to its current version, ensuring trust and reproducibility.
   - [ ] The current version of data, with no history.
   - [ ] The history of data from creation to its current version, with no current version.
   - [ ] The current version of data, with its history.
   > Answer: The history of data from creation to its current version, ensuring trust and reproducibility.

6. When is Recall preferred over Precision?
   - [ ] When the cost of false positives is higher than false negatives
   - [ ] When the cost of false negatives is higher than false positives
   - [ ] When the cost of false positives and false negatives are the same
   - [ ] When the cost of false positives and false negatives are not the same
   > Answer: When the cost of false negatives is higher than false positives

# word2vec

- word2vec is a neural network model that maps words into compact vectors.
- The principle behind word2vec is "You shall know a word by the company it keeps".
- Word vectors in word2vec are semantically meaningful, allowing operations like "king - man + woman = queen".
- word2vec architecture consists of one input layer, one hidden layer, and one output layer.
- The input to word2vec is a one-hot vector representing a word.
- The output of word2vec is the probability of a word appearing within a certain distance from the input word, calculated using softmax.
- Softmax function converts real numbers in the final layer into probabilities.
- Training word2vec with Negative Sampling addresses the computational expense of the softmax function by simplifying the training process.
- Negative Sampling involves selecting word pairs within a certain distance and classifying them using the sigmoid function based on their similarity.

### Example Questions
1. What is the primary learning principle behind word2vec?
   - [ ] Mapping words into vectors based on their alphabetical order
   - [ ] You shall know a word by the company it keeps
   - [ ] Words are represented as one-hot vectors
   - [ ] All words in a corpus have the same vector representation
   > Answer: You shall know a word by the company it keeps

2. What does the output layer of word2vec represent?
   - [ ] The alphabetical order of words
   - [ ] The probability of a word appearing within a certain distance from the input word
   - [ ] A binary classification of words
   - [ ] The frequency of words in the corpus
   > Answer: The probability of a word appearing within a certain distance from the input word

3. Which function is used in word2vec to convert real numbers into probabilities?
   - [ ] Sigmoid function
   - [ ] Tanh function
   - [ ] ReLU function
   - [ ] Softmax function
   > Answer: Softmax function

4. What is the purpose of Negative Sampling in training word2vec?
   - [ ] To increase the size of the dataset
   - [ ] To simplify the training process by addressing the computational expense of softmax
   - [ ] To convert word vectors into one-hot vectors
   - [ ] To ensure all words have the same vector representation
   > Answer: To simplify the training process by addressing the computational expense of softmax

# Mechanics of PyTorch

- Neural networks are computational models that mimic the brain's structure and function, consisting of interconnected nodes or neurons.
- Each neuron in a neural network processes input data, applies a weight, and passes it through an activation function to produce an output.
- The learning process in neural networks involves adjusting the weights of connections based on the error between the predicted and actual outputs.
- Non-linear activation functions like ReLU, Sigmoid, and Tanh enable the network to learn complex patterns.
- Backpropagation is a method used in training neural networks, where the error is propagated back through the network to update the weights.
- Gradient descent is an optimization algorithm used to minimize the error in neural networks by iteratively adjusting the weights.
- Overfitting occurs when a neural network learns the training data too well, including its noise, leading to poor performance on new, unseen data.
- Dropout is a regularization technique used to prevent neural networks from overfitting.
- It works by randomly setting a fraction of input units to 0 at each update during training time, which helps to break up happenstance correlations in the training data.
- The dropout rate is a hyperparameter that determines the probability of an input unit being excluded from a training update.
- Dropout simplifies the network during training and uses the complete network with adjusted weights for testing.
- Dropout helps in making the model more robust and less sensitive to the specific weights of neurons, which contributes to a better generalization on unseen data.


### Example Questions
1. What is the primary function of an activation function in a neural network?
   - [ ] To prevent overfitting
   - [ ] To adjust the weights of the connections
   - [ ] To determine whether a neuron should be activated
   - [ ] To propagate the error back through the network
   > Answer: To determine whether a neuron should be activated

2. Which technique is commonly used to prevent overfitting in neural networks?
   - [ ] Activation functions
   - [ ] Backpropagation
   - [ ] Gradient descent
   - [ ] Dropout
   > Answer: Regularization

3. What is the role of backpropagation in neural network training?
   - [ ] To activate neurons based on their input
   - [ ] To minimize the error by adjusting weights
   - [ ] To apply a penalty on the size of the weights
   - [ ] To process data with a grid-like topology
   > Answer: To minimize the error by adjusting weights

# RNNs

- Recurrent Neural Networks (RNNs) are designed to process sequential data, capturing temporal dependencies.
- RNNs incorporate a memory function that retains information from previous inputs to influence future outputs.
- The basic structure of an RNN includes an input layer, one or more hidden layers, and an output layer.
- Hidden states in RNNs serve as the network's memory, carrying information across processing steps.
- RNNs face challenges with long-term dependencies due to the vanishing gradient problem, where gradients become too small for effective learning.
- The vanishing gradient problem is a significant challenge in training RNNs, making it difficult for the model to learn long-term dependencies.
- Long Short Term Memory (LSTM) networks are a type of RNN designed to address the vanishing gradient problem by introducing gates that regulate information flow.
- LSTMs incorporate mechanisms called gates, including input, forget, and output gates, which regulate the flow of information and allow the network to learn what to remember and what to forget.
- The cell state in LSTM is a vector that carries relevant information throughout the processing steps.


### Example Questions
1. What is the primary purpose of hidden states in RNNs?
   - [ ] To directly predict the output
   - [ ] To serve as the network's memory
   - [ ] To increase the model's computational speed
   - [ ] To reduce the dimensionality of the input data
   > Answer: To serve as the network's memory

2. How do LSTMs address the vanishing gradient problem in RNNs?
   - [ ] By reducing the complexity of the model
   - [ ] By introducing gates that regulate information flow
   - [ ] By increasing the size of the hidden layers
   - [ ] By shortening the sequence length of the input data
   > Answer: By introducing gates that regulate information flow

3. Why do LSTMs include a forget gate in their architecture?
   - [ ] To reset the cell state at each time step
   - [ ] To selectively remember or forget information from the cell state
   - [ ] To increase the computational speed of the network
   - [ ] To decrease the number of parameters in the model
   > Answer: To selectively remember or forget information from the cell state

# seq2seq

- Seq2Seq models consist of an Encoder and a Decoder for processing sequences.
- The Encoder generates hidden states from the input sequence.
- The Decoder uses the last hidden state to generate the output sequence.
- GRU (Gated Recurrent Unit) simplifies LSTM by omitting the cell state.
- Multi-layered GRU means GRU units are stacked, taking the previous layer's hidden state as input.
- Teacher forcing is a strategy where the Decoder is conditioned by the ground-truth target tokens during training.
- Dropout is applied to input embedding vectors to prevent overfitting.
- The softmax function in the Decoder assigns probabilities to each word in the output sequence.
- The initial hidden state of the Decoder is conditioned by the Encoder's last hidden state.
- Attention mechanisms help Seq2Seq models produce better translations for longer sentences by focusing on specific parts of the input for each output step.
- Bidirectional Encoders improve context understanding by processing the input forwards and backwards.
- Using [SOS] and [EOS] tokens at the start and end of the target sequence during training clearly marks the beginning and end for the Decoder, aiding in creating coherent outputs.

### Example Questions
1. What is the primary function of the Encoder in a Seq2Seq model?
   - [ ] To generate the output sequence directly
   - [ ] To process the input sequence and generate hidden states
   - [ ] To apply dropout to the input embedding vectors
   - [ ] To assign probabilities to each word in the output sequence
   > Answer: To process the input sequence and generate hidden states

2. How does teacher forcing affect the training of a Seq2Seq model?
   - [ ] It slows down the training process
   - [ ] It conditions the Decoder with the model's own predictions
   - [ ] It conditions the Decoder with the ground-truth target tokens
   - [ ] It increases the likelihood of overfitting
   > Answer: It conditions the Decoder with the ground-truth target tokens

3. Which component is responsible for generating the output sequence in a Seq2Seq model?
   - [ ] Encoder
   - [ ] Multi-layered GRU
   - [ ] Decoder
   - [ ] Dropout layer
   > Answer: Decoder

# BERT

- Transformers eliminate the need for RNNs by using attention mechanisms to process multiple tokens simultaneously.
- The self-attention mechanism in transformers allows each token to interact with every other token in the input sequence.
- BERT (Bidirectional Encoder Representations from Transformers) generates contextual embeddings for tokens by considering both left and right context.
- Transformers are parallelizable, making them faster to train compared to RNNs.
- BERT combines token embeddings, segment embeddings, and position embeddings to understand the context of words in a sentence.
- The attention mechanism in transformers helps in capturing long-term dependencies in the data.
- Transformers architecture is based on the principle that "Attention is All You Need".
- BERT has been pre-trained on a large corpus of text and can be fine-tuned for various NLP tasks.


### Example Questions
1. What is the primary advantage of transformers over RNNs?
   - [ ] Faster training due to parallelization
   - [ ] Better at capturing short-term dependencies
   - [ ] Do not require pre-training
   - [ ] Simpler architecture
   > Answer: Faster training due to parallelization

2. How does BERT generate contextual embeddings?
   - [ ] By using only token embeddings
   - [ ] By considering the position of tokens in the sequence
   - [ ] By leveraging the attention mechanism to mix the vector of each token
   - [ ] By using a fixed embedding for each token
   > Answer: By leveraging the attention mechanism to mix the vector of each token

3. What is a key feature of the self-attention mechanism in transformers?
   - [ ] It processes one token at a time
   - [ ] It allows each token to interact with every other token
   - [ ] It ignores the context of the sentence
   - [ ] It uses a fixed attention pattern for all tokens
   > Answer: It allows each token to interact with every other token
