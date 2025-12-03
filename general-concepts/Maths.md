## Core Concepts in Classification & Probabilistic Modeling

Here's a breakdown of Negative Log Likelihood, Logit, Softmax, and Log Softmax, commonly used in machine learning, especially for classification tasks.

---

### 1. Negative Log Likelihood (NLL)

*   **Concept:**
    NLL is a **loss function** used to evaluate how well the predicted probabilities of a model match the actual outcomes (ground truth). It quantifies the "cost" of the model's predictions. Training a model involves *minimizing* this NLL. Minimizing NLL is equivalent to maximizing the likelihood of observing the true data given the model's predictions (Maximum Likelihood Estimation - MLE).

*   **Purpose:**
    Primarily used for training classification models and other probabilistic models by adjusting model parameters to reduce the loss.

*   **Equation(s):**
    Let $C$ be the number of classes.
    Let $\mathbf{y}$ be the true label, often represented as a one-hot vector $[y_1, y_2, \dots, y_C]$ where $y_k=1$ for the true class and $0$ otherwise.
    Let $\mathbf{p} = [p_1, p_2, \dots, p_C]$ be the vector of predicted probabilities from the model, where $p_k$ is the predicted probability for class $k$, and $\sum_{k=1}^C p_k = 1$.

    *   **NLL for a single data point:** If the true class is $i$, the likelihood is $p_i$. The NLL is:
        $$
        \text{NLL}(\mathbf{y}, \mathbf{p}) = - \log(p_i)
        $$
        Using the one-hot vector $\mathbf{y}$:
        $$
        \text{NLL}(\mathbf{y}, \mathbf{p}) = - \sum_{k=1}^C y_k \log(p_k)
        $$
        (Note: Since only one $y_k$ is 1, this sum collapses to the log probability of the true class).

    *   **Average NLL over a dataset** of $N$ points: Let $\mathbf{y}^{(n)}$ and $\mathbf{p}^{(n)}$ be the true label and predicted probabilities for the $n$-th data point.
        $$
        \text{NLL}_{\text{dataset}} = - \frac{1}{N} \sum_{n=1}^N \sum_{k=1}^C y_k^{(n)} \log(p_k^{(n)})
        $$

---

### 2. Logit

*   **Concept:**
    The logit function is the **logarithm of the odds**. The odds are the ratio of the probability of an event occurring ($p$) to the probability of it not occurring ($1-p$). The logit function maps a probability $p \in (0, 1)$ to the entire real number line $(-\infty, +\infty)$.

*   **Purpose:**
    It's the core transformation in **logistic regression**. In neural networks, the raw, unnormalized outputs of the final linear layer (before activation functions like sigmoid or softmax) are often referred to as "logits". It's the inverse of the standard logistic (sigmoid) function.

*   **Equation(s):**
    For a probability $p$:
    $$
    \text{logit}(p) = \log\left(\frac{p}{1-p}\right)
    $$
    Where $\log$ typically denotes the natural logarithm (ln).
    The inverse function (Sigmoid or Logistic function) maps a logit $z$ back to a probability:
    $$
    \sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z} = p
    $$

---

### 3. Softmax

*   **Concept:**
    Softmax is an **activation function** that converts a vector of arbitrary real-valued scores (logits) into a vector representing a probability distribution. The outputs are non-negative and sum to 1.

*   **Purpose:**
    Used typically in the output layer of a neural network for **multi-class classification** problems, transforming the model's raw scores into probabilities for each class.

*   **Equation(s):**
    Let $\mathbf{z} = [z_1, z_2, \dots, z_K]$ be the input vector of $K$ scores (logits).
    The Softmax function applied to the $i$-th element $z_i$ yields the probability $p_i$:
    $$
    p_i = \text{softmax}(\mathbf{z})_i = \frac{\exp(z_i)}{\sum_{j=1}^K \exp(z_j)}
    $$
    The output vector is $\mathbf{p} = [p_1, p_2, \dots, p_K]$, where $\sum_{i=1}^K p_i = 1$ and $p_i \ge 0$ for all $i$.

---

### 4. Log Softmax

*   **Concept:**
    Log Softmax computes the logarithm of the Softmax output directly. Instead of probabilities, it outputs log-probabilities.

*   **Purpose:**
    It's often used in combination with the NLL loss function for training classification models. This combination (`LogSoftmax` followed by `NLLLoss`) is mathematically equivalent to **Cross-Entropy Loss** but provides better **numerical stability** and computational efficiency compared to calculating Softmax and then taking the logarithm.

*   **Equation(s):**
    Let $\mathbf{z} = [z_1, z_2, \dots, z_K]$ be the input vector of $K$ scores (logits).
    The Log Softmax value for the $i$-th element is:
    $$
    \text{log\_softmax}(\mathbf{z})_i = \log\left( \text{softmax}(\mathbf{z})_i \right)
    $$
    Substituting the Softmax formula:
    $$
    \text{log\_softmax}(\mathbf{z})_i = \log\left( \frac{\exp(z_i)}{\sum_{j=1}^K \exp(z_j)} \right)
    $$
    Using the property $\log(a/b) = \log(a) - \log(b)$:
    $$
    \text{log\_softmax}(\mathbf{z})_i = \log(\exp(z_i)) - \log\left( \sum_{j=1}^K \exp(z_j) \right)
    $$
    Simplifying $\log(\exp(z_i))$:
    $$
    \boxed{ \text{log\_softmax}(\mathbf{z})_i = z_i - \log\left( \sum_{j=1}^K \exp(z_j) \right) }
    $$
    The term $\log\left( \sum_{j=1}^K \exp(z_j) \right)$ is the **LogSumExp** function, often implemented with numerical stabilization techniques (like subtracting the maximum logit before exponentiating).

---

**Relationships:** In many deep learning frameworks, the `CrossEntropyLoss` function implicitly performs `LogSoftmax` on the input logits and then calculates the `NLLLoss`. This is the standard and numerically preferred way to train multi-class classification networks.

---

## 5. Dot Product

*   **Concept:**
    The **dot product** (also called the *inner product* or *scalar product*) is a fundamental operation in linear algebra. It measures the degree of alignment between two vectors. A large positive value indicates the vectors point in a similar direction; zero means they are orthogonal; a large negative value means they point in opposite directions.

    In machine learning, the dot product appears everywhere: linear layers, attention mechanisms, similarity measures, gradient calculations, and geometric interpretations of embeddings.

*   **Purpose:**
    * Computes weighted sums in neural networks (e.g., linear layers: ( Wx + b )).
    * Measures similarity between vectors (e.g., cosine similarity uses a normalized dot product).
    * Projects one vector onto another.
    * Forms the basis of matrix multiplication.

*   **Equation(s):**

    Let $\mathbf{a} = [a_1, a_2, \dots, a_n]$, $\mathbf{b} = [b_1, b_2, \dots, b_n]$
    be two vectors of equal dimension ( n ).

*   **Dot product definition:**

    Let  
    $\mathbf{a} = [a_1, a_2, \dots, a_n]$ and $\mathbf{b} = [b_1, b_2, \dots, b_n]$.

    The dot product is:
    $$
    \mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^n a_i b_i
    $$

    Expanded form:
    $$
    \mathbf{a} \cdot \mathbf{b}
    = a_1 b_1 + a_2 b_2 + \dots + a_n b_n
    $$

*   **Geometric interpretation:**
    $$
    \mathbf{a} \cdot \mathbf{b}
    = \|\mathbf{a}\|\, \|\mathbf{b}\| \cos(\theta)
    $$

    where $\theta$ is the angle between the vectors, and
    $$
    \|\mathbf{a}\| = \sqrt{\sum_{i=1}^n a_i^2}
    $$
    is the Euclidean norm.

    This form shows that the dot product encodes *directional similarity*. When:

    * $\theta = 0^\circ$: vectors perfectly aligned → dot product is maximal.  
    * $\theta = 90^\circ$: vectors orthogonal → dot product is zero.  
    * $\theta = 180^\circ$: vectors opposite → dot product is negative.

*   **Matrix form (common in deep learning):**

    If $\mathbf{W}$ is a weight matrix and $\mathbf{x}$ is an input vector:
    $$
    \mathbf{W}\mathbf{x}
    $$
    is computed as a series of dot products between the rows of $\mathbf{W}$ and the vector $\mathbf{x}$.

---

**Relationship:**  
    The dot product is the building block of linear transformations, attention mechanisms (via **query·key** similarity scores), embeddings, and most neural network computations. Without dot products, deep learning would not exist in its current

---