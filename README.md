# **Machine Learning**
**Machine Learning is a subset of Artificial Intelligence. Machine Learning is the study of making machines more human-like in their behavior and decisions making by giving them the ability to learn and develop their own programs.**

### **Difference between Traditional Programming and Machine Learning**
- **Traditional programming and machine learning represent two different approaches to solving problems in computer science, each with its own methodologies, techniques, and applications.**
    - **Traditional Programming:** 
        - It's like giving direct instructions to someone. 
        - Imagine you're giving step-by-step instructions to your friend on how to bake a cake. You tell them exactly what ingredients to use, how much of each, and the precise steps to follow - like mix flour, sugar, and eggs, then bake at a certain temperature for a certain time. 
        - You have to lay out every detail.

    - **Machine Learning:** 
        - Now, imagine teaching your friend to recognize a good cake recipe by showing them lots of examples of good and bad cakes. You don't give them explicit instructions on what makes a cake good or bad, but they start to notice patterns on their own. They might learn that cakes with certain ingredients or baked for a certain amount of time tend to be better. 
        - Machine learning works like that - you provide lots of examples (data) and algorithms help the computer learn to make predictions or decisions without explicitly telling it what rules to follow.
          
### **Types of Machine Learning Algorithms**
- **Machine learning algorithms can be categorized into three types based on the way they learn and make predictions:**
    - **Supervised Learning**
        - **In supervised learning, the algorithm learns from labeled data, meaning each input data point is paired with a corresponding target label.**
        - **The goal is to learn a mapping from inputs to outputs based on the labeled examples provided during training.**
        - **Common supervised learning algorithms include:**
            - **Linear Regression**
            - **Logistic Regression**
            - **SVM**
            - **Decision Tree**

    - **Unsupervised Learning**
        - **In unsupervised learning, the algorithm learns from unlabeled data, meaning there are no target labels associated with the input data.**
        - **The goal is to uncover hidden patterns or structures in the data.**
        - **Common unsupervised learning algorithms include:**
            - **Clustering**
            - **Association Rule Learning**

    - **Reinforcement Learning**
        - **In reinforcement learning, the algorithm learns to make decisions by interacting with an environment. It learns from trial and error, receiving feedback in the form of rewards or penalties.**
        - **The goal is to learn a policy or strategy that maximizes cumulative reward over time.**
        - **Common reinforcement learning algorithms include:**
            - **Q-Learning**
            - **Deep Q Networks (DQN)**
    
# **Supervised Learning**
## **Linear Regression**
- **Linear regression is a statistical method used to model the relationship between a dependent variable y and one or more independent variables X.**
- **It assumes that this relationship is approximately linear, meaning that changes in X are associated with constant changes in y.**
### **Linear Regression Process:**
Suppose we have the following dataset:

|($\ x$) | ($\ y$)|
|----|----|
| 1  | 7  |
| 2  | 8  |
| 2  | 7  |
| 3  | 9  |
| 4  | 11 |
| 5  | 10 | 
| 5  | 12 |

**We want to fit a linear regression model to this data, where $\ x$ is the independent variable and $\ y$ is the dependent variable. Our goal is to find the equation of the line that best fits these data points.**

- **Hypothesis: $\ 𝑌=𝑚𝑋+𝑐 \to ℎ_θ (𝑥) = θ_0+θ_1 𝑋_1$**
- **Parameters: $\ θ_0, θ_1$**
- **Cost function: $\ 𝐽(𝑥) = \frac{1}{2𝑚} ∑_i[(ℎ_𝜃 (𝑥_𝑖)−𝑦_𝑖)]^2 $**
- **Goal: minimize  $\ 𝐽(𝑥)$**
    
    **$\ 𝜃_𝑖≔𝜃_𝑖−𝛼 \frac{1}{2𝑚} ∑_i[((ℎ_𝜃 (𝑥_𝑖 )−𝑦_𝑖 ) 𝑥_𝑖)]$**

**Suppose:**
- **$ℎ_θ (𝑥) = θ_0+θ_1 𝑋_1$**
- **$θ_0 = 5, θ_1 = 2$**
- **$α=0.01$**

| $X$ | $Y$ | $ℎ_θ (𝑥)$ | $ℎ_θ (𝑥) - y$ | $[(ℎ_𝜃 (𝑥_𝑖)−𝑦_𝑖)]^2 $|
|-----|-----|-----------|---------------|----------------------|
|  1  |  7  |    7      |       0       |           0          |
|  2  |  8  |    9      |       1       |           1          |
|  2  |  7  |    9      |       2       |           4          |
|  3  |  9  |    11     |       2       |           4          |
|  4  |  11 |    13     |       2       |           4          |
|  5  |  10 |    15     |       5       |           25         |
|  5  |  12 |    15     |       3       |           9          |

**$\ 𝐽(𝑥) = \frac{1}{2𝑚} ∑_i[(ℎ_𝜃 (𝑥_𝑖)−𝑦_𝑖)]^2 $**

**$\ 𝐽(𝑥) = \frac{1}{14} (0 + 1 + 4 + 4 + 4 + 25 + 9)$**

**$\ 𝐽(𝑥) = \frac{47}{14} = 3.3$**

- **Let's update parameters:**

- **Using **$\ 𝜃_𝑖≔𝜃_𝑖−𝛼 \frac{1}{2𝑚} ∑_i[((ℎ_𝜃 (𝑥_𝑖 )−𝑦_𝑖 ) 𝑥_𝑖)]$**,  we get new parameter estimates for $θ_0$ and $θ_1$**
    - **$𝜃_0≔5−0.01 × \frac{1}{14} (0 + 1 + 4 + 4 + 4 + 25 + 9) = 4.989$**
    - **$𝜃_1≔2−0.01 × \frac{1}{14} ((0×1)+(1×2)+(2×2)+(2×3)+(2×4)+(5×5)+(3×5)) = 1.992$**
### **Evaluation Metrics**
- **Mean Squared Error:**
    - **MSE is calculated by taking the average of the squared differences between the predicted values and the actual values.**
    - **$\ 𝐽(𝑥) = \frac{1}{𝑚} ∑_i[(ℎ_𝜃 (𝑥_𝑖)−𝑦_𝑖)]^2 $**
    - **It penalizes large errors more heavily than small errors because of the squaring operation.**
    - **When using Mean Squared Error (MSE) as the evaluation metric, larger mistakes have a disproportionately greater impact on the overall error compared to smaller mistakes. This is because MSE calculates the average of the squared differences between the predicted values and the actual values.**
    - **Suppose you have three predictions for house prices:**
        - Prediction 1: Actual price = $200, Predicted price = $250, Error = $50
        - Prediction 2: Actual price = $300, Predicted price = $400, Error = $100
        - Prediction 3: Actual price = $400, Predicted price = $450, Error = $50
    
    **If we calculate the MSE, the errors are squared before averaging:**
    - **$\ 𝐽(𝑥) = \frac{50^2 + 100^2 + 50^2}{3} = \frac{2500 + 10000 + 2500}{3} = \frac{15000}{3} = 5000$**

    **Notice that the error for Prediction 2, which is larger ($100), contributes more to the MSE compared to the errors for Prediction 1 and Prediction 3, which are smaller ($50 each).**

    **In fact, Prediction 2's error is squared, making it have a weight five times greater than the other two errors combined. This is why MSE is more sensitive to larger mistakes.**

- **Mean Absolute Error**
    - **MAE is calculated by taking the average of the absolute differences between the predicted values and the actual values.**
    - **It treats all errors equally regardless of their magnitude.**
    - **$\ 𝐽(𝑥) = \frac{1}{𝑚} ∑_i|ℎ_𝜃 (𝑥_𝑖)−𝑦_𝑖|$**
    - **MAE is less sensitive to outliers and provides a more intuitive measure of average error, making it suitable for applications where all errors are considered equally important.**
