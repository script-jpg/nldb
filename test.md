<!---


This is markdown file template.

- item 1
- item 2
    - item 2.1


1. list 1
2. list 2
3. list 3
   1. list 3.1


math symbols:

$\mu, \sigma, \varphi, \delta, \varnothing$


    This is a code block.


Display an image.

![image](C:\courses\4A\AODA.png)



<!---
This is a comment.
--->

<h1>
STAT 333

Stochastic Process 1
</h1>

#### Instructor: Pengfei Li



### Probability Functions
- events
- outcomes






### Examples
> A coin is continually and independently tossed, where the probability of head $(H)$ on a toss is $1/2$.
> - $P(1^{st} \text{ two tosses give } HH) = (1/2)^2 = 1/4$
> - $P(1^{st} \text{ two tosses give } TH) = (1/2)^2 = 1/4$
> - $P(TH \text{ occurs before } HH) = P(\text{case 1}) + P(\text{case 2}) = 1/2 + 1/4 = 3/4$
>   - case 1: $1^{st}$ toss gets $T$, $TH$ occurs before $HH$ ($1/2$)
>   - case 2: $1^{st}$ toss gets $H$, $2^{nd}$ one gets $T$, then $TH$ occur before $HH$ ($1/4$)
>   - case 3: $1^{st}$ toss gets $H$, $2^{nd}$ one gets $H$, then $HH$ occur before $TH$ ($1/4$)


### Conditional Probability

Suppose $E \& F$ are two events with $P(F) > 0$, then 
$$
P(E|F) \text{ [conditional probability]} = \frac{P(E\cap F)\text{ [joint probability]}}{P(F) \text{ [marginal probability]}}
$$

- $P(E \cap F) = P(E | F) \cdot P(F)$
- if $E$ and $F$ are independent, then $P(E|F) = P(E)$


#### Bayes' Formula

Suppose there is a sequence of events $F_1, F_2, ..., F_n$ such that 

- $F(F_i) > 0$
- $F_i \cap F_j = \varnothing$ for all $i \not = j$
- $\cup_{i} F_i = S$ (sample space) 

then
1. $P(E) = \sum_{i} P(E\cap F_i) = \sum_i P(E| F_i) P(F_i)$
2. $P(F_k | E) = \frac{P(F_k \cap E)}{P(E)} = \frac{P(E|F_k) P(F_k)}{\sum_i P(E|F_i) P(F_i)}$ [Bayes' Formula]


> There are 3 doors $(A, B, C)$ befind which there are two goats and a car.
> Monty knows the location of the car, but you do not.\
> You select a door at random (say $A$) and at this point the chance of winning the car is $1/3$.\
> Then Monty opens one of the remaining two doors, either door $B$ or $C$, to reveal a goat.\
> $P(\text{winning the car if switching the door}) = 2/3$ since
> - Method 1:
>   - P($A \to$ goat) = $2/3$
>   - P($A \to$ car) = $1/3$ 
> - Method 2 (conditional probability):
>   - Suppose you choose door $A$ and Monty opens door $B$ [event $E$]
>   - let $F_k$ = car is behind door $k$, $k = A, B, C$, then
>   - $P(\text{win if switch}) = P(F_C|E) = \frac{P(E | F_C) P(F_C)}{P(E|F_A) P(F_A) + P(E|F_B)P(F_B) + P(E|F_C)P(F_C)} = 2/3$
>       - $P(E|F_A) = 1/2$, $P(E|F_B) = 0$, $P(E|F_C) = 1$, $P(F_A) = P(F_B) = P(F_C) = 1/3$


### Random Variables
- a function defined on a sample space to real line, $X: S \to R \subseteq \mathbb{R}$
  - Discrete RV: all possible values are at most countable
  - Continuous RV: all possible values [uniform contain an interval]

### Bernoulli Trials
1. each tial has 2 outcomes
2. all trials are independent
3. probability of 's' on each trial is the same. 
   - $P('s') = p$, $P('F') = q = 1-p$

#### `Bernoulli distribution`: $Bernoulli(P)$:
- $I_i = 1$ if $'s'$ appears in $i^{th}$ trial ($= 0$ otherwise) 
    - $P(I_i = 1) = p, P(I_i = 0) = q$
    - $I_1, I_2$ ... are a sequence of independent identically distributed Bernoulli random variables
#### `Binomial Distribution` $X \sim Bin(n, p)$: number of $'s'$ in $n$ Bernoulli trials
  - Range: $\{0, 1, ..., n\}$
  - probability: $P(X = k) = {n \choose k} p^k (1-p)^{n-k}$ for $k = \{0, 1, ..., n\}$
  - $X = \sum_{i=1}^n I_i$
  - if $X_1 \sim Bin(n_1, p)$, $X_2 \sim Bin(n_2, p)$ and they are independent, then $X_1 + X_2 = Bin(n_1+n_2, p)$

#### `Geometric Distribution` $Geo(p)$:
- $X = \{\text{Number of trails to get the first `s'}\}$
  - Range: $\{1,2,3,...\}$
  - pmf: $P(X = k) = (1-p)^{k-1}p$, $k = 1, 2, 3$
  - `No memory property`
    - $P(X > n+m | x > m) = P(x > n)$
    - Given that we don't observe $'s'$, the remaining time and original time $\sim Geo(p)$.

> A fair coin is tossed repeatedly and independently. The objective is to observe the $1^{st}$ head. Let $X$ be the corresponding waiting time $(X \sim Geo(p), p = 1/2)$.\
> Suppose we got 6 tails in the first tosses.\
> Then
> - $P(X = 10 | \text{the first 6 tosses give 6 tails}) = P(10-6) = P(4) = (1/2)^{4} = \frac{1}{16}$
> - $E(X | \text{the first 6 tosses give 6 tails}) = E(6 + \text{Remaining Time}) = 6 + \frac{1}{1/2} = 8$

#### Negative Binomial Random Variable $NegBin(r, p)$:
- $X = \# \text{ of trials to get } r\ 's' \text{ in total} \sim NegBin(r, p)$ 
  - Range: $\{r, r+1, ...\}$
  - pmf: ????
  - property
    - $x_1 =$ waiting time to observe the first 's',\
     $x_2 =$ waiting time to observe the second 's' (after the observation of first 's')\
     $x_r =$ waiting time to observe the $r^{th}$ 's' (after the observation of $(r-1)^{th}$ 's')
    - $X = \sum_{i=1}^{r} x_i$


> A fair coin is tossed repeatedly and independently. Then objective is to observe the two heads in total. Let $X$ be the corresponding waiting time.
>   - $X \sim NegBin(2, 1/2)$
>   - $E(X | \text{the first 3 toss give `HTT'}) = E(3+\text{Remaining time to observe the second `H' after the first `H'}) = 3 + \frac{1}{1/2} = 5$
>     - Remaining time to observe the second 'H' after the first 'H' $\sim Geo(1/2)$
>   - $E(X | \text{the first 3 toss give `TTT'}) = E(3 + 2 + 2) = 7$

#### Poission Random Variable $Pois(\lambda)$:
- Range: $\{0, 1, 2, ...\}$
- pmf: $P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$, k = 0, 1, 2, ...
  - $\lambda:$ rate parameter; $E(X) = \lambda$
- Property:
  - $X_1 \sim Pois(\lambda_1), X_2 \sim Pois(\lambda_2)$, $X_1, X_2$ are independent\
    then $X_1+X_2 \sim Pois(\lambda_1+\lambda_2)$


### Continuous Random Variables
#### Exponential Random Variable $Exp(\lambda)$:
- Probility density function: 
$$
f(x) = \begin{cases} \lambda e^{-\lambda x} & x > 0\\ 0 & otherwise \end{cases}
$$

- $\lambda$: rate parameter; $E(X) = \frac{1}{\lambda}$
- tail probability: $P(x > t) = e^{-\lambda t}$
- no memory property: $P(X-s > t|X > s) = P(X > t+s|X > s) = P(X > t)$
  - Given that we don't observe the event, remaining time and original time $\sim Exp(\lambda)$

> Suppose ther waiting time $(X)$ for customers coming to $T.H.$ follows $Exp(2)$. In the first 3 minutes, there is no customer. Then
> - $P(X>5 | X > 3) = P(X > 5-3) = P(2) = e^{-4}$
> - $E(X | \text{no customer in the first 3 minutes}) = E(3 + \text{Remaining time}) = 3 + \frac{1}{\lambda} = 3.5$

## 1.3 Expectation and Variance
#### Expectation
$$E(X) = \begin{cases}\sum_{i=0}^{\infty} x_i P(x = x_i) & X\text{ is a discrete random variable with range} (x_0, x_1, ...) \\ \int_{-\infty}^{\infty} xf(x) \ dx\ & X \text{ is a continuous RV}\end{cases}$$

- $X \sim Bernoulli(p)$, $E(X) = 1 P(X = 1) + 0 P(X = 0) = p$

Let $g(x)$ be a function of $x$, then 
$$E(g(x)) = \begin{cases}\sum_{i=0}^{\infty} g(x_i) P(X = x_i) & \text{discrete rv} \\ \int_{-\infty}^{\infty} g(x) f(x) \ dx & \text{continuous rv}\end{cases}$$

- $X \sim Bernoulli(p)$, then $E(X^2) = 1^2 P(X=1) + 0^2 P(X = 0) = p$

#### Variance

$$Var(X) = E(X^2) - E(X)^2$$

- $X \sim Bernoulli(p)$, then $Var(X) = p - p^2 = p(1-p) = P(X = 1)P(X = 0)$


#### Properties

1. $E(\sum_{i=1}^n a_iX_i) = \sum_{i=1}^n a_iE(X_i)$ (linearity)
2. If $X_1, X_2,...,X_n$ are independent, then 
   - $Var(\sum_{i=1}^n a_iX_i) = \sum_{i=1}^{n} Var(a_iX_i) = \sum_{i=1}^n a_i^2 Var(X_i)$
3. $Var(aX+b) = Var(aX) = a^2Var(X)$
4. $Var(\sum_{i=1}^n a_i X_i) = \sum_{i=1}^n a_{i}^2 Var(X_i) + \sum_{i\not = j} a_i a_j Cov(X_i, X_j) = \sum_{i=1}^n a_{i}^2 Var(X_i) + 2\sum_{i \le j} a_i a_j Cov(X_i, X_j)$
    - $Cov(X, Y) = E(XY) - E(X)E(Y)$
    - $X$ and $Y$ are independent $\Rightarrow Cov(X, Y) = 0$

> If $X, Y$ are independent, $Var(X-Y) = Var(X) + Var(Y)$\
> $Var(X_1 + X_2 + X_3) = Var(X_1) + Var(X_2) + Var(X_3) + 2 Cov(X_1, X_2) + 2 Cov(X_1, X_3) + 2 Cov(X_2, X_3)$


### 1.4 Indicator Random Variables

$I_A = \begin{cases} 1 & \text{if A occurs} \\ 0 & \text{otherwise} \end{cases}$

Suppose $P(A) = p$, then $P(I_A=1) = p$ and $P(I_A = 0) = 1-p$

- $E(I_A) = p = P(I_A = 1)$
- $Var(I_A) = p(1-p) = P(I_A = 0)P(I_A = 1)$

> Suppose $X \sim Bin(n, p)$
>   - $E(X) = \sum_{i=1}^n E(I_i)$, where $I_i \sim^{iid} Bernoulli(p) \\\hspace{1cm}= np$
>   - $Var(X) = Var(\sum_{i=1}^nI_i) = \sum_{i=1}^nVar(I_i) = np(1-p)$

> Red box has 4 red balls and 6 black balls, and Black box has 6 red balls and 4 black balls.\
> Toss a coin, $H \to$ red box and $T \to$ black box\
> $R \to$ red box, $B \to$ black box\
> $X = \#$ of R balls in first 2 steps. 
> - Let $I_i = 1 \iff$ getting $R$ balls in the $i^{th}$ step.
>   - $P(R_1) = P(I_1 = 1) = P(R_1|H) P(H) + P(R_1|T)P(T) = 0.4*0.5 + 0.6*0.5 = 0.5$
>   - $P(R_2) = P(I_2 = 1) = P(R_2|R_1) P(R_1) + P(R_2|B_1)P(B_1) = 0.4*0.5 + 0.6*0.5 = 0.5$
>   - $Cov(I_1, I_2) = E(I_1I_2) - E(I_1) E(I_2) = 0.2 - 0.25 = -0.05$
>     - $E(I_1, I_2) = P(R_1, R_2) = P(R_2|R_1) P(R_1) = 0.2$
>   - $Var(X) = Var(I_1) Var(I_2) - 2Cov(I_1, I_2) = 0.5^2 *2 - 0.05*2$
>     
> 
>   
> - $X = I_1 + I_2$
> - $E(X) = 0.5+0.5 = 1$


### Waiting Time Random Variables

Suppose there is a sequence of trial and the goal is to observe an event $E$ based on the sequence of trials

- $T_E =$ number of trials / waiting time to observe $E = \min(n | E \text{ occurs on } n^{th} \text{ trials})$
- $Range(T_E) = \{1,2,...\} \cup \{\infty\}$
  - $T_E < \infty$: can observe the event
  - if $T_E < \infty$, $E(T_E) < \infty$
- Classification of $T_E$:
  - if $P(T_E < \infty) < 1$ (or $P(T_E = \infty) > 0$), $T_E$ is improper.
    - $E(T_E) = \infty$ since $P(T_E = \infty) > 0$
  - if $P(T_E < \infty) = 1$ (or $P(T_E = \infty) = 0$), $T_E$ is proper.
    - if $E(T_E < \infty) = \infty$, $T_E$ is null proper
    - if $E(T_E < \infty) < \infty$, $T_E$ is short proper
  1. if $T_E$ is improper, $E(T_E) = \infty$
  2. if $E(T_E) < \infty$, $T_E$ is short proper
  
#### $\sum_{n=1}^\infty a_n$ 
  - doesn't not include $\infty$ in the summation
  - $\sum_{n=1}^{\infty} a_n = lim_{m \to \infty} \sum_{n=1}^ma_n$
  - $P(T_E < \infty) = \sum_{n=1}^\infty P(T_E = n)$


> Toss a coin repeatedly and independently with $P(H_i)$ = p. Let $T_H$ be the waiting time for the first $H$.
> - $T_H$ is a short proper waiting time random variable since
>   - $P(T_H = n) = (1-p)^{n-1}p, n \ge 1$
>   - $P(T_H < \infty) = \sum_{i=1}^{\infty} (1-p)^{n-1}p = \frac{p}{1-(1-p)} = 1$
>   - $E(T_H) = \frac{1}{p} < \infty$, $T_H \sim Geo(p)$

> Toss a coin repeatedly and independently, with probability getting $H$ at $i^{th}$ toss $= \frac{1}{i+1}$.
> Let $T_H$ be the waiting time for getting $H$.
>
> - $T_H$ is a null proper waiting time random variable sicne
>   - $P(T_H = n) = P(T...TH) = \frac{1}{n+1}\prod_{i=1}^{n-1} (1-\frac{1}{i+1}) = \frac{1}{n(n+1)} = \frac{1}{n} - \frac{1}{n+1}$
>   - $P(T_H < \infty) = \sum_{k=1}^\infty P(T_H = k) = 1 - \frac{1}{2} + \frac{1}{2} - ... = 1$
>   - $E(T_H) = \sum_{k=1}^\infty \frac{1}{n+1} = \infty$


> Toss a coin repeatedly and independently, with probability getting $H$ at $i^{th}$ toss $= 2^{-n}$.
> Let $T_H$ be the waiting time for getting $H$.
>
> - $T_H$ is a improper waiting time random variable sicne
>   - $P(T_H = n) = P(T...TH) = 2^{-n}\prod_{i=1}^{n-1} (1-2^{-i}) = \frac{1}{2^n}\prod_{i=1}^{n-1} \frac{2^i-1}{2^i} = \frac{1}{2^n} \frac{1}{2} C, (C < 1) \\\hspace{1.8cm}< \frac{1}{2^{n+1}}$
>   - $P(T_H < \infty) = \sum_{k=1}^\infty P(T_H = k) < \sum_{k=1}^\infty \frac{1}{2^{k+1}} < 1$

### Conditional Expectation
#### Joint Random Variables $X \& Y$

- Discrete: 
  - both $X$ and $Y$ are discrete
  - pmf:
    - $f_{X,Y}(x,y) = P(X=x\& Y=y)$
      - $f_{X,Y}(x,y) \ge 0$
      - $\sum_{x}\sum_{y} f_{X,Y}(x,y) = 1$
      - $f_X(x) = P(X = x) = \sum_{x} f_{X,Y}(x,y)$
      - $f_Y(y) = P(Y = y) = \sum_{x} f_{X,Y}(x,y)$
      - $E(h(X,Y)) = \sum_{x} \sum_{y} h(x,y) f_{X,Y}(x,y)$
        - $E(XY) = \sum_{x} \sum_{y} xy f_{X,Y}(x,y)$
        - $E(X) = \sum_{x}\sum_{y} xf_{X,Y}(x,y) = \sum_{x} xf_X(x)$


- Continuous:
  - $X$ and $Y$ are continuous and $P(X \le x, Y \le y) = \int_{-\infty}^x [\int_{-\infty}^y f_{X,Y}(s,t)\ dt]\ ds$
  - $f_{X,Y}(x,y) \ge 0$
  - $\int_{-\infty}^{\infty} [\int_{-\infty}^\infty f_{X,Y}(x,y)\ dy]\ dx = 1$
  - marginal pdf of $X$: $f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x, y) dy$
  - marginal pdf of $Y$: $f_Y(y) = \int_{-\infty}^{\infty} f_{X,Y}(x, y) dx$
  - Expectation:
    - $E(h(X,Y)) = \int_{-\infty}^{\infty} [\int_{-\infty}^\infty h(x,y) f_{X,Y}(x,y)\ dy]\ dx$
    - $E(XY) = \int_{-\infty}^{\infty} [\int_{-\infty}^\infty xy f_{X,Y}(x,y)\ dy]\ dx$
    - $E(X) = \int_{-\infty}^{\infty} [\int_{-\infty}^\infty x f_{X,Y}(x,y)\ dy]\ dx = \int_{-\infty}^\infty x f_{X}(x)\ dx$

- Independence $\iff f_{X,Y}(x, y) = f_X(x) f_Y(y)$
- $X$, $Y$ are independent $\Rightarrow g(X), h(Y)$ are independent.
  - $E(g(X)h(Y)) = E(g(X))E(h(Y))$
  - $Cov(g(X), h(Y)) = 0$
    - $Cov(g(X), h(Y)) = 0$ doesn't imply independence

> $X \sim Unif[-1,1]$, $Y = X^2$,\
> $f_X(x) = \begin{cases}\frac{1}{2}& -1 < x < 1 \\ 0 &\text{otherwise}\end{cases}$\
> $Cov (X,Y) = 0$ but $X, Y$ are not independent.
> 


#### Conditional pmf

##### Discrete Case
$f_{X|Y}(x|y) = \frac{f_{X,Y}(x,y)}{f_Y(y)} = \frac{\text{joint}}{\text{marginal}}$

- Properties:
  1. $f_{X|Y}(x,y) \ge 0$
  2. $\sum_{x}f_{X|Y}(x|y) = 1$
      - $\sum_{x}f_{X|Y}(x|y) = \frac{\sum_{x}f_{X,Y}(x,y)}{f_Y(y)} = \frac{f_Y(y)}{f_Y(y)} = 1$
  
- Expectation
  - $E(X|Y=y) = \sum_{x} xf_{X|Y}(x|y)$
  - $E(g(X) | Y=y) = \sum_x g(x) f_{X|Y}(x|y)$
- Independence Properties
   - if $X$ and $Y$ are independent, then
     - $E(X|Y=y) = E(X)$
     - $E(g(X)|Y=y) = E(g(x))$
     - $f_{X|Y}(x,y) = \frac{f_{X,Y}(x,y)}{f_Y(y)} = \frac{f_X(x) f_Y(y)}{f_Y(y)} = f_{X}(x)$


> suppose $X_1 \sim Pois(\lambda_1), X_2 \sim Pois(\lambda_2)$, and $X_1, X_2$ are independent. \
> Let $X = X_1, Y = X_1 + X_2$
> - $f_{X|Y}(x|y) = \frac{f_{X,Y}(x,y)}{f_Y(y)} = {y\choose x} (\frac{\lambda_1}{\lambda_1+\lambda_2})^x (\frac{\lambda_2}{\lambda_1 + \lambda_2})^{y-x}$
>   - $f_{X,Y}(x,y) = P(X_1 = x, X_1+X_2 = y)\\
      \hspace{1.6cm} = P(X_1 = x, X_2 = y-x)\\
      \hspace{1.6cm} = P(X_1=x)P(X_1=y-x) \\
      \hspace{1.6cm} = \frac{\lambda_1^xe^{-\lambda_1}}{x!} \frac{\lambda_2^{y-x}e^{-\lambda_2}}{(y-x)!}$
>   - $f_Y(y) = \frac{(\lambda_1 + \lambda_2)^ye^{-\lambda_1-\lambda_2}}{y!}$
>   - $X|Y=y \sim Bin(y, \frac{\lambda_1}{\lambda_1+\lambda_2})$

##### Continuous Case

$$f_{X|Y}(x|y) = \frac{f_{X,Y}(x,y)}{f_Y(y)}$$

- properties:
  - $f_{X|Y}(x|y) \ge 0$
  - $\int_{-\infty}^\infty f_{X|Y}(x|y)\ dx = 1$

- Expectation:
  - $E(X|Y=y) = \int_{-\infty}^\infty x f_{X|Y}(x|y)\ dx$
  - $E(g(X)|Y=y) = \int_{-\infty}^\infty g(x) f_{X|Y}(x|y)\ dx$

- Independence Properties:
  - if $X,Y$ are independent
    - $E(X| Y=y) = E(X)$
    - $E(g(X)|Y=y) = E(g(X))$

> Given $f_{X,Y}(x,y) = \begin{cases}xe^{-xy} & x > 0, y > 1 \\ 0 & \text{otherwise} \end{cases}$
> - $f_{X|Y}(x|y) = \frac{f_{X,Y}(x,y)}{f_{Y}(y)}  = xe^{-xy}y^2$
>    - $f_{Y}(y) = \int_{0}^\infty f_{X,Y}(x,y)\ dx\\
          \hspace{1cm} = \int_{0}^\infty xe^{-xy}\ dx\\
          \hspace{1cm} = \int_{0}^\infty te^{-t}/y^2\ dt\\
          \hspace{1cm} = \frac{1}{y^2} \Gamma(2)\\
          \hspace{1cm} = \frac{1}{y^2}\\$


- General properties
  1. $E(\sum_{i=1}^n a_iX_i|Y=y) = \sum_{i=1}^n a_iE(X_i|Y=y)$
  2. Sunstitution rule: 
      - $E(X\cdot g(Y)|Y=y) = E(X\cdot g(y)|Y=y) = g(y)E(X|Y=y)$
      - $E(h(X,Y)|Y=y) = E(h(X,y)|Y=y)$
  3. Independence Properties (Assuming $X,Y$ are independent):
      - $E(X|Y=y) = E(X)$
      - $E(g(X)|Y=y) = E(g(X))$


### Calaulating Expectation by Conditioning

- $E(X) = E(E(X|Y)) = E(g(Y)) = \begin{cases}\sum_{y} E(X|Y=y) F_Y(y)& Y\text{ is discrete}\\ \int_{-\infty}^\infty E(X|Y=y) F_Y(y) \ dy & Y\text{ is continuous} \end{cases}$
  - $E(X|Y) = g(Y)$ is a random variable depends on $Y$.
  - $g(y) = E(X|Y=y)$
  - $E(X) = \sum_{x}\sum_{y}x f_{X,Y}(x,y) = \sum_{y}\sum_{x}x f_{X|Y}(x,y) f_Y(y) = \sum_yE(X|Y=y)f_Y(y)$


> Let $Y_1, Y_2, ...$ be independently distributed random variables such that for $n \ge 1$, $Y_n \sim Pois(n)$.
>
> Let $N \sim Geo(0.5)$, and it is independent to $Y_i$.\
> Let $X = Y_N$, then
> - $E(X) = E(E(X|N)) = E(g(N)) = E(N) = 2$
>   - $g(n) = E(X|N = n) = E(Y_N| N = n) = E(Y_n|N=n) = E(Y_n) = n$
>
> - $E(X) = \sum_{n=1}^\infty E(X|N=n) P(N=n) = \sum_{n=1}^\infty n P(N=n) = E(N) = 2$


> Tossing a coin repeatedly and independently with probability getting 'H' $= p$.\
> Let $X$ be the waiting time for the first 'H'.
>
> Let $Y = \begin{cases}1 & \text{first outcome } = H \\ 0 & \text{second outcome } = T \end{cases}$
>
> - $E(X) = E(X|Y=1)P(Y=1) + E(X|Y=0)P(Y=0) = 1p + E(1+X) (1-p) = 1 - E(x)(p-1) \Rightarrow E(X) = 1/p$


> A miner is trapped. There are 3 doors:
> - door 1 leads to safety after 2 hours
> - door 2 wastes 3 hours
> - door 3 wastes 4 hours
> 
> Suppose the miner choose the door randomly. 
> 
> Let $X$ be the length of time until the minor get out.
>
> Let $Y \in \{1,2,3\}$ be the door chosen each time.
>
> - $P(Y=1) = P(Y=2) = P(Y=3) = 1/3$
> - $R =$ remaining time
>   - Note that $E(X) = E(R)$
> - $X|Y=1 = 2$, $X|Y=2 = 3+R$, $X|Y=3 = 4+R$
> - $E(X) = E(X|Y=1) P(Y=1) + E(X|Y=2)P(Y=2) + E(X|P=3) P(Y=3)\\
\hspace{1cm} =2/3 + (3+E(X))/3 + (4+E(X))/3\\
\hspace{1cm} = 3+\frac{2}{3}E(X)\Rightarrow E(X) = 9$


### Computing Probability by Conditioning

Let $A$ be an event and $I_A = \begin{cases} 1 & \text{if } A \text{ occurs} \\ 0 & \text{otherwise} \end{cases}$

- $P(A) = E(I_A) = E(E(I_A|Y)) = \begin{cases} \sum_y E(I_A | Y=y) f_Y(y) & \text{discrete} \\ \int_{-\infty}^\infty E(I_A | Y=y) f_Y(y) \ dy & \text{continuous} \end{cases}\\
\hspace{4.6cm}= \begin{cases} \sum_y P(A|Y=y) f_Y(y) & \text{discrete} \\ \int_{-\infty}^\infty P(A|Y=y) f_Y(y) \ dy & \text{continuous} \end{cases}$

- $P(I_A=1|Y=y) = E(I_A|Y=y) = P(A|Y=y)$

> Let $X_1, X_2, X_3$ be idd random variables from uniform distribution on $[0,1]$.
> - $P(X_1 < X_2) = P(X_1 = min(X_1,X_2)) = \int_{-\infty}^\infty P(X_1 < X_2|X_2 = y) f_{X_2} (y) \ dy\\ 
\hspace{2.2cm} = \int_{0}^1 P(X_1 < y|X_2 = y) dy\\ 
\hspace{2.2cm} = \int_{0}^1 P(X_1 < y)\ dy\\
\hspace{2.2cm} = \frac{y^2}{2}|^1_0 = 1/2$
>   - $P(X_1 < y) = \int^{y}_0 f_{X_1}(x)\ dx = y$
>
> - $P(X_1 < X_2 < X_3) = \int_{-\infty}^\infty P(X_1 < X_2 < X_3|X_2 = y) f_{X_2}(y)\ dy\\
\hspace{2.9cm} = \int_{0}^1 P(X_1 < y < X_3) dy\\
\hspace{2.9cm} = \int_{0}^1 P(X_1 < y)P(y < X_3) dy\\
\hspace{2.9cm} = \int_{0}^1 y(1-y) dy\\
\hspace{2.9cm} = 1/6$


> $Y$ has pdf $f(y) = \begin{cases}ye^{-y} & y>0 \\ 0&  y\le 0 \end{cases}$
>
> $X|Y=y \sim Pois(y)$
>
> - $P(X=n) = \int_{0}^\infty P(X=n|Y=y) f_Y(y)\ dy\\
      \hspace{1.7cm} = \int_{0}^\infty \frac{y^{n}e^{-y}}{n!} ye^{-y}\ dy\\
      \hspace{1.7cm} = (n+1)/2^{n+2}$ (for $n\ge 0$)


### Calulating Variance by Conditioning

#### By Definition

- $Var(X) = E(X^2) - E(X)^2 = E(E(X^2|Y)) - E(E(X|Y))^2$
  - $E(f(X)) = E(E(f(X)|Y))$


> A miner is trapped, there are 3 doors:
>   - door 1 leads to safety after 2 hrs
>   - door 2 wastes 3 hrs
>   - door 3 wastes 4 hrs
>
> The Miner chooses a door randomly each time.
>
> Let $X$ be the waiting time.
>
> - $Y =$ door \# the miner chooses each time.
>
> - $E(X^2)\\
\hspace{1.3cm} = \sum_{y} E(X^2|Y=y) P(Y=y)\\
\hspace{1.3cm} = E(X^2|Y=1)P(Y=1)  + E(X^2|Y=2)P(Y=2) + E(X^2|Y=3)P(Y=3)\\
\hspace{1.3cm} = 2^2/3  + E((X+3)^2)/3 + E((X+4)^2)/3\\
\hspace{1.3cm} = 4/3  + E(X^2 + 6X + 9)/3 + E(X^2 + 8X + 16)/3\\
\hspace{1.3cm} = 4/3  + E(X^2)/3 + 2E(X) + 3 + E(X^2)/3 + 8E(X)/3 + 16/3\\
\hspace{1.3cm} = 2E(X^2)/3 + 155/3
\Rightarrow E(X^2) = 155$
> - $Var(X) = E(X^2) - E(X)^2 = 155-81 = 74$

#### Conditional Variance Formula

1. Given $Y=y$, the conditional variance of $X$ is
    - $Var(X|Y=y) = E(X^2|Y=y) - E(X|Y=y)^2$
2. $Var(X|Y=y) =: h(y)$ is a function of $y$
    - $X|Y=y \sim Pois(y) \Rightarrow h(y) = Var(X|Y=y) = y$
3. Apply $h(y)$ to $Y$ to get $h(Y) = Var(X|Y)$
    - $Var(X|Y) = E(X^2|Y) - E(X|Y)^2$
    - Find $Var(X|Y)$:
      1. find $h(y) = Var(X|Y=y)$
      2. get $Var(X|Y) = h(Y)$

4. Comments on $Var(X|Y=y)$:
    - substitution rule is still applicable
    - if $X\& Y$ are independent, then $Var(X|Y=y) = Var(X)$

5. Formula to calculate $Var(X)$:
    - $Var(X) = E(Var(X|Y)) + Var(E(X|Y))$
      - $Var(X) = E(X^2) + E(X)^2\\
          \hspace{1.3cm} = E(E(X^2|Y)) - E(E(X|Y))^2\\
          \hspace{1.3cm} = E(E(X^2|Y) - E(X|Y)^2) + E(E(X|Y)^2) - E(E(X|Y))^2\\
          \hspace{1.3cm} = E(Var(X|Y)) + Var(E(X|Y))$



> A coin is weight such that $P(H) = 1/4$
>
> Let $N$ be the number of tosses required to get 3 $H$s by using the weight coin
>
> Let $X$ be the number of heads in $N$ tosses of a fair coin.
>
> - $N \sim NegBin(3,1/4)$, $X|N = n = \#$ of heads in $n$ tosses $\sim Bin(n, 1/2)$.
>
> - $E(X) = E(E(X|N)) = E(N/2) = 6$
>
> - $Var(X) = E(Var(X|N)) + Var(E(X|N)) \\
    \hspace{1.2cm} = E(N/4) + Var(N/2) = 1/4 E(N) + 1/2Var(N) = 12$
>   - $Var(X|N=n) = n (1/2) (1-1/2) = n/4$


#### Compound RV Formula

- Setup:
  - $X_1, X_2, ...$: iif random variables
  - $N$: non-negative random varaible independent to $X_1, X_2, ...$
  - $W = \sum_{i=1}^N X_i$: compound RV

- Result:
  - $E(W) = E(N) * E(X_1)$
  - $Var(W) = E(N) Var(X_1) + Var(N)E(X_1)^2$

- Proof
  - $Var(W|N=n) = Var(\sum_{i=1}^N X_i|N=n)\\
  \hspace{2.4cm} =Var(\sum_{i=1}^n X_i|N=n)\\
  \hspace{2.4cm} =Var(\sum_{i=1}^n X_i)\\
  \hspace{2.4cm} =\sum_{i=1}^n Var(X_i)\\
  \hspace{2.4cm} = n Var(X_1) =: h(n)\\
  \Rightarrow Var(W|N) = h(N) = N\ Var(X_1)$
  - $Var(W) = E(Var(W|N)) + Var(E(W|N))\\
    \hspace{1.3cm}= E(N\ Var(X_1)) + Var(N\ E(X_1))\\
    \hspace{1.3cm} = E(N)Var(X_1) + Var(N) E(X_1)^2$



> A coin is weight such that $P(H) = 1/4$
>
> Let $N$ be the number of tosses required to get 3 $H$s by using the weight coin
>
> Let $X$ be the number of heads in $N$ tosses of a fair coin.
>
> Let $X_i = \begin{cases}1 &\text{if the } i^{th} \text{ toss give H} \\ 0 & \text{otherwise} \end{cases}$
>   - $X = \sum_{i=1}^N X_i$
>   - $X_i \sim Bernoulli(1/2)$
>   - $E(X) = E(N)E(X_1) = \frac{3}{1/4} \frac{1}{2} = 6$
>   - $Var(X) = E(N)Var(X_1) + Var(N)E(X_1)^2 = 12$


> $Y$ has pdf $f(y) = \begin{cases}ye^{-y} & y > 0 \\ 0 & y \le 0 \end{cases}$
>
> $X|Y = y \sim Pois(y)$
>
> - $E(X) = E(E(X|Y)) = E(Y) = \int_{0}^\infty y\cdot f(y) \ dy = \int_{0}^\infty y^2e^{-y}\cdot \ dy = \Gamma(3) = 2$ 
>   - $E(X|Y) = Y$
> - $Var(X) = E(Y) + Var(Y) = E(Y) + E(Y^2) - E(Y)^2 = \Gamma(4) + 2 - 4 = 4$


### Generating Function

Given a sequence of real numbers $S = \{a_0, a_1, ...\}$, $A(S) = \sum^{\infty}_{i=0} a_i S^i$

- $A(S)$ converges iff $|S| < R$ (or $|S| > R$) for some convergence radius $R$

> $a_n = 1$, $A(S) = \sum_{n=0}^\infty S^n = \frac{1}{1-S}$ if $|S| < 1$
> 
> $a_{n} = \frac{1}{n!}$, $A(S) = \sum_{n=0}^\infty \frac{S^n}{n!} = e^S$


- There is one-to-one correspondence between $A(S)$ and $a_n|^{\infty}_{n=0}$
  - given $a_{n}|^\infty_{n=0}$, $A(S)$ is uniquely defined
  - given $A(S)$, $a_0 = A(0)$, $a_n = A^{(n)}(0)/n!$


- Commonly used power series:
  1. Geometric: $A(S) = \sum_{n=0}^\infty S^n = \frac{1}{1-S}$
      - $R=1, |S|< R$
  2. Alternate Geometric: $A(S) = \sum_{n=0}^\infty(-1)^n S^n = \frac{1}{1+S}$
      - $R=1, <$
  3. Exponential: $A(S) = \sum_{n=0}^{\infty} \frac{S^n}{n!} = e^S$
      - $R = \infty$, $a_n = \frac{1}{n!}$
  4. Binomial: $A(S) = \sum_{k=0}^n {n \choose k} S^k = (1+S)^n$, $n > 0$
      - $R = \infty$, $a_n = {n \choose k}$
  5. General Binomial: $A(S) = \sum_{n=0}^\infty {\alpha \choose n} S^n = (1+S)^\alpha$
      - $\alpha \in \mathbb{R} \le 0$ 
      - ${\alpha \choose n} = \frac{\alpha (\alpha-1) ... (\alpha-n+1)}{n!}$


> $A(S) = (1+S)^{-1}$
>
>  - Geometric $\Rightarrow a_n = (-1)^n$
>  - Binomial $\Rightarrow a_n = {-1 \choose n} = (-1)^n$

#### Properties of gfs

Given $A(S) = \sum_{n=0}^\infty a_n S^n, R_A; B(S) = \sum_{n=0}^\infty b_n S^n, R_B$


- Sumation: 
  - $C(S) = A(S) + B(S) =  \sum_{n=0}^\infty (a_n + b_n) S^n$
  - $c_n = a_n + b_n$ 
  - $R_C = min(R_A, R_B)$
- Product:
  - $C(S) = A(S) \times B(S)$
  - $c_n = \sum_{k=0}^na_kb_{n-k}$
  - $R_C = min(R_A, R_B)$


> $C(S) = \frac{1}{(1-S)(1+S)}\\
  \hspace{0.8cm}= \frac{1}{2} (\frac{1}{1-S} + \frac{1}{1+S})\\
  \hspace{0.8cm}= \sum_{n=0}^\infty \frac{1}{2} (1+(-1)^n) S^n$

> $C(S) = \frac{1}{(1-S)^2}$
> 
> - $\hspace{0.8cm} = (\sum_{n=0}^\infty S^n)(\sum_{n=0}^\infty S^n)\\
  \hspace{0.8cm} = \sum_{n=0}^\infty (\sum_{k=0}^n 1) S^n\\
  \hspace{0.8cm} = \sum_{n=0}^\infty (n+1) S^n$
>
> - $\hspace{0.8cm} = \sum_{n=0}^\infty {-2 \choose n} (-S)^n\\
     \hspace{0.8cm} = \sum_{n=0}^\infty (-1)^n{-2 \choose n} S^n\\
     \hspace{0.8cm} = \sum_{n=0}^\infty (n-1) S^n$

### Probability Generating Function

Suppose $X$ is a non-negative integer rv with range $\{0,1,2,...\} \cup \{\infty\}$.
Let $p_n = P(X=n)$, 

Then $G_X(S) = \sum_{n=0}^\infty p_n S^n = \sum_{n=0}^\infty P(X=n) S^n$ is the pgf of $X$.

- If $X$ is proper ($P(X = \infty) = 0$), then $G_X(S) = \sum_{X=n}^\infty S^n P(X=n) = E(S^X)$

- $G_X(1) = \sum_{n=0}^\infty p_n = 1-P(X = \infty) = P(X < \infty)$
- if $|S| \le 1$, $|G_X(S)| < \infty$ converge

- Properties:
  - Given $G_X(S)$:
    - $p_n = G_X^{(n)}(0)/n!$ for $n \ge 1$
    - $p_0 = G_X(0)$

