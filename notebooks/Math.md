### Least-Squares Solutions

#### Ordinary and Weighted Least Squares Solutions 

Suppose we have a set of $n$ observations $\mathbf y$ and an unknown set of $m$ variables $ \mathbf x$ that are related by a known linear operator $E$. Mathematically, this is written as: $$ \mathbf{y = E x}$$

One way to determine $\mathbf{x}$ that are consistent with $\mathbf{y}$ and $\mathbf{E}$ is to define and minimize the cost function $J$ where 
$$J  = (\mathbf{y - E x})^T(\mathbf{y - Ex})$$
Minimizing this particular cost function with respect to $x$ is known as ordinary least squares. 

If one wishes to weight particular observations more than others, an alternative form of $J$ could be: 
$$J  = (\mathbf{y - E x})^T \mathbf{W}^{-1} (\mathbf{y - Ex})$$
where $\mathbf{W} = \mathbf{I} ~ [\frac{1}{w_1}, \frac{1}{w_2}, ..., \frac{1}{w_n}]^T$ and $\mathbf{I} $ is the identity matrix

The solution that minimizes both ordinary and least squares cost functions are given by 
$$ \mathbf{\tilde x} = (\mathbf{E}^T \mathbf{W}^{-1} \mathbf{E})^{-1} \mathbf{E}^T \mathbf{W}^{-1} \mathbf{y}$$

#### Taper Weighted Least Squares Solutions

Again, suppose we have a set of $n$ observations $\mathbf y$ and an unknown set of $m$ variables $ \mathbf x$ that are related by a known linear operator $E$. In this case, let's assume we have many less observations than variables we wish to estimate (i.e. $ n << m$). 

In linear algebra terms, this is an *underdetermined problem*, meaning that the matrix $\mathbf{E}^T \mathbf{W}^{-1} \mathbf{E}$ will be *ill-conditioned* and therefore cannot be inverted. Therefore, the solution to the ordinary and weighted least squares cost functions cannot be numerically computed. 

The issue that we are dealing with is the *underdetermined problem* has infinitely many ways to fit the data. One way to limit the number of solutions is to add constraints to $\mathbf x$. Adding constraints to $\mathbf x$ can be done in the cost function, $J$, this is known as adding regularization to our cost function. One form of regularization is known as *taper weighted least squares*. This particular cost function is defined as: 

$$ J = (\mathbf{y - E x})^T \mathbf{W}^{-1} (\mathbf{y - Ex}) + (\mathbf{x}_0 - \mathbf{x})^T \mathbf{S}^{-1} (\mathbf{x}_0 - \mathbf{x})$$

The second term is the reglarization term and consists of an additional weighting matrix $S$ and a relaxation solution $x_0$. The solution to this particular cost function is 
$$ \mathbf{\tilde x} = (\mathbf{E}^T \mathbf{W}^{-1} \mathbf{E} + \mathbf{S}^{-1})^{-1}(\mathbf{E}^T \mathbf{W}^{-1} \mathbf{y} + \mathbf{S}^{-1}\mathbf{x}_0)$$

When $\mathbf{x}_0 = 0 $ and $\mathbf{S}^{-1} = \epsilon \mathbf{I}$ where $\epsilon \approx 0$, the result is the solution to the *standard form Tikhonov regularization cost function*. 

$$ \mathbf{\tilde x}_{Tikhonov} = (\mathbf{E}^T \mathbf{W}^{-1} \mathbf{E} +  \epsilon \mathbf{I})^{-1}\mathbf{E}^T \mathbf{W}^{-1} \mathbf{y}$$
### Matrix-Vector Statistics

#### Expected Values
Suppose we have a scalar random variable $a$ that follows some distribution $\mathcal D$. The expected value of $a$ is given by 
$$<a> = \int_{-\infty}^\infty a f(a) ~ da$$
where $f$ is the probability density function of $a$. 

Now, suppose we have vector of random variables $\mathbf{a} = [a_1, a_2, a_3, ...]^T$ where each $a_i$ follows a distribution $D_i$. The expectation of $\mathbf{a}$ is 
$$<\mathbf{a}> = [<a_1>, <a_2>, <a_3>, ...]^T$$


#### Covariance Matrices
Here, we introduce the *covariance matrix*, which is given vectors $\mathbf{a}$ and $\mathbf{b}$ is defined as 
$$\mathbf{C}_{ab} = <(\mathbf{a} - <\mathbf{a}>)(\mathbf{b} - <\mathbf{b}>)^T>$$

Here, $\mathbf{C}_{ab}$ represents the average deviation of all elements of $<\mathbf{a}>$ and $<\mathbf{b}>$ with one another. 
$$\mathbf{C}_{ab} = \begin{pmatrix}
<a_1' b_1' > & <a_1' b_2'> & ...\\
<a_2' b_1' > & <a_2' b_2'> & ...\\
... & ... & <a_n' b_n'>  &  
\end{pmatrix}>$$

where $a_i' = a_i - < a_i >$ and $b_i' = b_i - < b_i >$
### Least-Squares Solution Covariances
Suppose our observations $\mathbf y$ are contaminated by some normal noise centered at zero , $\mathbf n \sim \mathcal{N}(0, \sigma)$: 
$$\mathbf y = \mathbf y_{true} + \mathbf n$$

Our previous estimate $\mathbf{\tilde x}$ was obtained in a situation which would have been equivalent to assuming $\mathbf{n} = 0$. This was equivalent to solving the maximum likelihood problem, which itself is equivalent to the least squares problem where $\mathbf{W}= \mathbf{C}_{nn}$. In this case, $\mathbf{\tilde x}$ is the also a random variable that has been contaminated by some noise. 

To determine the uncertainty associated with contamination, we can compute the solution covariance. 
$$\mathbf{C}_{\mathbf{\tilde x \tilde x}} = <(\mathbf{\tilde x} - <\mathbf{\tilde x}>)(\mathbf{\tilde x} - <\mathbf{\tilde x}>)^T>$$
Using the rules of expectation, we find that 
$$\mathbf{C}_{\mathbf{\tilde x \tilde x}} = (\mathbf{E}^T \mathbf{C}^{-1}_{nn} \mathbf{E})^{-1}\mathbf{E}^T \mathbf{C}^{-1}_{nn} \mathbf{C}_{nn} \mathbf{C}^{-1}_{nn} \mathbf{E} (\mathbf{E}^T \mathbf{C}^{-1}_{nn} \mathbf{E})^{-1}$$

#### Least-Squares Solution Uncertainty
Suppose that we want to determine the solution uncertainty, which is the expected deviation of our estimate from the true answer, accounting for all uncertainties. 
$$\mathbf{P}= < (\mathbf{\tilde{x}} - \mathbf{x}_{true}) (\mathbf{\tilde{x}} - \mathbf{x}_{true})^T > $$
Define $\mathbf{\tilde{x}}' = \mathbf{\tilde{x}} - <\mathbf{\tilde{x}}> $ and $b = <\mathbf{\tilde{x}}> - \mathbf{x}_{true}$. Now, 
$$\mathbf{P}= < (\mathbf{\tilde{x}} + \mathbf{b}) (\mathbf{\tilde{x}} + \mathbf{b})^T>$$
Using some algebra, we find that 
$$\mathbf{P} =  \mathbf{C}_{\mathbf{\tilde{x}}' \mathbf{\tilde{x}}'} + 2 \mathbf{C}_{\mathbf{\tilde{x}}' \mathbf{b}}  + \mathbf{C}_{\mathbf{b} \mathbf{b}}$$

### Gauss-Markov Estimation 
In the least squares formulation, we had a relationship $\mathbf{y = E x + n}$ and sought to minmize a cost function that minimized data model mismatches. Alternatively, we could seek solutions that minimize our resulting *uncertainties*. Hence, we define a new cost *set of functions*
$$\mathbf{J} = \mathbf{P} = <(\mathbf{\tilde x}  -\mathbf{x}_{true}) (\mathbf{\tilde x}  -\mathbf{x}_{true})^T >$$
This new cost function will produce a solution for which there is minimal variance about the true values (which themselves are random variables)?. Essentially, we account for all uncertainties simultaneously, rather than by some bulk metric. This requires solving a larger system of equations (I think) and assumes that individual data points are of interst to use.  

To proceed with typically matrix algebra, assume that the centered solution is a linear combination of centered observations
$$\mathbf{\tilde x}' = \mathbf{F} \mathbf{y}'$$
which is equivalent to 
$$\mathbf{\tilde x} - <\mathbf{\tilde x}> = \mathbf{F} (\mathbf{y} - <\mathbf{y}>)$$

Additionally, we assume the linear relationship now exists for 
 $\mathbf{y' = E x' + n}$ where $\mathbf y' = \mathbf y'_{true} + \mathbf n_y$ and $\mathbf x' = \mathbf x'_{true} + \mathbf n_x$. Additionally, we enforce that $<\mathbf{y}> = \mathbf{E}<\mathbf{x}>$, which is equivalent to assumption that $<\mathbf{y} - \mathbf{E}\mathbf{x}> = 0$
 
The Gauss-Markov Theorem produces the $\mathbf{F}$ matrix that  minimizes the diagonal elements of $\mathbf{P}$. Let's evaluate $P$ 

\begin{align}
\mathbf{P} &= <(\mathbf{\tilde x}'  -\mathbf{x}_{true}') (\mathbf{\tilde x}'  -\mathbf{x}_{true}')^T> \\
&=<\mathbf{\tilde  x'} \mathbf{\tilde  x'}^T > - <\mathbf{x}_{true} \mathbf{\tilde  x'}^T > - <\mathbf{\tilde  x'} \mathbf{x}_{true}^T > + <\mathbf{x}_{true} \mathbf{x}_{true}^T >\\
\end{align}
Assume $<\mathbf{x}_{true} \mathbf{\tilde  x'}^T > = <\mathbf{\tilde  x'}\mathbf{x}_{true}^T >$. Then, 
\begin{align}
\mathbf{P}&=\mathbf{F} \mathbf{C}_{\mathbf{yy}} \mathbf{F}^T -  \mathbf{C}_{\mathbf{x_{true} y}}\mathbf{F}^T - \mathbf{F} \mathbf{C}_{\mathbf{x_{true} y}}^T + \mathbf{C}_{\mathbf{x_{true} x_{true}}} 
\end{align}

Now, we can use a relationship known as "completing the square". This relationship says that given square matrices $\mathbf A$, $\mathbf B$ and symmetric matrix $\mathbf C$: 
\begin{align}
\mathbf{ACA^T - BA^T - AB^T} &= \mathbf{ACA^T - BA^T - AB^T} + (\mathbf{B C^{-1} B^T} - \mathbf{B C^{-1} B^T})\\
 &= \mathbf{(AC - B)A^T - (AC-B)C^{-1}B^T} - \mathbf{B C^{-1} B^T}\\
  &= \mathbf{(AC - B)(A^T - C^{-1}B^T)} - \mathbf{B C^{-1} B^T}\\
  &= \mathbf{(AC - B)(A - BC^{-1})^T} - \mathbf{B C^{-1} B^T}\\
&= \mathbf{(A - BC^{-1})C (A - BC^{-1})^T - B C^{-1} B^T}
\end{align}

Then, $\mathbf P$ is 
$$\mathbf{P} =(\mathbf{F}-  \mathbf{C}_{\mathbf{x_{true} y}} \mathbf{C}_{\mathbf{yy}}^{-1}) \mathbf{C}_{\mathbf{yy}} (\mathbf{F}-  \mathbf{C}_{\mathbf{x_{true} y}} \mathbf{C}_{\mathbf{yy}}^{-1})^T - \mathbf{C}_{\mathbf{x_{true} y}} \mathbf{C}_{\mathbf{yy}}^{-1} \mathbf{C}_{\mathbf{x_{true} y}}^T + \mathbf{C}_{\mathbf{x_{true} x_{true}}} $$
 In this case, $\mathbf P$ is minimized when $\mathbf F = \mathbf{C}_{\mathbf{x_{true} y}} \mathbf{C}_{\mathbf{yy}}^{-1}$

 