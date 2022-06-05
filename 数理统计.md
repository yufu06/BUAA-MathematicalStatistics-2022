# 数理统计

## 基础知识

### 基本概念

#### 总体、个体、样本

##### 分布函数

如果总体$X$的分布函数为$F(x)$，则简单随机样本$X1, X2,\cdots,X_n$的联合分布可以由总体$X$的分布函数$F(x)$来确定，样本的联合分布函数为$F(x1, x2, \cdots, x_n) = \prod\limits_{i = 1}^nF(x_i)$，其中$F(x_i)$是$X_i(i = 1, 2, \cdots, n)$的分布函数。

##### 参数空间和总体分布族

为了方便，我们将总体参数所属的范围称为参数空间，记为$\Theta$，相应的总体分布范围$\begin{Bmatrix}P_\theta:\theta\in\Theta\end{Bmatrix}$称为`总体分布族`，也称为`参数分布族`。

#### 统计量和充分统计量

##### 统计量

设$X_1, X_2, \cdots, X_n$为来自总体$X$的简单样本，若样本的函数$T(X_1, X_2, \cdots, X_n)$中不包含任何未知参数，则称此函数为`统计量`。

##### 样本 k 阶原点矩

统计量$A_k = \frac{1}{n}\sum\limits_{i = 1}^nX_i^k(k = 1, 2, \cdots)$称为`样本k阶原点矩`。

##### 样本 k 阶中心矩

统计量$B_k = \frac{1}{n}\sum\limits_{i = 1}^n(X_i - \overline{X})^k(k = 1, 2, \cdots)$称为`样本k阶中心矩`。

由大数定律知，若总体的`r`阶矩$E(x^r)(r\ge1)$存在，则样本的直到`r`阶的各阶矩依概率收敛于总体的相应阶矩。

##### 顺序统计量

把样本$X_1, X_2, \cdots, X_n$的观察值$x_1, x_2, \cdots, x_n$按从小到大递增的顺序进行排序，记为$x_{(1)}, x_{(2)}, \cdots, x_{(n)}$，它满足$x_{(1)}\le x_{(2)}\le\cdots\le x_{(n)}$。若样本$X_1, X_2, \cdots, X_n$的观察值为$x_1, x_2, \cdots, x_n$，定义排在`k`位置的数$x_{(k)}(1\le k\le n)$为随机变量$X_{(k)}$的观察值，称$X_{(1)}, X_{(2)}, \cdots, X_{(n)}$为顺序统计量。

##### 样本极差

样本极差定义为$R = X_{(n)} - X_{(1)}$，它可以描述数据分散程度的常用统计量，但没有$S^2$和$\hat{\sigma}^2$精细准确。

##### 样本中位数

样本中位数的定义为$$m_{0.5} = \begin{cases} X(\frac{n + 1}{2}) & \text{n是奇数}\\ \frac{1}{2}(X_{\frac{n}{2}} + X_{\frac{n}{2} + 1})& \text{n是偶数} \end{cases}$$，它表示把样本观察值从小到大进行排序时处在中间的数，反应了样本的平均取值。在有些情况下，它比样本均值更具有代表性。

##### 样本分位数

样本分位数的定义为$$m_{p} = \begin{cases} X([np + 1]) & \text{np不是整数}\\ \frac{1}{2}(X_{np} + X_{np + 1})& \text{np是整数} \end{cases}$$。

##### 充分统计量

设总体分布族为$\begin{Bmatrix}P_\theta:\theta\in\Theta\end{Bmatrix}$，$X_1, X_2, \cdots, X_n$是来自总体的简单样本，样本的函数$T(X_1, X_2, \cdots, X_n)$是统计量。如果在给定$T(X_1, X_2, \cdots, X_n) = t$的条件下，样本$X_1, X_2, \cdots, X_n$的条件分布函数$F_\theta(x_1, x_2, \cdots, x_n | t)$与参数$\theta$无关，则称统计量$T(X_1, X_2, \cdots, X_n)$为参数$\theta$的充分统计量。

##### 因子分解定理

设总体分布族为$\begin{Bmatrix}P_\theta:\theta\in\Theta\end{Bmatrix}$，则统计量$T(x)$是充分的，当且仅当存在一个定义在$I\times\Theta$上的实值函数$g(t, \theta)$及定义在样本空间$M$上的不依赖于参数$\theta$的实值函数$h(x)$使得样本$x_1, x_2, \cdots, x_n$的联合分布$p(x; \theta)$的分解式$p(x; \theta) = g(T(x), \theta)h(x)$对所有的$x\in M$都成立，其中$I$是统计量$T(x)$的值域，$p(x, \theta)$是样本的联合分布列或密度函数。

#### 经验分布函数

##### 经验分布函数

设样本$x_1, x_2, \cdots, x_n$的顺序统计量为$x_{(1)}, x_{(2)}, \cdots, x_{(n)}$，对任意实数$x(x\in R)$定义函数$$F_n(x) = \frac{v_n(x)}{n} = \begin{cases} 0 & x<x_{(1)}\\ \frac{k}{n} & x_{(k)} \le x < x_{(k + 1)}, k = 1, 2, \cdots, n - 1 \\1 & x\ge x_{(n)}\end{cases}$$，称$F_n(x)$为总体$X$的经验分布函数。相应地，称总体的分布函数$F(x)$为理论分布函数。

##### Glivenko 定理

当$n\to\infty$时，经验分布函数$F_n(x)$以概率$1$关于$x$一致收敛于总体的分布函数$F(x)$，即$P\begin{Bmatrix}\lim\limits_{n\to\infty}\sup\limits_{-\infty < x < \infty}|F_n(x) - F(x)| = 0\end{Bmatrix} = 1$

### 抽样分布

#### 特征函数

设$X$是随机变量，称函数$e^{itX}$的数学期望$\varphi_X(t) = E(e^{itX})$为$X$的特征函数，其中$i = \sqrt{-1}$，$t\in(-\infty, \infty)$。

#### $ \chi^2$ 分布、$t$ 分布及$F$ 分布

##### $\chi^2$ 分布

设$X_1, X_2, \cdots, X_n$是相互独立的随机变量，且$X_i\sim N(0, 1)(i = 1, 2, \cdots, n)$，则称随机变量$\chi^2 = X_1^2 + X_2^2 + \cdots + X_n^2$所服从的分布是自由度为$n$的$\chi^2$分布，记为$\chi^2\sim\chi^2(n)$。

##### t 分布

设$X\sim N(0, 1), Y\sim\chi^2(n)$，则$X$和$Y$相互独立，则称随机变量$T = \frac{X}{\sqrt{\frac{Y}{n}}}$所服从的分布是自由度为$n$的$t$分布，记为$T\sim t(n)$。

##### F 分布

设$X\sim \chi^2(n_1), Y\sim\chi^2(n_2)$，则$X$与$Y$相互独立，则称随机变量$F = \frac{\frac{X}{n_1}}{\frac{Y}{n_2}}$所服从的分布是自由度为$(n_1, n_2)$的$F$的分布，记为$F\sim F(n_1, n_2)$。

## 参数估计

### 参数的点估计

#### 频率替换法

考虑$n$次独立重复试验，每次试验有$m$种可能的结果$D_1, D_2, \cdots, D_m$，每个结果$D_i$发生的概率$P\{D_i\} = p_i$是未知的，且$\sum\limits_{i = 1}^mp_i = 1$。用$n_i$表示$n$次独立重复试验中结果$D_i$发生的次数，则$(n_1, n_2, \cdots, n_m)$的精确分布称为多项分布，其分布列为$p(n_1, n_2, \cdots, n_m) = \frac{n!}{n_1!n_2!\cdots n_m!}p_1^{n_1}p_2^{n_2}\cdots p_m^{n_m}$，其中$\sum\limits_{i = 1}^nn_i = n$。概率$p_i$最简单的直观估计是$\frac{n_i}{n}$，即$\hat{p_i} = \frac{n_i}{n}$，称$\hat{p_i}$为$p_i(i = 1, 2, \cdots, m)$的频率替换估计。

#### 矩估计法

矩估计法的主要思想是基于替换原理，将待估计的参数$q(\theta)$表示成总体各阶矩的函数，用样本矩替换相应的总体矩，获得矩估计。

#### 极大似然估计法

若在参数空间$\Theta$中存在$\hat{\theta}(x_1, x_2, \cdots, x_n)$，使得下式$L(x_1, x_2, \cdots, x_n; \hat{\theta}(x_1, x_2, \cdots, x_n)) = \sup\limits_{\theta\in\Theta}\{L(x_1, x_2, \cdots, x_n; \theta\}$成立，则称$\hat{\theta}(x_1, x_2, \cdots, x_n)$为参数$\theta$的极大似然估计。

### 估计量的评优准则

#### 均方误差准则

$MSE_\theta(T(x)) = E_\theta[T(x)] - q(\theta)^2$称为均方误差，简记为$MSE$。若$MSE_\theta(T(x))<\infty$，则有$MSE_\theta(T(x)) = Var_\theta(T(x)) + b^2(q(\theta), T)$，其中$b(q(\theta), T) = E_\theta[T(x) - q(\theta)]$，称$b(q(\theta), T)$为用$T(x)$估计$q(\theta)$所产生的偏差。

#### 无偏估计

设统计量$T(x)$是参数$q(\theta)$的一个估计，若对所有的$\theta\in\Theta$，有$E_\theta(T(x)) = q(\theta)$成立，即偏差$b(q(\theta), T(x)) = 0$，则称$T(x)$为$q(\theta)$的无偏估计。

#### 一致最小方差无偏估计

若存在无偏估计$T^*(x)\in U_q$，使得对任何估计$T(x)\in U_q$，不等式$Var_\theta(T^*(x))\le Var_\theta(T(x))$对所有的$\theta\in\Theta$都成立，则称$T^*(x)$为参数$q(\theta)$的一致最小方差无偏估计，简称$UMVUE$。

设$x_1, x_2, \cdots, x_n$是来自总体$\{P_\theta: \theta\in\Theta\}$的简单样本，总体的密度函数（或分布列）为$p(x; \theta)$，且样本$x_1, x_2, \cdots, x_n$的联合密度函数（或联合分布列）可分解为$p(x_1, x_2, \cdots, x_n; \theta) = c(\theta)h(x_1, x_2, \cdots, x_n)e^{\sum\limits_{k = 1}^mw_k(\theta)T_k(x_1, x_2, \cdots, x_n)}$，其中$h(x_1, x_2, \cdots, x_n)$仅是$x_1, x_2, \cdots, x_n$的函数，$w = w(\theta) = (w_1(\theta), \cdots, w_m(\theta))$是定义在$m$维参数空间$\Theta$的向量函数，$c(\theta)$仅是$\theta$的函数。如果$w(\theta)$的值域包含内点，则$m$维统计量$T(x_1, x_2, \cdots, x_n) = (T_1(x_1, x_2, \cdots, x_n), T_2(x_1, x_2, \cdots, x_n), \cdots, T_m(x_1, x_2, \cdots, x_n))$是完全充分的。

### 信息不等式

#### Cramer-Rao 正则族

设总体分布族为$\{p(x; \theta): \theta\in\Theta\}$，其中$p(x; \theta)$为密度函数，$\Theta$是直线上的某一开区间。若分布族$\{p(x; \theta): \theta\in\Theta\}$满足以下条件：

1. 支撑$A_\theta = \{x: p(x; \theta) > 0\}$与参数$\theta$无关，且对任一固定的$x\in A_\theta$，在参数空间$\Theta$上偏导数$\frac{\partial p(x; \theta)}{\partial\theta}$存在；

2. 如果对一切$\theta\in\Theta$，$T(x_1, x_2, \cdots, x_n)$是满足$E_\theta|T| < +\infty$的任一统计量，则有$\frac{\partial}{\partial\theta}\int_{-\infty}^{+\infty}\cdots\int_{-\infty}^{+\infty}T(x_1, x_2, \cdots, x_n)p(x_1, x_2, \cdots, x_n; \theta)dx_1d_2\cdots d_n $

   $= \int_{-\infty}^{+\infty}\cdots\int_{-\infty}^{+\infty}T(x_1, x_2, \cdots, x_n)\frac{\partial}{\partial\theta}p(x_1, x_2, \cdots, x_n; \theta)dx_1d_2\cdots d_n $

其中$p(x_1, x_2, \cdots, x_n; \theta)$为来自总体$\{p(x; \theta):\theta\in\Theta\}$的简单样本$x_1, x_2, \cdots, x_n$的联合密度函数。则称分布族$\{p(x; \theta):\theta\in\Theta\}$为`Cramer-Rao`正则族。

#### Fisher 信息量

当`Cramer-Rao`正则族定义中的条件1成立时，可定义`Fisher`信息量为$I(\theta) = E_\theta[\frac{\partial}{\partial\theta}\ln p(x; \theta)]^2$。

如果$\frac{d^2}{d\theta^2}\int_{-\infty}^{+\infty}p(x; \theta)dx=\int_{-\infty}^{+\infty}\frac{\partial^2p(x; \theta)}{\partial\theta^2}$成立，则$E_\theta[\frac{\partial}{\partial\theta}\ln p(x; \theta)]^2 = -E_\theta[\frac{\partial^2}{\partial\theta^2}\ln p(x; \theta)]$

#### 信息不等式

设总体的密度函数族$\{p(x; \theta): \theta\in\Theta\}$是`Carmer-Rao`正则族，且$0 < I(\theta) < +\infty$，$T(x_1, x_2, \cdots, x_n)$是对一切$\theta\in\Theta$满足$Var_\theta(T(x_1, x_2, \cdots, x_n)) < \infty$的统计量，令$\varphi(\theta) = E_\theta(T(x_1, x_2, \cdots, x_n))$，则对一切$\theta\in\Theta$，$\varphi(\theta)$是可微的，且$Var_\theta(T(x_1, x_2, \cdots, x_n))\ge\frac{[\varphi'(\theta)]^2}{nI(\theta)}$。

#### 有效估计

设分布族$\{P_\theta: \theta\in\Theta\}$是`Cramer-Rao`正则族，$q(\theta)$是可估参数，若存在某个无偏估计$\hat{q}\in U_q$对所有的$\theta\in\Theta$有$Var_\theta(\hat{q}) = \frac{[q'(\theta)]^2}{nI(\theta)}$，则称$\hat{q}$为参数$q(\theta)$的有效估计。

### 相合估计

设$\hat{q_n} = \hat{q_n}(x_1, x_2, \cdots, x_n)$是参数$q(\theta)$的任一估计序列，如果$\{\hat{q_n}\}$依概率收敛于参数真值$q(\theta)$，即对任意的$\epsilon>0$，有$\lim\limits_{n\to\infty}P_\theta\{|\hat{q_n} - q(\theta)|\ge\epsilon\} = 0$对任意的$\theta\in\Theta$成立，则称$\hat{q_n}$是$q(\theta)$的相合估计。

如果$\hat{q_n}$是$q(\theta)$的相合估计，且函数$g(y)$在$y = q(\theta)$处连续，则$g(\hat{q_n})$是$g(q(\theta))$的相合估计。

### 区间估计

设总体的分布族$\{P_\theta: \theta\in\Theta\}$，其中$\theta$是一维参数，若存在两个统计量$T_1(x)$及$T_2(x)$，对给定的$\alpha(0 < \alpha < 1)$有$P_\theta\{T_1(x_1, x_2, \cdots, x_n\le\theta\le T_2(x_1, x_2, \cdots, x_n)\}\ge 1 - \alpha$对所有的$\theta\in\Theta$都成立，则称随机区间$[T_1, T_2]$为参数$\theta$的置信水平为$1 - \alpha$的置信区间，称$T_1$为置信下限，称$T_2$为置信上限，称$1 - \alpha$为置信水平或置信度。

#### 枢轴变量法

1. 从参数$\theta$的一个具有优良性的点估计$\hat{\theta}$出发，构造一个仅包含统计量$\hat{\theta}$和参数$\theta$，而不含其他未知量的函数$g(\hat{\theta}, \theta)$，使得$g(\hat{\theta}, \theta)$的分布是完全已知的，且与参数$\theta$无关，称函数$g(\hat{\theta}, \theta)$为枢轴变量。
2. 对给定的置信水平$1 - \alpha(0 < \alpha < 1)$，选取两个常数$a$和$b(a < b)$，使得$P_\theta\{a\le g(\hat{\theta}, \theta\le b\}\ge 1 - \alpha,\forall\theta\in\Theta$。当$g(\hat{\theta}, \theta)$的分布为连续型时，可选取$a$和$b$使得$P_\theta\{a\le g(\hat{\theta}, \theta\le b\} = 1 - \alpha$。
3. 若不等式$a\le g(\hat{\theta}, \theta)\le b$可等价地变换为$\hat{\theta_1}(x_1, x_2, \cdots, x_n)\le\theta\le\theta_2(x_1, x_2, \cdots, x_n)$的形式，则$[\hat{\theta_1}, \hat{\theta_2}]$就是$\theta$的一个置信水平为$1 - \alpha$的置信区间。

设总体的分布族$\{P_\theta: \theta\in\Theta\}$，其中$\theta$是一维参数，若存在两个统计量$T_1(x)$，对给定的$\alpha(0 < \alpha < 1)$有$P_\theta\{\theta\ge T_1(x_1, x_2, \cdots, x_n)\}\ge 1 - \alpha$对所有的$\theta\in\Theta$都成立，则$T_1$为参数$\theta$的置信水平为$1 - \alpha$的置信下限。若存在两个统计量$T_2(x)$，对给定的$\alpha(0 < \alpha < 1)$有$P_\theta\{\theta\le T_1(x_1, x_2, \cdots, x_n)\}\ge 1 - \alpha$对所有的$\theta\in\Theta$都成立，则$T_1$为参数$\theta$的置信水平为$1 - \alpha$的置信上限。

## 假设检验

### 基本概念

设总体$X$的参数分布族为$\{p(x; \theta): \theta\in\Theta\}$，关于总体分布中的参数$\theta$的推测$\theta\in\overline{\Theta}\subset\Theta$称为假设，记为$H: \theta\in\overline{\Theta}$，其中$\overline{\Theta}$是参数空间$\Theta$的非空子集。如果$\overline{\Theta}$仅包含一个参数，即$\overline{\Theta} = \{\theta_0\}$，则称$H$为简单假设，否则称为复合假设。

一个检验等同于样本空间两个互不相交的子集$W$和$W^c$，当$(x_1, x_2, \cdots, x_n)\in W$时就拒绝$H_0$，认为备择假设$H_1$成立，而当$(x_1, x_2, \cdots, x_n)\in W^c$时就接受$H_0$，认为$H_0$成立。称$W$为拒绝域，$W^c$为接受域。

当原假设$H_0$本来成立时，样本观察值却落入拒绝域$W$，我们错误地拒绝了$H_0$，这种错误通常称为第一类错误，其概率为$\alpha(\theta) = P_\theta\{x\in W\}, \theta\in\Theta$。

当原假设$H_0$本来不成立时，样本观察值却落入接受域$W^c$，我们错误地接受了$H_0$，这种错误通常称为第二类错误，其概率为$\beta(\theta) = P_\theta\{x\notin W\}, \theta\in\Theta$

称$H_0$不成立时拒绝$H_0$的概率，即$\gamma(\theta) = P_\theta\{x\in W\} = 1 - \beta(\theta), \theta\in\Theta_1$为一个检验的势或功效。而一个检验犯第一类错误的概率和势可以看成是函数$g(\theta) = P_\theta\{x\in W\} = E_\theta(\varphi(x)),  \theta\in\Theta$的不同取值，这个函数称为势函数或功效函数。

当$\theta\in\Theta_0$时，$g(\theta) = \alpha(\theta)$；当$\theta\in\Theta_1$时，$g(\theta) = \gamma(\theta)$。

对给定的$\alpha\in(0, 1)$，若检验函数$\varphi(x)$对所有的参数$\theta\in\Theta_0$，满足$E_\theta(\varphi(x))\le\alpha$，则称$\varphi(x)$是一个显著性水平为$\alpha$的检验函数，简称水平为$\alpha$的检验。

### 正态总体参数的假设检验

#### 单个正态总体方差已知时总体均值的检验

设$x_1, x_2, \cdots, x_n$是来自正态总体$N(\mu, \sigma^2)$的简单样本，其中$\sigma^2$已知，考虑假设检验问题$H_0: \mu = \mu_0, H_1: \mu \neq \mu_0$。设其检验统计量为$z = \frac{\overline{x} - \mu_0}{\frac{\sigma}{\sqrt{n}}}$，当原假设$H_0$成立时，有$z\sim N(0, 1)$，显著性水平$\alpha$下的拒绝域为$W = \{(x_1, x_2, \cdots, x_n): |z|\ge z_{1-\frac{\alpha}{2}}\}$。

#### 单个正态总体方差未知时总体均值的检验

设其检验统计量为$z = \frac{\overline{x} - \mu_0}{\frac{S}{\sqrt{n}}}$，当原假设$H_0$成立时，有$z\sim t(n - 1)$，显著性水平$\alpha$下的拒绝域为$W = \{(x_1, x_2, \cdots, x_n): |z|\ge t_{1-\frac{\alpha}{2}}(n - 1)\}$。

#### 单个正态总体方差的检验

考虑假设检验问题$H_0: \sigma^2 = \sigma_0^2, H_1: \sigma^2\neq\sigma_0^2$，设其检验统计量为$\chi^2 = \frac{(n - 1)S^2}{\sigma_0^2}$，显著性水平$\alpha$下的拒绝域为$W = \{(x_1, x_2, \cdots, x_n): \chi^2\le\chi_{\frac{\alpha}{2}}^2(n - 1)\}\cup\{(x_1, x_2, \cdots, x_n): \chi^2\ge\chi_{1 - \frac{\alpha}{2}}^2(n - 1)\}$。

#### 两个正态总体均值相等的检验

##### 方差$\sigma_1^2, \sigma_2^2$已知

设其假设检验量为$z = \frac{\overline{x} - \overline{y}}{\sqrt{\frac{\sigma_1^2}{2} + \frac{\sigma_2^2}{2}}}\sim N(0, 1)$，显著性水平$\alpha$下的拒绝域为$W = \{|z| \ge z_{1 - \frac{\alpha}{2}}\}$。

##### 方差$\sigma_1^2, \sigma_2^2$未知，$\sigma_1^2 = \sigma_2^2 = \sigma^2$

设$S_W^2 = \frac{(n_1 - 1)S_1^2 + (n_2 - 1)S_2^2}{n_1 + n_2 - 2}$，设其假设检验量为$t = \frac{\overline{x} - \overline{y}}{S_W\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}\sim t(n_1 + n_2 - 2)$，显著性水平$\alpha$下的拒绝域为$W = \{|t| \ge t_{1-\frac{\alpha}{2}}(n_1 + n_2 - 2)\}$。

##### 方差$\sigma_1^2, \sigma_2^2$未知，$n_1 = n_2 = n$

令$z_i = x_i - y_i$，可将问题转换为$H_0: \mu = 0, H_1: \mu\neq 0$，设其检验统计量为$t = \frac{\overline{z}}{\frac{S}{\sqrt{n}}}$，显著性水平$\alpha$下的拒绝为$W = \{|t|\ge t_{1 - \frac{\alpha}{2}}(n - 1)\}$。

##### 方差$\sigma_1^2, \sigma_2^2$未知，$\sigma_1^2 \neq \sigma_2^2$，$n_1 \neq n_2$

不妨假设$n_1 < n_2$，令$z_i = x_i - \sqrt{\frac{n_1}{n_2}}y_i + \frac{1}{\sqrt{n_1n_2}}\sum\limits_{j = 1}^{n_1}y_j - \frac{1}{n_2}\sum_{j = 1}^{n_2}y_j, i = 1, 2, \cdots, n_1$，可将问题转换为$H_0: \mu = 0, H_1: \mu\neq 0$，设其检验统计量为$t = \frac{\overline{z}}{\frac{S}{\sqrt{n_1}}}$，显著性水平$\alpha$下拒绝域为$W = \{|t|\ge t_{1 - \frac{\alpha}{2}}(n_1 - 1)\}$。

#### 两个正态总体方差相等的检验

设总体$X\sim N(\mu_1, \sigma_1^2), Y\sim N(\mu_2, \sigma_2^2)$，$x_1, x_2, \cdots, x_n$和$y_1, y_2, \cdots, y_n$是分别来自总体$X$和$Y$的简单样本，且两样本相互独立。考虑假设检验问题$H_0:\sigma_1^2 = \sigma_2^2, H_1: \sigma_1^2\neq\sigma_2^2$。设其检验统计量为$F = \frac{S_1^2}{S_2^2}$，显著性水平$\alpha$下拒绝域为$W = \{F\le F_{\frac{\alpha}{2}}(n_1 - 1, n_2 - 1)\}\cup\{F\ge F_{1 - \frac{\alpha}{2}}(n_1 - 1, n_2 - 1)\}$。

### Pearson 检验法

设总体$X$的分布函数为$F(x)$，$F_0(x)$为完全指定的分布函数，不包含任何未知参数，$x_1, x_2, \cdots, x_n$是来自总体$X$的简单样本，考虑检验假设$H_0: F(x) = F_0(x)$。

若样本容量$n$充分大，则无论总体服从何种分布$F_0(x)$，$\chi^2 = \sum\limits_{i = 1}^k\frac{(f_i - np_i)^2}{np_i}$中的统计量$\chi^2$总是近似地服从自由度为$k - 1$的$\chi^2$分布，其中$F_0(x)$完全确定，不含任何未知参数。

根据以上描述，检验假设$H_0$的$\chi^2$拟合检验法如下：

1. 把实轴$(-\infty, +\infty)$分成$k$个互不相交的区间$A_i = (a_i, a_{i + 1}], i = 1, 2, \cdots, k$，其中$a_1$和$a_{k+1}$可分别取$-\infty$和$+\infty$。区间的划分方法应视具体情况而定。
2. 计算概率$p_i = P\{X\in A_i\} = F_0(a_{i + 1}) - F_0(a_i), i = 1, 2, \cdots, k$，并计算$np_i$，称为理论频数。
3. 计算样本观察值$x_1, x_2, \cdots, x_n$落在区间$A_i = (a_i, a_{i + 1}]$中的个数$f_i(i = 1, 2, \cdots, k)$，称为实际频数。
4. 由$\chi^2 = \sum\limits_{i = 1}^k\frac{(f_i - np_i)^2}{np_i}$计算$\chi^2$的值。
5. 对于给定的显著性水平$\alpha$，查表得临界值$\chi_{1 - \alpha}(k - 1)$。
6. 做出推断，拒绝域为$W = \{(x_1, x_2, \cdots, x_n): \chi^2\ge\chi_{1 - \alpha}^2(k - 1)\}$。

### 似然比检验

考虑似然比$\lambda(x) = \frac{p(x_1, x_2, \cdots, x_n; \theta_1)}{p(x_1, x_2, \cdots, x_n; \theta_0)}$，其中$x$表示样本点$(x_1, x_2, \cdots, x_n)$，称$\lambda(x)$为似然比统计量。

对于一半的假设检验问题，$H_0: \theta\in\Theta_0, H_1:\theta\in\Theta_1$，类似地可以定义似然比检验统计量$\lambda(x) = \frac{\sup\limits_{\theta\in\Theta_1}\{p(x_1, x_2, \cdots, x_n; \theta)\}}{\sup\limits_{\theta\in\Theta_0}\{p(x_1, x_2, \cdots, x_n; \theta)\}}$。

### 检验的优良性

#### Neyman-Pearson 引理

对假设检验问题，如果存在水平为$\alpha$的检验函数$\varphi^*\in\Phi_\alpha$，使得对任一水平为$\alpha$的检验函数$\varphi\in\Phi_\alpha$，都有不等式$E_{\theta_1}(\varphi^*(x)) \ge E_{\theta_1}(\varphi(x))$成立，则称检验函数$\varphi^*$为检验假设问题的水平为$\alpha$的最优势检验，简称$MPT$。

给定水平$\alpha(0 < \alpha < 1)$，对假设检验问题有：

1. 存在常数$k\ge0$及检验函数$ \varphi(x) =\left\{\begin{aligned}1, \quad \lambda(x) > k\\0, \quad \lambda(x) < k\end{aligned}\right.$ 满足$E_{\theta_0}(\varphi(x)) = \alpha$，即$\varphi\in\Phi_\alpha$，且检验函数$\varphi(x)$是水平为$\alpha$的最优势检验，其中 $\lambda(x) = \frac{p(x; \theta_1)}{p(x; \theta_0)}$ 是似然比统计量。
2. 如果检验函数 $\varphi(x)$ 是水平为$\alpha$的最优势检验，则必存在常数$k\ge0$，使得检验函数$\varphi(x)$满足$ \varphi(x) =\left\{\begin{aligned}1, \quad \lambda(x) > k\\0, \quad \lambda(x) < k\end{aligned}\right.$。进一步，如果$\varphi(x)$的势$E_{\theta_1}(\varphi(x)) < 1$，则$\varphi(x)$也满足$E_{\theta_0}(\varphi(x)) = \alpha$。

#### 一致最优势检验

设总体分布族为$\{p(x; \theta): \theta\in\Theta\}$，考虑假设检验问题$H_0: \theta\in\Theta_0, H_1: \theta\in\Theta$，将水平为$\alpha$的所有检验函数的集合记为$\Phi_\alpha = \{\varphi(x): \sup\limits_{\theta\in\Theta_0}\{\varphi(x)\le\alpha\}\}$定义最优势检验如下：

对于上述假设检验问题，若存在水平为$\alpha$的检验函数$\varphi^*\in\Phi_\alpha$，使得对任一水平为$\alpha$的检验函数$\varphi\in\Phi_\alpha$有不等式，$E_\theta(\varphi^*(x))\ge E_\theta(\varphi(x))$对所有的$\theta\in\Theta$都成立，则称$\varphi^*(x)$是水平为$\alpha$的一致最优势检验，简记为$UMPT$。

如果样本$x_1, x_2, \cdots, x_n$的联合密度函数（或分布列）$p(x; \theta)(\theta\in\Theta)$是单参数的并可以表示为$p(x; \theta) = d(\theta)h(x)e^{c(\theta)T(x)}$，其中$\theta$是实值函数，且$c(\theta)$关于$\theta$是严格单调递增函数，则对单侧检验问题$H_0: \theta\le\theta_0, H_1: \theta>\theta_0$：

1. 水平为$\alpha$的一致最优势检验存在，其检验函数为$ \varphi^*(x) =\left\{\begin{aligned}1, \quad T(x) > c\\r, \quad T(x) = c\\0, \quad T(x) < c\end{aligned}\right.$其中常数$c$和$r\in[0, 1]$由$E_{\theta_0}(\varphi*(x)) = \alpha$确定。
2. 水平为$\alpha$的一致最优势检验$\varphi^*(x)$的势函数$E_\theta(\varphi^*(x))$是$\theta$的单调增函数。

## 回归分析

### 一元线性回归

#### 未知参数的估计及其统计性质

对于线性回归方程$\hat{y} = \hat{a} + \hat{b} x$，使偏差平方和达到最小值的$\hat{a}$和$\hat{b}$是$a$，$b$的无偏估计，且是$a$，$b$的一致最小方差线性无偏估计，其中$\hat{a} = \overline{y} - \hat{b}\overline{x}$，$\hat{b} = \frac{L_{xy}}{L_{xx}} = \frac{\sum\limits_{i = 1}^n(x_i - \overline{x})(y_i - \overline{y})}{\sum\limits_{i = 1}^n(x_i - \overline{x})^2} = \frac{\sum\limits_{i = 1}^n(x_i - \overline{x})y_i}{{\sum\limits_{i = 1}^n(x_i - \overline{x})x_i}}$ 。

#### 预测

设一元线性回归模型$y_i = a + bx_i + \epsilon_i, i = 1, 2, \cdots, n$，其中$\epsilon_i\sim N(0, \sigma^2)$，且$\epsilon_0, \epsilon_1, \cdots, \epsilon_n$相互独立，则$t = \frac{y_0 - \hat{y_0}}{\hat{\sigma}\sqrt{1 + \frac{1}{n} + \frac{(x_0 - \overline{x})^2}{L_{xx}}}}\sim t(n - 2)$，其中$\hat{\sigma} = \sqrt{\frac{Q}{n - 2}}$，可以证明$Q = \sum\limits_{i = 1}^n(y_i - \hat{a} - \hat{b}x_i)^2 = \sum\limits_{i = 1}^ny_i^2 - \hat{a}\sum\limits_{i = 1}^ny_i - \hat{b\sum\limits_{i = 1}^n}x_iy_i$。

$y$的置信水平为$1 - \alpha$的预测区间为$[\hat{y}(x) - \var(x_0), \hat{y}(x) + \var(x_0)]$，其中$\var(x) = \hat{\sigma}t_{1 - \frac{\alpha}{2}}(n - 2)\sqrt{1 + \frac{1}{n} + \frac{(x - \overline{x})^2}{L_{xx}}}\approx\hat{\sigma}t_{1 - \frac{\alpha}{n}}$。

#### 控制

对于给定的$y_1^* < y_2^*$，若要使观察值$y$至少以概率$1 - \alpha$落在某个区间$(y_1^*, y_2^*)$内，当$\hat{b} > 0$时，$x$的控制区间为$(x_1^*, x_2^*)$；而当$\hat{b} < 0$时，$x$的控制区间为$(x_2^*, x_1^*)$，其中$x_1^* = \frac{1}{b}(y_1^* - \hat{a} + \hat{\sigma}z_{1 - \frac{\alpha}{2}}), x_2^* = \frac{1}{b}(y_2^* - \hat{a} - \hat{\sigma}z_{1 - \frac{\alpha}{2}})$。

## 方差分析与正交试验设计

### 单因素试验方差分析

一般地，设因素$A$有$p$个不同水平$A_1, A_2, \cdots, A_p$，且在每个水平$A_i$下，总体$X_i$服从同方差的正态分布$N(\mu_i, \sigma^2), i = 1, 2, \cdots, p$，其中$\mu_i$和$\sigma^2$均为未知参数。为确定数据差异主要是由随机误差引起的还是由$A$的水平变化引起的，需要定义一个指标来度量数据差异程度的大小，通常用数据离差平方和作为度量指标，即$S_T = \sum\limits_{i = 1}^p\sum\limits_{j = 1}^{n_i}(x_{ij} - \overline{x})^2$，其中$\overline{x} = \frac{1}{n}\sum\limits_{i = 1}^p\sum\limits_{j = 1}^{n_i}x_{ij}, n = \sum\limits_{i = 1}^pn_i$。

### 双因素试验方差分析

类似定义总离差平方和$S_T = \sum\limits_{i = 1}^p\sum\limits_{j = 1}^q(x_{ij} - \overline{x})^2$，其中$\overline{x} = \frac{1}{pq}\sum\limits_{i = 1}^p\sum\limits_{j = 1}^qx_{ij}$

