### Models

- **Bayesian Models**
  - **_Black-Litterman Model_**:The Black-Litterman (BL) model is one of the many successfully used portfolio allocation models out there. Developed by Fischer Black and Robert Litterman at Goldman Sachs, it combines Capital Asset Pricing Theory (CAPM) with Bayesian statistics and Markowitz’s modern portfolio theory (Mean-Variance Optimisation) to produce efficient estimates of the portfolio weights.
  - **_Entropy Pooling_**:The Entropy Pooling (EP) algorithm developed by Atillio Meucci is a methodology that allows investors to specify non-linear views on the market and generate a posterior distribution that aligns with these views. Unlike the Black-Litterman model which only allows for views on returns, EP can be used for specifying views not only on returns but on different other market factors like correlations, covariances, rankings etc…
  - **_Robust Bayesian Allocation_**: The Robust Bayesian Allocation (RBA) algorithm, first developed by Atillio Meucci, makes assumptions about the prior market parameters, calculates the posterior market distribution and generates robust portfolios along the Bayesian Efficient Frontier. Depending on the prior chosen by the investor and the observed market data, the final posterior distribution can be modeled efficiently in sync with the true market parameters.
- **Clustering Models**
  - **_Hierarchical Risk Parity (HRP)_**: One of the more recently developed optimisation algorithms, the Hierarchical Risk Parity approach uses unsupervised machine learning (hierarchical tree clustering) to segregate assets into clusters based on risk. It is considered as a significant improvement over classic mean-variance based techniques and has been shown to generate diversified portfolios with robust out-of-sample properties without the need for a positive-definite return covariance matrix.
  - **_Hierarchical Equal Risk Contribution (HERC)_**:Developed by Thomas Raffinot in 2018, this algorithm which takes inspiration from the Hierarchical Risk Parity (HRP) by utilising unsupervised tree clustering to cluster assets in a portfolio. It is a powerful algorithm that can produce robust portfolios and avoids many of the problems seen with Modern Portfolio Theory and HRP.
  - **_Nested Clustered Optimization (NCO)_**:The algorithm estimates optimal weight allocation to either maximize the Sharpe ratio or minimize the variance of a portfolio. As evident by the name, it clusters the covariance matrix of asset returns to a reduced, denoised form and produce efficient weight allocations.
- **Risk and Return Estimators**
  - **_Critical Line Algorithm (CLA)_**: This is a famous portfolio optimisation algorithm that overcomes some limitations of Markowitz’s Mean-Variance Optimisation approach. The most important trait of this method is that it provides the investor with an option to place lower and upper bounds on the weights of the assets.
  - **_Mean-Variance Optimisation_**: A collection of classic Mean-Variance portfolios – Inverse Variance, Minimum Volatility, Quadratic Utility, Maximum Sharpe, Efficient Risk etc… It also provides users with ability to create custom portfolios with custom objectives and constraints.
  - **_Entropy Pooling_**:The Entropy Pooling (EP) algorithm developed by Atillio Meucci is a methodology that allows investors to specify non-linear views on the market and generate a posterior distribution that aligns with these views. Unlike the Black-Litterman model which only allows for views on returns, EP can be used for specifying views not only on returns but on different other market factors like correlations, covariances, rankings etc…
- **Statistical**
  - **_SSD_**:
- **Online Portfolio Selection**:
Online Portfolio Selection is an algorithmic trading strategy that sequentially allocates capital among a group of assets to maximize the final returns of the investment. Traditional theories for portfolio selection, such as Markowitz’s Modern Portfolio Theory, optimize the balance between the portfolio’s risks and returns. However, OLPS is founded on the capital growth theory, which solely focuses on maximizing the returns of the current portfolio.
  - Benchmarks
  - Momentum
  - Mean Reversion
  - Pattern Matching

### Filters