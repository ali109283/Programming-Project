# Proposal for Advanced Programming Project #

## Modeling and Forecasting the Luxury Goods Sector Using Statistical and Machine Learning Approaches ##

### Category ###

Financial Data Analysis and Predictive Modeling

### Problem Statement & Motivation ###

Financial markets are complex systems influenced by a wide range of factors, including macroeconomic trends, investor sentiment, and company performance. In particular, the luxury goods industry, represented by global brands such as LVMH, Hermès, Kering, Richemont, Burberry, Prada, and Moncler, is heavily affected by shifts in consumer demand, brand perception, and global market confidence.

This project aims to analyze and forecast stock price behavior within this luxury sector by constructing a panel dataset of these companies. Rather than focusing on a single firm, a multi-company panel allows for richer comparisons, the identification of common patterns, and improved predictive modeling. As a finance student, my motivation is to design my first analytical tool that can extract insights useful for investment analysis, portfolio diversification, and risk management.

### Planned Approach and Technologies ###

The project will begin with the collection of historical stock data for the selected luxury companies, primarily using Yahoo Finance’s API or other financial data providers. Where if necessary, web scraping techniques may be used to supplement missing data.

Data cleaning and preprocessing will be performed in Python using libraries such as pandas, NumPy, and matplotlib for transformation and visualization. The cleaned panel dataset will then be used to perform regression analyses to uncover statistical relationships between firms and over time.

For forecasting, I plan to implement Neural Networks, specifically LSTM (Long Short-Term Memory) architectures, to capture time-dependent and non-linear patterns. Additionally, volatility estimation will be conducted using both historical and implied approaches, providing a more complete picture of risk dynamics across the sector. Backtesting procedures will validate the predictive models by comparing forecasted versus actual price outcomes.

### Expected Challenges and Mitigation Strategies ###

A key challenge will be ensuring data quality and consistency across multiple firms, different exchanges, and multiple currencies. This will be mitigated through robust data-cleaning routines, careful handling of missing values, and cross-validation. Another challenge may involve scaling and maintaining a large dataset that updates dynamically.

### Success Criteria ###

Success will be measured by the accuracy and robustness of the forecasting models, evaluated using metrics such as RMSE (Root Mean Squared Error) as well as the project’s ability to uncover meaningful insights into inter-company relationships, volatility trends, and sector-wide dynamics within the luxury goods industry. Lastly, the posibility of this tool be used for identifying potential investment opportunities in the luxury segment.


### Stretch Goals ###

A very feasible goal will be to include investment recommendations, of "Which Luxury Brand you should Buy" based on the current market situation. 

Let me know what you think of this new proposal and if aligns better with the project idea, complexity and suitability

