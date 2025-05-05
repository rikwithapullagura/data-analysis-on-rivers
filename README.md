import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
# Independent and dependent variables
X = filtered_data['Full_reservoir_level'].values.reshape(-1, 1)
y = filtered_data['Storage'].values
# Linear Regression Model
model = LinearRegression()
model.fit(X, y)
# Predictions
y_pred = model.predict(X)
# Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x=filtered_data['Full_reservoir_level'], y=filtered_data['Storage'], alpha=0.6, label="Actual Data")
plt.plot(filtered_data['Full_reservoir_level'], y_pred, color='red', label="Regression Line")
plt.title("Linear Regression: Full Reservoir Level vs Storage")
plt.xlabel("Full Reservoir Level")
plt.ylabel("Storage")
plt.legend()
plt.show()
# Coefficients
model.coef_, model.intercept_
 
Linear Regression Results:
●	Model Equation: Storage=−4.018×10−5⋅Full Reservoir Level+0.443\text{Storage} = -4.018 \times 10^{-5} \cdot \text{Full Reservoir Level} + 0.443Storage=−4.018×10−5⋅Full Reservoir Level+0.443
o	Slope (coefficient): −4.018×10−5-4.018 \times 10^{-5}−4.018×10−5
o	Intercept: 0.4430.4430.443
o	The slope indicates a very slight negative relationship between full reservoir level and storage, though the impact appears minimal
Calculate residuals
residuals = y_test - y_pred

# Plot residuals
plt.figure(figsize=(8, 6))
sns.residplot(x=y_pred, y=residuals, color='purple')
plt.axhline(0, linestyle='--', color='black')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.show()
 
print("Model Coefficient:", model.coef_)
print("Model Intercept:", model.intercept_)
 

