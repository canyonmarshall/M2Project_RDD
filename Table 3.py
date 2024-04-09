import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load the data into a DataFrame named 'dwi'
dwi = pd.read_csv('hansen_dwi.csv')

# Assuming 'dui' column is missing, adjust the required columns list accordingly
required_columns = ['bac1', 'recidivism', 'year', 'male', 'white', 'aged']  # 'dui' removed
missing_columns = [col for col in required_columns if col not in dwi.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Define the function to calculate rectangular kernel weights
def calculate_rectangular_weights(df, center, bandwidth):
    return np.where((df['bac1'] >= center - bandwidth) & (df['bac1'] <= center + bandwidth), 1, 0)

# Apply the function to calculate weights for 'dwi'
dwi['wt_3A'] = calculate_rectangular_weights(dwi, center=0.08, bandwidth=0.05)

# Filter the DataFrame to only include positive weights
dwi_filtered = dwi[dwi['wt_3A'] > 0]

# Ensure 'dwi_filtered' still contains all required columns after filtering
missing_columns_filtered = [col for col in required_columns if col not in dwi_filtered.columns]
if missing_columns_filtered:
    raise ValueError(f"Missing required columns after filtering: {missing_columns_filtered}")

# Adjust the formula to remove reference to the missing 'dui' column
formula = 'recidivism ~ bac1 + year + male + white + aged'

# Fit the model using WLS (Weighted Least Squares) with the calculated weights
model = smf.wls(formula, data=dwi_filtered, weights=dwi_filtered['wt_3A']).fit()

# Print the summary of the model
print(model.summary())


# Assuming 'dwi' DataFrame is already loaded with pd.read_stata or similar
# and contains the columns used in your model.

# Define function to calculate rectangular kernel weights
def calculate_rectangular_weights(df, center, bw):
    return np.where((df['bac1'] >= center - bw) & (df['bac1'] <= center + bw), 1, 0)

# Calculate weights and add them to the DataFrame
dwi['wt_3B'] = calculate_rectangular_weights(dwi, center=0.08, bw=0.025)

# Filter the DataFrame to only include positive weights
dwi_filtered = dwi[dwi['wt_3B'] > 0]

# Define the model using Patsy formula (similar to the R formula)
# Assuming 'recidivism', 'dui', 'bac1', 'year', 'male', 'white', and 'aged' are column names in your DataFrame.
formula = 'recidivism ~ dui + bac1 + dui * bac1 + year + male + white + aged'

# Fit the model using WLS (Weighted Least Squares) due to the presence of weights
model_3b1 = smf.wls(formula, data=dwi_filtered, weights=dwi_filtered['wt_3B']).fit()

# Print the summary of the model
print(model_3b1.summary())
