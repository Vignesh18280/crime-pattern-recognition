import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

# Create a dummy scaler for demonstration
# In a real scenario, this would be the actual scaler trained with your model
dummy_scaler = StandardScaler()
dummy_data = np.random.rand(10, 32) # 10 samples, 32 features
dummy_scaler.fit(dummy_data)

joblib.dump(dummy_scaler, 'scaler.pkl')
print("Created a dummy scaler.pkl file.")