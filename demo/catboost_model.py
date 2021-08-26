import joblib
import shap

cat_features = [
        "center_id", "meal_id", "emailer_for_promotion", 
        "homepage_featured", "category", "cuisine", 
        "city_code", "region_code", "center_type"
]
num_features = ["week", "checkout_price", "base_price", "op_area"]
feature_cols = num_features + cat_features

class DemandRegressor():
	def __init__(self):
		self.model = joblib.load('model.pkl')

	def predict_demand(self, data):
		return self.model.predict(data)

	def explain(self, data):
		explainer = shap.TreeExplainer(self.model)
		shap_values = explainer.shap_values(data)
		return shap.summary_plot(shap_values, data,
	                      max_display=25, auto_size_plot=True)