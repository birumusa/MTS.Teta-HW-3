import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import joblib
import shap
import math
from catboost_model import DemandRegressor


cat_features = [
        "center_id", "meal_id", "emailer_for_promotion", 
        "homepage_featured", "category", "cuisine", 
        "city_code", "region_code", "center_type"
]
num_features = ["week", "checkout_price", "base_price", "op_area"]
feature_cols = num_features + cat_features

st.sidebar.markdown('**Как Вы хотите ввести данные?**')
input_type = st.sidebar.selectbox("", ('Ввести данные с клавиатуры', 'Загрузить файл'))

unique_center_id = joblib.load('unique_center_id.pkl')
unique_meal_id = joblib.load('unique_meal_id.pkl')

st.title('Meal Demand Forecasting')

st.info(
 		"""
    		В меню слева можно выбрать способ ввода информации для прогноза.
    		"""
				)

@st.cache
def get_fulfilment_center_info():
	fulfilment_center_info = pd.read_csv('fulfilment_center_info.csv')
	return fulfilment_center_info

@st.cache
def get_meal_info():
	meal_info = pd.read_csv('meal_info.csv')
	return meal_info

def create_df(week, checkout_price, base_price, center_id, meal_id, emailer_for_promotion, homepage_featured):
	city_code = fulfilment_center_info[fulfilment_center_info['center_id'] == center_id]['city_code'].tolist()[0]
	region_code = fulfilment_center_info[fulfilment_center_info['center_id'] == center_id]['region_code'].tolist()[0]
	op_area = fulfilment_center_info[fulfilment_center_info['center_id'] == center_id]['op_area'].tolist()[0]
	center_type = fulfilment_center_info[fulfilment_center_info['center_id'] == center_id]['center_type'].tolist()[0]
		
	category = meal_info[meal_info['meal_id'] == meal_id]['category'].tolist()[0]
	cuisine =  meal_info[meal_info['meal_id'] == meal_id]['cuisine'].tolist()[0]

	df = pd.DataFrame({'week': [week], \
    	'checkout_price': [checkout_price], \
		'base_price': [base_price], \
		'op_area': [op_area], \
		'meal_id': [meal_id], \
		'emailer_for_promotion': [emailer_for_promotion], \
		'homepage_featured': [homepage_featured], \
		'category': [category], \
		'center_id': [center_id], \
		'cuisine': [cuisine], \
    	'city_code': [city_code], \
    	'region_code': [region_code], \
    	'center_type': [center_type]})
	#df['op_area'] = df['op_area'].astype('float')
	return df

	

def get_input(input_type):
	if input_type == 'Ввести данные с клавиатуры':
		global meal_id
		global center_id
		global week

		st.markdown('**Заполните поля ниже, чтобы получить прогноз**')

		week = int(st.text_input('Введите текущую неделю:', '146'))
		checkout_price = st.text_input('Введите checkout price')
		base_price = st.text_input('Введите base price')
		center_id = st.selectbox('Выберите center_id', unique_center_id)
		meal_id = st.selectbox('Выберите meal_id', unique_meal_id)

		emailer_for_promotion = st.selectbox('Была ли e-mail рассылка?', ['False', 'True'])
		if emailer_for_promotion == 'True':
			emailer_for_promotion = 1
		else:
			emailer_for_promotion = 0

		homepage_featured = st.selectbox('Есть ли продукция на главной странице сайта?', ['False', 'True'])
		if homepage_featured == 'True':
			homepage_featured = 1
		else:
			homepage_featured = 0

		if checkout_price != '' and base_price != '' and center_id != '' and meal_id != '':
			checkout_price = int(checkout_price)
			base_price = int(base_price)
			center_id = int(center_id)
			meal_id = int(meal_id)
			df = create_df(week, checkout_price, base_price, center_id, meal_id, emailer_for_promotion, homepage_featured)
			return df
		pass

	elif input_type == 'Загрузить файл':
		st.info(
 			"""
    		Загрузите файл формата CSV.\n
    		Файл не должен превышать 200 MB
    		"""
				)
		data = st.file_uploader("", type=['csv'])
		if data is not None:
			try:
				df = pd.read_csv(data)
				df = df.merge(meal_info, on="meal_id", how="left").merge(
		                   fulfilment_center_info, on="center_id", how="left")
			except:
				 st.error(
				 	"""
				 	Файл не подходит для прогноза!
				  	Он должен содержать поля: "checkout_price", "base_price", "center_id", 
				  	"meal_id", "week", "emailer_for_promotion" и "homepage_featured".
				  	"""
				  	)
				 return None
			df = df[feature_cols]
			return df
		else:
			return None


fulfilment_center_info = get_fulfilment_center_info()
meal_info = get_meal_info()
data = get_input(input_type)

if data is not None:
	st.markdown('**Данные**')
	st.dataframe(data)


	model = DemandRegressor()
	prediction = model.predict_demand(data)
	if input_type == 'Ввести данные с клавиатуры':
		st.write(f'Прогноз спроса на продукцию {meal_id} для центра {center_id}\
				 на {week} неделю: {math.ceil(prediction)} единиц.')
	else:
		st.markdown('**Прогноз спроса на продукцию**')
		st.dataframe(data[['week', 'center_id', 'meal_id']].join(pd.DataFrame(prediction.astype(int))))

	st.markdown('**Влияние признаков**')
	st.set_option('deprecation.showPyplotGlobalUse', False)
	plot = model.explain(data)
	st.pyplot(plot)
