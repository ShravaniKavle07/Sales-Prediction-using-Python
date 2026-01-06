<p align="center">
  <img src="https://img.icons8.com/color/96/000000/sales-performance.png" alt="Project Logo"/>
</p>

<h1 align="center">📈 Sales Prediction Using Python & Machine Learning</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-Linear%20Regression%20%7C%20Random%20Forest-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-orange?style=for-the-badge"/>
</p>

---

##  **Project Overview**

This project aims to predict **Sales** based on advertising budgets in **TV, Radio, and Newspaper** using machine learning techniques.  
The dataset used is the popular **Advertising.csv**, which contains marketing spend across 200 observations.

Machine Learning Models used:

- **Linear Regression**
- **Random Forest Regressor**

Both models were evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Actual vs Predicted Visualizations

---

##  **Project Structure**
├── README.md                                                                                   
├── Advertising.csv                                                                             
├── sales_prediction_using_python.ipynb                                                         
├── streamlit_app.py                                                                            
└── requirements.txt



---

##  **Features**
✔ Clean data preprocessing  
✔ Train-test split  
✔ Linear Regression & Random Forest models  
✔ Performance evaluation  
✔ Visualization of predictions  
✔ Interactive **Streamlit Web App**  
✔ Ready for deployment on Render / HuggingFace Spaces  

---

##  **Installation & Setup**

### 
1️ Clone the Repository

git clone https://github.com/ShravaniKavle07/Sales-Prediction-ML.git
cd Sales-Prediction-ML

2️ Install Required Libraries
bash
Copy code
pip install -r requirements.txt

3️ Run the Jupyter Notebook
jupyter notebook

4️ Run the Streamlit App
streamlit run streamlit_app.py


##  **Model Performance**
- Linear Regression

MAE: 1.46
RMSE: 1.78
R² Score: 0.899

- Random Forest Regressor

MAE: 0.83
RMSE: 1.109
R² Score: 0.961
Random Forest performed best, providing 96% accuracy and a much lower error rate.

##  **Visualizations**
✔ Actual vs Predicted (Linear Regression)
Shows how closely the model fits the trend.

✔ Actual vs Predicted (Random Forest)
Almost a perfect diagonal line — minimal error.

##  **Tech Stack Used**
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit


 ##  **Streamlit App Preview**
Your Streamlit app allows users to input:

TV Budget
Radio Budget
Newspaper Budget
And immediately get a sales prediction with a clean UI.

##  **Future Enhancements**
Add additional ML models (XGBoost, LightGBM)
Hyperparameter tuning using GridSearchCV
- Deploy full dashboard using Streamlit Cloud
- Add time series forecasting
- Add SHAP (Explainable AI) visualizations
- Add automated outlier detection

## **Contributing**
Pull requests are welcome.
For major changes, please open an issue for discussion first.

License
This project is licensed under the MIT License.
