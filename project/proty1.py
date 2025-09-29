import pandas as pd
file_path = "C:\Users\user\Documents\pranesh\OneDrive\Desktop\Pond Readings.xlsx"
xls = pd.ExcelFile(file_path)
df = pd.read_excel(xls, sheet_name="Pond Reading")
df_cleaned=df.drop(columns=['hardness'])
correlation_matrix=df_cleaned.select_dtypes(include=['float64','int64']).corr()
correlation_matrix=correlation_matrix[correlation_matrix.abs()>0.2]
correlation_matrix_filtered =correlation_matrix_filtered.dropna(how='all',axis=0).dropna(how='all',axis=1) 
print(correlation_matrix_filtered)