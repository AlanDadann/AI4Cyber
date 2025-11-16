from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import PowerTransformer



pt = PowerTransformer(method='yeo-johnson')


df['src_bytes'] = pt.fit_transform(df[['src_bytes']])  
df['dst_bytes'] = pt.fit_transform(df[['dst_bytes']])  
df['num_compromised'] = pt.fit_transform(df[['num_compromised']])  
df['num_root'] = pt.fit_transform(df[['num_root']])  
df['num_file_creations'] = pt.fit_transform(df[['num_file_creations']])  
df['num_access_files'] = pt.fit_transform(df[['num_access_files']])  


label_encoder = LabelEncoder()


y = df['outcome']
X = df.drop(columns=['outcome'])

df = df.drop(columns=['outcome'])

X = pd.get_dummies(X, drop_first=True)

y = label_encoder.fit_transform(y)

print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 

print(y_train)
print(y_test)
print(X_train_scaled)
print(X_test_scaled)
