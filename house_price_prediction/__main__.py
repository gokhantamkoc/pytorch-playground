import pandas

df = pandas.read_csv("data.csv")

# we checked column values with df.iloc[0] drop unusable columns
df.pop('date')
df.pop('street')
df.pop('statezip')

city_names_unique = df['country'].unique()
print(
    f"Country Names: {city_names_unique}"
)
# we dropped country after checking df['country'].unique(), because every house is located in USA
df.pop('country')

print(df.head())
print(f"# of Rows: {len(df.index)}")
print(f"Samples \n {df.iloc[1]}")

# string values are not usable for deep learning models. Therefore, we apply one hot encoding to them.
city_names_unique = df['city'].unique()
for city_name in city_names_unique:
    df[city_name.lower().replace(" ", "_")] = 0

for city_name in city_names_unique:
    df_filter = df['city'] == city_name
    df.loc[df_filter, city_name.lower().replace(" ", "_")] = 1

# print(f"Samples \n {df.iloc[100]}")
df.pop('city')


print(f"Samples \n {df.iloc[100]}")



