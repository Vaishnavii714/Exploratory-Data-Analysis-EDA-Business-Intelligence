import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("amazon_sales.csv")

df.columns = df.columns.str.strip().str.lower()

print("Columns in Dataset:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDescriptive Statistics:")
print(df.describe())

if 'product line' in df.columns:
    print("\nProduct Line Distribution:")
    print(df['product line'].value_counts())

    plt.figure()
    df['product line'].value_counts().plot(kind='bar')
    plt.title("Product Line Distribution")
    plt.xlabel("Product Line")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')

    if 'total' in df.columns:
        monthly_revenue = df.groupby('month')['total'].sum()

        print("\nMonthly Revenue Trend:")
        print(monthly_revenue)

        plt.figure()
        monthly_revenue.plot()
        plt.title("Monthly Revenue Trend")
        plt.xlabel("Month")
        plt.ylabel("Total Revenue")
        plt.show()

if 'city' in df.columns and 'total' in df.columns:
    city_revenue = df.groupby('city')['total'].sum()

    print("\nRevenue by City:")
    print(city_revenue)

    plt.figure()
    city_revenue.plot(kind='bar')
    plt.title("Revenue by City")
    plt.xlabel("City")
    plt.ylabel("Total Revenue")
    plt.xticks(rotation=45)
    plt.show()

if 'total' in df.columns:
    avg_order_value = df['total'].mean()
    print("\nAverage Order Value:", avg_order_value)

if 'quantity' in df.columns and 'total' in df.columns:
    plt.figure()
    plt.scatter(df['quantity'], df['total'])
    plt.title("Quantity vs Total Revenue")
    plt.xlabel("Quantity")
    plt.ylabel("Total Revenue")
    plt.show()

plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Matrix")
plt.show()

print("\nEDA Analysis Completed Successfully")