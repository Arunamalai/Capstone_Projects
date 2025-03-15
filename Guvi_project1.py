import streamlit as st
import pandas as pd
import mysql.connector
import os
import plotly.express as px



st.title("GUVI PROJECT-1")
col1, col2 = st.columns([3, 3])  # Adjust the column width ratio as needed

# Place the image in the second column (top-right)
with col2:
    st.image("C:/Users/Arunamalai/OneDrive/Gambar/New folder/Retail.png", caption="Retail Order Data Analysis", use_container_width=True)
st.header("Retail Order Data Analysis") 

st.subheader("Dashboard")

# MySQL connection
connection = mysql.connector.connect(
  host = "gateway01.us-west-2.prod.aws.tidbcloud.com",
  port = 4000,
  user = "2EDdC8GqjrzqdnT.root",
  password = "mxlVIkdYlR69SoPz",
  database = "demo"
)
mycursor = connection.cursor()

st.sidebar.title("Data Analysis with SQL")


# Sidebar filter for selecting the query
category_filter = st.sidebar.selectbox(
    "Select Category", 
    ['Top 10 highest revenue generating products', 
     'Top 5 cities with the highest profit margins', 
     'Total discount for each category', 
     'Average sale price per product category', 
     'Region with the highest average sale price',
     'Total profit per category',
     'Top 3 segments with the highest quantity of orders',
     'Average discount percentage per region',
    'Product category with the highest total profit',
    'Total revenue generated per year',
    'Top 3 product category with least profit', 
    'Product category with its income statement(profit or loss)',
    'Count of products having loss',
    'Rank products by its revenue',
    'Count the minimum and maximum discount amount',
    'Rank Products by its Profit_Margin_Percentage',
    'Top 10 city with the highest no of orders',
    'OrderCount based on shipmode for each city',
    'Top Selling product in year 2022',
    'Calculate Total profit in year2023']
)

# Adjust the SQL query based on the filter
if category_filter == 'Top 10 highest revenue generating products':
    query = """SELECT subcategory, SUM(Sale_Price) 
    FROM retail_order2 
    GROUP BY subcategory 
    ORDER BY SUM(Sale_Price) DESC LIMIT 10"""
elif category_filter == 'Top 5 cities with the highest profit margins':
    query = """SELECT City, (SUM(Profit)/SUM(Sale_Price)*100) AS Profit_Margin_Percentage
FROM (
    SELECT City, SUM(Sale_Price) AS Sale_Price, SUM(Profit) AS Profit
    FROM (
        SELECT ro1.order_id, ro1.City, ro2.Sale_Price, ro2.Profit
        FROM retail_order1 ro1
        LEFT JOIN retail_order2 ro2 ON ro1.order_id = ro2.order_id
    ) AS inner_subquery
    GROUP BY City
    ) AS outer_subquery
GROUP BY City ORDER BY Profit_Margin_Percentage DESC LIMIT 5;"""
# Add more queries as needed
elif category_filter == 'Total discount for each category':

    query = "SELECT Category, SUM(Discount_Amount) FROM retail_order2 GROUP BY Category ORDER BY SUM(Discount_Amount)DESC"
elif category_filter == 'Average sale price per product category': 
    query = "SELECT subcategory, AVG(Sale_Price) FROM retail_order2 GROUP BY subcategory" 
elif category_filter == 'Region with the highest average sale price':
    query ="""SELECT Region, AVG(Sale_Price)
     FROM (SELECT ro1.order_id, ro1.Region, ro2.Sale_Price FROM retail_order1 as ro1 
     INNER JOIN retail_order2 as ro2 on ro1.order_id=ro2.order_id) AS subquery
     GROUP BY Region ORDER BY AVG(Sale_Price) DESC"""
elif category_filter == 'Total profit per category':
   query ="SELECT category, SUM(Profit) FROM retail_order2 GROUP BY category"
elif category_filter == 'Top 3 segments with the highest quantity of orders':
   query ="SELECT segment,SUM(Quantity) FROM (SELECT ro1.Segment,ro2.Quantity FROM retail_order1 ro1 LEFT JOIN retail_order2 ro2 on ro1.order_ID=ro2.order_id) AS subquery GROUP BY Segment ORDER BY SUM(Quantity) DESC LIMIT 3"
elif category_filter == 'Average discount percentage per region':
   query ="SELECT Region,AVG(discount_percent) FROM (SELECT ro1.Region,ro2.discount_percent from retail_order1 ro1 LEFT JOIN Retail_order2 ro2 on ro1.order_id=ro2.order_id) As Subquery GROUP BY Region"
elif category_filter == 'Product category with the highest total profit':
   query ="SELECT subcategory, SUM(Profit) FROM retail_order2 GROUP BY subcategory ORDER BY SUM(Profit) DESC LIMIT 1"
elif category_filter == 'Total revenue generated per year':
   query ="SELECT EXTRACT(Year from order_date) AS Year, FORMAT(SUM(Sale_Price*Quantity),'N2') AS TotalRevenue FROM (SELECT ro1.order_date, ro2.Sale_Price, ro2.Quantity FROM retail_order1 ro1 LEFT JOIN retail_order2 ro2 on ro1.order_id=ro2.order_id) As subquery GROUP BY Year ORDER BY Year"

elif category_filter == 'Top 3 product category with least profit':
    query = """SELECT Subcategory, SUM(Profit) FROM retail_order2 GROUP BY Subcategory ORDER BY SUM(Profit) ASC LIMIT 3"""
elif category_filter == 'Product category with its income statement(profit or loss)':
    query="""SELECT 
        Subcategory,
        Profit,
        CASE
            WHEN Profit <= 0 THEN 'Loss'
            WHEN Profit BETWEEN 0 AND 200 THEN 'Profit'
            ELSE 'More Profit'
        END AS INCOME_STATEMENT
    FROM retail_order2 LIMIT 100"""
elif category_filter =='Count of products having loss':
   query="""
    SELECT Subcategory, 
           CASE
               WHEN Profit <= 0 THEN 'Loss'
               WHEN Profit BETWEEN 0 AND 200 THEN 'Profit'
               ELSE 'More Profit'
           END AS INCOME_STATEMENT, 
           COUNT(*) 
    FROM retail_order2 
    WHERE Profit <= 0
    GROUP BY Subcategory, INCOME_STATEMENT
"""
elif category_filter =='Rank products by its revenue':
   query="""SELECT 
    Subcategory, 
    SUM(Sale_Price) AS TotalRevenue,
    RANK() OVER (ORDER BY SUM(Sale_Price) DESC) AS RevenueRank
FROM (
    SELECT 
        Subcategory, 
        SUM(Sale_Price) AS Sale_Price
    FROM retail_order2
    GROUP BY Subcategory
) AS Subquery   
GROUP BY Subcategory
"""
elif category_filter =='Count the minimum and maximum discount amount':
   query="select min(Discount_Amount),max(Discount_Amount) from retail_order2"
elif category_filter =='Rank Products by its Profit_Margin_Percentage':
   query="""SELECT 
    Subcategory, 
    (SUM(Profit) / SUM(Sale_Price) * 100) AS Profit_Margin_Percentage,
    RANK() OVER (ORDER BY (SUM(Profit) / SUM(Sale_Price) * 100) DESC) AS Profit_Margin_Percentage_Rank
FROM (
    SELECT 
        Subcategory,
        SUM(Sale_Price) AS Sale_Price, 
        SUM(Profit) AS Profit
    FROM retail_order2
    GROUP BY Subcategory
) AS Subquery
GROUP BY Subcategory"""
elif category_filter =='Top 10 city with the highest no of orders':
   query="""
    SELECT City, COUNT(*) AS OrderCount
    FROM retail_order1
    GROUP BY City ORDER BY OrderCount DESC LIMIT 10
"""
elif category_filter =='OrderCount based on shipmode for each city':
   query="""
    SELECT City, ShipMode, COUNT(*) AS OrderCount
    FROM retail_order1
    GROUP BY City, ShipMode
"""
elif category_filter =='Top Selling product in year 2022':
   query="""SELECT 
    EXTRACT(YEAR FROM ro1.order_date) AS Year, 
    ro2.Subcategory, 
    SUM(ro2.Quantity) AS Total_sales
FROM 
    retail_order1 ro1 
LEFT JOIN 
    retail_order2 ro2 
ON 
    ro1.order_id = ro2.order_id
WHERE 
    EXTRACT(YEAR FROM ro1.order_date) = 2022
GROUP BY 
    EXTRACT(YEAR FROM ro1.order_date), ro2.Subcategory
ORDER BY 
    Total_sales DESC
LIMIT 1;
"""
elif category_filter =='Calculate Total profit in year2023':
   query="""SELECT 
    ro1.country, 
    EXTRACT(YEAR FROM ro1.order_date) AS Year,
    SUM(ro2.Profit) AS Total_Profit
FROM 
    retail_order1 ro1
LEFT JOIN 
    retail_order2 ro2 
ON 
    ro1.order_id = ro2.order_id
WHERE 
    EXTRACT(YEAR FROM ro1.order_date) = 2023
GROUP BY 
    ro1.country, EXTRACT(YEAR FROM ro1.order_date)
ORDER BY 
    Total_Profit DESC;
"""
else:
    query = "SELECT 'No query available for this selection.' AS message"  # Default query if no valid option

# Display the SQL query being executed
st.sidebar.write("Executed SQL Query:")
st.sidebar.code(query)

# Execute the selected query
try:
    mycursor.execute(query)
    out = mycursor.fetchall()

    # Convert the query output to a pandas DataFrame
    df = pd.DataFrame(out, columns=[i[0] for i in mycursor.description])

    # Handle empty data
    if df.empty:
        st.warning("No data available for the selected query.")
    else:
        st.write(f"Results for: {category_filter}")
        st.write(df)

        # Rename columns dynamically for chart display
        if 'SUM(Sale_Price)' in df.columns:
            df.rename(columns={'SUM(Sale_Price)': 'Total_Sale_Price'}, inplace=True)
        if 'SUM(Discount_Amount)' in df.columns:
            df.rename(columns={'SUM(Discount_Amount)': 'Total_Discount_Amount'}, inplace=True)

        # Display download button
        st.download_button(
            label="Download Data as CSV",
            data=df.to_csv(index=False),
            file_name="retail_analysis_results.csv",
            mime="text/csv"
        )
    
    
    # Display a bar chart of Sale Price by Subcategory
    if category_filter == 'Top 10 highest revenue generating products':
        st.bar_chart(df.set_index('subcategory')['Total_Sale_Price']) 
    elif category_filter == 'Top 5 cities with the highest profit margins':
     st.area_chart(df.set_index('City')['Profit_Margin_Percentage'])
    elif category_filter == 'Total discount for each category':
     st.bar_chart(df.set_index('Category')['Total_Discount_Amount'])
    elif category_filter == 'Average sale price per product category':
     st.bar_chart(df.set_index('subcategory')['AVG(Sale_Price)']) 
    elif category_filter == 'Region with the highest average sale price':
       st.bar_chart(df.set_index('Region')['AVG(Sale_Price)'])
    elif category_filter == 'Total profit per category':
       st.bar_chart(df.set_index('category')['SUM(Profit)'])
    elif category_filter == 'Top 3 segments with the highest quantity of orders':
       st.bar_chart(df.set_index('segment')['SUM(Quantity)'])
    elif category_filter == 'Average discount percentage per region':
       st.bar_chart(df.set_index('Region')['AVG(discount_percent)'])
    elif category_filter == 'Product category with the highest total profit':
       st.bar_chart(df.set_index('subcategory')['SUM(Profit)'])
    elif category_filter == 'Total revenue generated per year':
       st.bar_chart(df.set_index('Year')['TotalRevenue'])
    elif category_filter == 'Top 3 product category with least profit':
     st.bar_chart(df.set_index('Subcategory')['SUM(Profit)']) 
    elif category_filter == 'Count of products having loss':
     st.bar_chart(df.set_index('Subcategory')['COUNT(*)'])
    elif category_filter == 'Rank products by its revenue':
     st.bar_chart(df.set_index('Subcategory')['TotalRevenue']) 
    
    elif category_filter == 'Rank Products by its Profit_Margin_Percentage':
     st.bar_chart(df.set_index('Subcategory')['Profit_Margin_Percentage'])
    elif category_filter ==  'Top 10 city with the highest no of orders':
     st.bar_chart(df.set_index('City')['OrderCount'])
    else:
     pass

except Exception as e:
    st.error(f"Error executing query: {e}")

   

    

    
    






