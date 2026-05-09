import pymysql
import pandas as pd
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

# Database Connection
def get_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )

# 1.KPI Cards
def get_total_revenue():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ROUND(SUM(o.quantity * p.price), 2) AS total_revenue
        FROM orders o
        JOIN products p ON o.product_id = p.product_id
        WHERE o.payment_status = 'Paid'
    """)
    result = cursor.fetchone()
    conn.close()
    return float(result[0]) if result[0] else 0.0

def get_total_orders():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM orders")
    result = cursor.fetchone()
    conn.close()
    return int(result[0])

def get_total_customers():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    result = cursor.fetchone()
    conn.close()
    return int(result[0])

def get_total_products():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM products")
    result = cursor.fetchone()
    conn.close()
    return int(result[0])

#2. Monthly Revenue Trend
def get_monthly_revenue():
    conn = get_connection()
    query = """
        SELECT 
            DATE_FORMAT(o.order_date, '%b %Y') AS Month,
            MONTHNAME(o.order_date) AS MonthName,
            MONTH(o.order_date) AS MonthNum,
            YEAR(o.order_date) AS Year,
            ROUND(SUM(o.quantity * p.price), 2) AS Revenue
        FROM orders o
        JOIN products p ON o.product_id = p.product_id
        WHERE o.payment_status = 'Paid'
        GROUP BY Year, MonthNum, MonthName, Month
        ORDER BY Year, MonthNum
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# 2b. Weekly Revenue Trend
def get_weekly_revenue():
    conn = get_connection()
    query = """
        SELECT 
            YEARWEEK(o.order_date, 1) AS YearWeek,
            DATE_FORMAT(MIN(o.order_date), '%d %b') AS Week_Start,
            ROUND(SUM(o.quantity * p.price), 2) AS Revenue
        FROM orders o
        JOIN products p ON o.product_id = p.product_id
        WHERE o.payment_status = 'Paid'
          AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 WEEK)
        GROUP BY YearWeek
        ORDER BY YearWeek
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

#3. City-wise Sales
def get_city_sales():
    conn = get_connection()
    query = """
        SELECT 
            u.city AS City,
            ROUND(SUM(o.quantity * p.price), 2) AS Revenue
        FROM orders o
        JOIN products p ON o.product_id = p.product_id
        JOIN users u ON o.user_id = u.user_id
        WHERE o.payment_status = 'Paid'
        GROUP BY u.city
        ORDER BY Revenue DESC
        LIMIT 10
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# 4. Category-wise Revenue
def get_category_sales():
    conn = get_connection()
    query = """
        SELECT 
            p.category AS Category,
            ROUND(SUM(o.quantity * p.price), 2) AS Revenue
        FROM orders o
        JOIN products p ON o.product_id = p.product_id
        WHERE o.payment_status = 'Paid'
        GROUP BY p.category
        ORDER BY Revenue DESC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# 5. Top 10 Customers
def get_top_customers():
    conn = get_connection()
    query = """
        SELECT 
            u.name AS Customer,
            ROUND(SUM(o.quantity * p.price), 2) AS Total_Spent
        FROM orders o
        JOIN users u ON o.user_id = u.user_id
        JOIN products p ON o.product_id = p.product_id
        WHERE o.payment_status = 'Paid'
        GROUP BY u.user_id, u.name
        ORDER BY Total_Spent DESC
        LIMIT 10
    """
    df = pd.read_sql(query, conn)
    conn.close()
    # Format as currency string
    df["Total_Spent"] = df["Total_Spent"].apply(lambda x: f"₹{x:,.2f}")
    return df

# 6. Order Status Summary
def get_order_status():
    conn = get_connection()
    query = """
        SELECT 
            payment_status AS Status,
            COUNT(*) AS Count
        FROM orders
        GROUP BY payment_status
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# 7. Average Order Value
def get_avg_order_value():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ROUND(SUM(o.quantity * p.price) / COUNT(DISTINCT o.order_id), 2)
        FROM orders o
        JOIN products p ON o.product_id = p.product_id
        WHERE o.payment_status = 'Paid'
    """)
    result = cursor.fetchone()
    conn.close()
    return float(result[0]) if result[0] else 0.0

# 8. New Customers This Month
def get_new_customers_this_month():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(DISTINCT user_id)
        FROM orders
        WHERE MONTH(order_date) = MONTH(CURDATE())
          AND YEAR(order_date) = YEAR(CURDATE())
          AND user_id NOT IN (
              SELECT DISTINCT user_id FROM orders
              WHERE order_date < DATE_FORMAT(CURDATE(), '%Y-%m-01')
          )
    """)
    result = cursor.fetchone()
    conn.close()
    return int(result[0]) if result[0] else 0

# 9. Repeat Customer Rate (%)
def get_repeat_customer_rate():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            ROUND(
                COUNT(CASE WHEN order_count > 1 THEN 1 END) * 100.0 / COUNT(*), 1
            ) AS repeat_rate
        FROM (
            SELECT user_id, COUNT(order_id) AS order_count
            FROM orders
            WHERE payment_status = 'Paid'
            GROUP BY user_id
        ) AS customer_orders
    """)
    result = cursor.fetchone()
    conn.close()
    return float(result[0]) if result[0] else 0.0

# 10. Top Products by Revenue
def get_top_products(limit=5):
    conn = get_connection()
    query = f"""
        SELECT 
            p.title AS Product,
            p.category AS Category,
            SUM(o.quantity) AS Units_Sold,
            ROUND(SUM(o.quantity * p.price), 2) AS Revenue
        FROM orders o
        JOIN products p ON o.product_id = p.product_id
        WHERE o.payment_status = 'Paid'
        GROUP BY p.product_id, p.title, p.category
        ORDER BY Revenue DESC
        LIMIT {limit}
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# 11. Sales 
def get_sales_funnel():
    conn = get_connection()
    cursor = conn.cursor()

    # Total orders placed (Paid)
    cursor.execute("SELECT COUNT(*) FROM orders WHERE payment_status = 'Paid'")
    paid = int(cursor.fetchone()[0])

    # All orders including unpaid (attempted checkouts)
    cursor.execute("SELECT COUNT(*) FROM orders")
    total_orders = int(cursor.fetchone()[0])

    # Unique users who ever ordered
    cursor.execute("SELECT COUNT(DISTINCT user_id) FROM orders")
    buyers = int(cursor.fetchone()[0])

    # Total users (visitors proxy)
    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = int(cursor.fetchone()[0])

    conn.close()

    return {
        "visitors":      total_users,
        "add_to_cart":   buyers,
        "checkout":      total_orders,
        "orders_placed": paid,
    }

# 12. Customer Segments (RFM-based)
def get_customer_segments():
    conn = get_connection()
    query = """
        SELECT 
            user_id,
            COUNT(order_id)                    AS frequency,
            ROUND(SUM(quantity * 1), 0)        AS monetary_proxy,
            DATEDIFF(CURDATE(), MAX(order_date)) AS recency_days
        FROM orders
        WHERE payment_status = 'Paid'
        GROUP BY user_id
    """
    df = pd.read_sql(query, conn)
    conn.close()

    def segment(row):
        if row['recency_days'] <= 30 and row['frequency'] >= 5:
            return 'Champions'
        elif row['recency_days'] <= 60 and row['frequency'] >= 3:
            return 'Loyal'
        elif row['recency_days'] <= 90:
            return 'Potential'
        elif row['recency_days'] <= 180:
            return 'At-risk'
        else:
            return 'Lost'

    df['Segment'] = df.apply(segment, axis=1)
    seg_counts = df['Segment'].value_counts().reset_index()
    seg_counts.columns = ['Segment', 'Count']
    return seg_counts

# 13. Enhanced Top Customers with extra columns
def get_top_customers_enhanced(limit=10):
    conn = get_connection()
    query = f"""
        SELECT 
            u.name AS Customer,
            COUNT(DISTINCT o.order_id)                      AS Orders,
            ROUND(SUM(o.quantity * p.price), 2)             AS Total_Spent,
            ROUND(SUM(o.quantity * p.price) / COUNT(DISTINCT o.order_id), 2) AS Avg_Order
        FROM orders o
        JOIN users u ON o.user_id = u.user_id
        JOIN products p ON o.product_id = p.product_id
        WHERE o.payment_status = 'Paid'
        GROUP BY u.user_id, u.name
        ORDER BY Total_Spent DESC
        LIMIT {limit}
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df['Total_Spent'] = df['Total_Spent'].apply(lambda x: f"₹{x:,.2f}")
    df['Avg_Order']   = df['Avg_Order'].apply(lambda x: f"₹{x:,.2f}")
    return df

# 14. Fulfillment Rate (Paid / Total orders %)
def get_fulfillment_rate():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            ROUND(
                SUM(CASE WHEN payment_status = 'Paid' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1
            )
        FROM orders
    """)
    result = cursor.fetchone()
    conn.close()
    return float(result[0]) if result[0] else 0.0