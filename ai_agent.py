import os
import re
import json
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from langchain_groq import ChatGroq

# ─── 1. INITIALIZATION ───────────────────────────────────────────
load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

def get_engine():
    connection_url = URL.create(
        drivername="mysql+pymysql",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
    )
    return create_engine(connection_url)

engine = get_engine()

# ─── 2. SCHEMA + REGION CONTEXT (only thing that stays fixed) ────
DB_SCHEMA = """
Tables:
1. orders   (order_id, user_id, product_id, quantity, payment_status, order_date)
2. products (product_id, title, category, price)
3. users    (user_id, name, email, city)

Important notes:
- Revenue = SUM(o.quantity * p.price)
- Always filter: WHERE o.payment_status = 'Paid'
- Region is NOT a column. Always derive it from users.city using this CASE:
    CASE
        WHEN u.city IN ('Mumbai','Pune','Ahmedabad','Surat','Nagpur')                    THEN 'West'
        WHEN u.city IN ('Delhi','Jaipur','Lucknow','Chandigarh','Noida','Gurgaon')       THEN 'North'
        WHEN u.city IN ('Kolkata','Patna','Bhopal','Indore')                             THEN 'East'
        WHEN u.city IN ('Bangalore','Hyderabad','Chennai','Coimbatore','Visakhapatnam')  THEN 'South'
        ELSE 'Other'
    END
- For monthly queries: GROUP BY MONTH(o.order_date), MONTHNAME(o.order_date) ORDER BY MONTH(o.order_date)
- Database: MySQL
"""

# ─── 3. GENERATE SQL USING LLM ───────────────────────────────────
def generate_sql(user_query):
    prompt = f"""
You are an expert MySQL analyst working on an e-commerce database.

{DB_SCHEMA}

Examples:
Q: What is the total revenue?
A: SELECT SUM(o.quantity * p.price) AS total_revenue FROM orders o JOIN products p ON o.product_id = p.product_id WHERE o.payment_status = 'Paid';

Q: Which city has the lowest sales?
A: SELECT u.city, SUM(o.quantity * p.price) AS total_sales FROM orders o JOIN users u ON o.user_id = u.user_id JOIN products p ON o.product_id = p.product_id WHERE o.payment_status = 'Paid' GROUP BY u.city ORDER BY total_sales ASC LIMIT 1;

Q: Revenue by region?
A: SELECT region, SUM(sales) AS revenue
FROM (
    SELECT 
        CASE 
            WHEN u.city IN ('Mumbai','Pune','Ahmedabad','Surat','Nagpur') THEN 'West'
            WHEN u.city IN ('Delhi','Jaipur','Lucknow','Chandigarh','Noida','Gurgaon') THEN 'North'
            WHEN u.city IN ('Kolkata','Patna','Bhopal','Indore') THEN 'East'
            WHEN u.city IN ('Bangalore','Hyderabad','Chennai','Coimbatore','Visakhapatnam') THEN 'South'
            ELSE 'Other'
        END AS region,
        (o.quantity * p.price) AS sales
    FROM orders o
    JOIN users u ON o.user_id = u.user_id
    JOIN products p ON o.product_id = p.product_id
    WHERE o.payment_status = 'Paid'
) AS sub
GROUP BY region
ORDER BY revenue DESC;

Q: Top 5 customers by spending?
A: SELECT u.name, SUM(o.quantity * p.price) AS total_spent FROM orders o JOIN users u ON o.user_id = u.user_id JOIN products p ON o.product_id = p.product_id WHERE o.payment_status = 'Paid' GROUP BY u.user_id, u.name ORDER BY total_spent DESC LIMIT 5;

Now answer ONLY with a valid MySQL SELECT statement ending in semicolon. No explanation.
Q: {user_query}
A:"""

    response = llm.invoke(prompt)
    raw = response.content.strip()
    raw = re.sub(r"```sql|```", "", raw).strip()
    match = re.search(r"(SELECT[\s\S]+?;)", raw, re.IGNORECASE)
    return match.group(1).strip() if match else "CANNOT_ANSWER"


# ─── 4. DETECT SCALAR VS TABLE RESULT ────────────────────────────
def is_scalar_query(sql):
    sql_upper = sql.upper()
    has_aggregate = bool(re.search(r"\b(SUM|AVG|COUNT|MAX|MIN)\s*\(", sql_upper))
    has_group_by  = "GROUP BY" in sql_upper
    has_limit_1   = bool(re.search(r"LIMIT\s+1\b", sql_upper))
    return (has_aggregate and not has_group_by) or has_limit_1


# ─── 5. RUN SQL ──────────────────────────────────────────────────
def run_sql(sql, fetch_one=False):
    with engine.connect() as conn:
        result = conn.execute(text(sql)).mappings()
        if fetch_one:
            row = result.fetchone()
            return dict(row) if row else None
        return [dict(row) for row in result.fetchall()]


# ─── 6. FORMAT RESULT ────────────────────────────────────────────
def format_result(raw, fetch_one=False):
    if raw is None:
        return None
    if fetch_one and isinstance(raw, dict):
        values = list(raw.values())
        return values[0] if len(values) == 1 else raw
    if isinstance(raw, list) and len(raw) == 0:
        return "No data found."
    return raw


# ─── 7. GENERATE INSIGHT ─────────────────────────────────────────
def generate_insight(data, user_query):
    if isinstance(data, (int, float)):
        return f"The answer is {data:,.2f}."
    if isinstance(data, str):
        return data

    prompt = f"""
You are an e-commerce business analyst.

The user asked: "{user_query}"

Query result:
{json.dumps(data, indent=2, default=str)}

Give:
1. A direct, clear answer to the question
2. One short business insight or recommendation

Keep it to 3-4 sentences max.
"""
    return llm.invoke(prompt).content.strip()


# ─── 8. MAIN ENTRY POINT ─────────────────────────────────────────
def run_query(user_query, conversation_context=None):

    # 🔥 FIX: Handle region query manually (prevents GROUP BY error)
    if "region" in user_query.lower():
        sql = """
        SELECT region, SUM(sales) AS sales
        FROM (
            SELECT 
                CASE 
                    WHEN u.city IN ('Mumbai','Pune','Ahmedabad','Surat','Nagpur') THEN 'West'
                    WHEN u.city IN ('Delhi','Jaipur','Lucknow','Chandigarh','Noida','Gurgaon') THEN 'North'
                    WHEN u.city IN ('Kolkata','Patna','Bhopal','Indore') THEN 'East'
                    WHEN u.city IN ('Bangalore','Hyderabad','Chennai','Coimbatore','Visakhapatnam') THEN 'South'
                    ELSE 'Other'
                END AS region,
                (o.quantity * p.price) AS sales
            FROM orders o
            JOIN users u ON o.user_id = u.user_id
            JOIN products p ON o.product_id = p.product_id
            WHERE o.payment_status = 'Paid'
        ) AS sub
        GROUP BY region
        ORDER BY sales DESC;
        """
    else:
        sql = generate_sql(user_query)

    if sql == "CANNOT_ANSWER":
        return {"error": "Could not generate a valid SQL query for this question."}

    fetch_one = is_scalar_query(sql)

    try:
        raw_result = run_sql(sql, fetch_one=fetch_one)
    except Exception as e:
        return {"error": f"SQL execution failed: {str(e)}", "sql": sql}

    data    = format_result(raw_result, fetch_one=fetch_one)
    insight = generate_insight(data, user_query)

    return {
        "query":   user_query,
        "sql":     sql,
        "data":    data,
        "insight": insight
    }
# ─── 9. CLI LOOP ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("E-Commerce AI Analytics — type 'exit' to quit\n")
    while True:
        q = input("Ask: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        res = run_query(q)

        if "error" in res:
            print(f"\n Error: {res['error']}")
            if "sql" in res:
                print(f"   SQL tried: {res['sql']}")
        else:
            print(f"\n Answer:\n{res['insight']}\n")