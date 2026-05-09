import pymysql
import random
import os
from faker import Faker
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

fake = Faker('en_IN')

# ─── Database Connection ─────────────────────────────────────────
connection = pymysql.connect(
    host=os.getenv("DB_HOST", "localhost"),
    user=os.getenv("DB_USER", "root"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME", "ecommerce_ai")
)
cursor = connection.cursor()

# ─── Configuration ───────────────────────────────────────────────
NUM_USERS = 500
NUM_PRODUCTS = 200
NUM_ORDERS = 10000

# ─── Clear Old Data (IMPORTANT FIX) ──────────────────────────────
print("Clearing old data...")

cursor.execute("SET FOREIGN_KEY_CHECKS = 0")

cursor.execute("TRUNCATE TABLE orders")
cursor.execute("TRUNCATE TABLE products")
cursor.execute("TRUNCATE TABLE users")

cursor.execute("SET FOREIGN_KEY_CHECKS = 1")

print("  ✓ Old data cleared")

# ─── Step 1: Generate Users ──────────────────────────────────────
print(f"Generating {NUM_USERS} users...")

users_data = []
indian_cities = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Pune",
    "Chennai", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow",
    "Surat", "Nagpur", "Indore", "Bhopal", "Patna",
    "Chandigarh", "Noida", "Gurgaon", "Visakhapatnam", "Coimbatore"
]

for i in range(1, NUM_USERS + 1):
    name = fake.name()
    email = fake.unique.email()
    city = random.choice(indian_cities)
    users_data.append((i, name, email, city))

cursor.executemany(
    "INSERT INTO users(user_id, name, email, city) VALUES (%s, %s, %s, %s)",
    users_data
)

print(f"  ✓ {NUM_USERS} users inserted")

# ─── Step 2: Generate Products ───────────────────────────────────
print(f"Generating {NUM_PRODUCTS} products...")

categories = [
    "electronics", "clothing", "home & kitchen",
    "sports", "books", "beauty", "toys", "automotive"
]

product_names = {
    "electronics": [
        "Wireless Bluetooth Earbuds", "Smart Watch", "Laptop", "Gaming Mouse",
        "LED Monitor", "Power Bank", "Noise Cancelling Headphones"
    ],
    "clothing": [
        "Men's T-Shirt", "Women's Dress", "Jeans", "Jacket",
        "Kurti", "Hoodie", "Formal Shirt"
    ],
    "home & kitchen": [
        "Cookware Set", "Mixer Grinder", "Electric Kettle",
        "Storage Containers", "Dining Set", "Gas Stove"
    ],
    "sports": [
        "Cricket Bat", "Football", "Badminton Racket",
        "Gym Dumbbells", "Yoga Mat", "Tennis Ball Set"
    ],
    "books": [
        "Self-Help Book", "Science Book", "Novel",
        "Story Book", "Biography", "Motivational Book"
    ],
    "beauty": [
        "Face Wash", "Lipstick", "Face Cream",
        "Hair Oil", "Makeup Kit", "Sunscreen Lotion"
    ],
    "toys": [
        "Teddy Bear", "RC Car", "Puzzle Game",
        "Building Blocks", "Doll Set", "Toy Train"
    ],
    "automotive": [
        "Car Engine Oil", "Bike Helmet", "Car Cover",
        "Phone Holder", "Car Vacuum Cleaner"
    ]
}

products_data = []

for i in range(1, NUM_PRODUCTS + 1):
    category = random.choice(categories)
    base_name = random.choice(product_names[category])
    prefix = random.choice(["Premium", "Classic", "Smart", "Pro", "Advanced", "Ultra"])

    title = f"{prefix} {base_name}"
    price = round(random.uniform(5.99, 499.99), 2)

    products_data.append((i, title, price, category))

cursor.executemany(
    "INSERT INTO products(product_id, title, price, category) VALUES (%s, %s, %s, %s)",
    products_data
)

print(f"  ✓ {NUM_PRODUCTS} products inserted")

# ─── Step 3: Generate Orders ─────────────────────────────────────
print(f"Generating {NUM_ORDERS} orders...")

regions = ["North", "South", "East", "West"]
payment_statuses = ["Paid", "Paid", "Paid", "Pending", "Failed"]

orders_data = []

for i in range(1, NUM_ORDERS + 1):
    user_id = random.randint(1, NUM_USERS)
    product_id = random.randint(1, NUM_PRODUCTS)
    quantity = random.randint(1, 5)

    order_date = (datetime.now() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d")
    
    region = random.choice(regions)
    payment_status = random.choice(payment_statuses)

    orders_data.append((i, user_id, product_id, quantity, order_date, region, payment_status))

cursor.executemany(
    "INSERT INTO orders(order_id, user_id, product_id, quantity, order_date, region, payment_status) VALUES (%s, %s, %s, %s, %s, %s, %s)",
    orders_data
)

print(f"  ✓ {NUM_ORDERS} orders inserted")

# ─── Commit & Close ──────────────────────────────────────────────
connection.commit()
cursor.close()
connection.close()

print("\n✅ Data generation complete!")
print(f"   Users    : {NUM_USERS}")
print(f"   Products : {NUM_PRODUCTS}")
print(f"   Orders   : {NUM_ORDERS}")