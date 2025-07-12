import sqlite3
import datetime
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np

# Connect to SQLite database
conn = sqlite3.connect('inventory.db')
cursor = conn.cursor()

# Create tables
cursor.execute('''
    CREATE TABLE IF NOT EXISTS inventory (
        product_id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_name TEXT NOT NULL,
        quantity INTEGER NOT NULL
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS sales (
        sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER,
        date TEXT,
        quantity_sold INTEGER,
        FOREIGN KEY(product_id) REFERENCES inventory(product_id)
    )
''')

conn.commit()

# Predict demand based on last 7 days
def predict_demand(product_id):
    today = datetime.date.today()
    week_ago = today - datetime.timedelta(days=7)
    cursor.execute('''
        SELECT quantity_sold FROM sales
        WHERE product_id = ? AND date BETWEEN ? AND ?
    ''', (product_id, str(week_ago), str(today)))
    sales = cursor.fetchall()
    sales_qty = [s[0] for s in sales]
    if sales_qty:
        avg = np.mean(sales_qty)
        return int(avg * 1.2)
    return 0

# Restocking alert
def check_restock():
    cursor.execute("SELECT * FROM inventory")
    items = cursor.fetchall()
    alerts = []
    for pid, name, qty in items:
        predicted = predict_demand(pid)
        if predicted > qty:
            alerts.append(f"{name} - Needed: {predicted}, Available: {qty}")
    return alerts

# GUI Application
root = tk.Tk()
root.title("Smart Inventory Management")

frame = ttk.Frame(root, padding=10)
frame.grid(row=0, column=0)

# Entry fields
tk.Label(frame, text="Product Name:").grid(row=0, column=0)
product_name_entry = tk.Entry(frame)
product_name_entry.grid(row=0, column=1)

tk.Label(frame, text="Quantity:").grid(row=1, column=0)
quantity_entry = tk.Entry(frame)
quantity_entry.grid(row=1, column=1)

# Add inventory
def add_inventory():
    name = product_name_entry.get()
    qty = quantity_entry.get()
    if name and qty.isdigit():
        cursor.execute("INSERT INTO inventory (product_name, quantity) VALUES (?, ?)", (name, int(qty)))
        conn.commit()
        messagebox.showinfo("Added", "Inventory item added!")
    else:
        messagebox.showerror("Error", "Enter valid name and quantity.")

# Record sale
def record_sale():
    pid = product_name_entry.get()
    qty = quantity_entry.get()
    if pid.isdigit() and qty.isdigit():
        pid = int(pid)
        qty = int(qty)
        cursor.execute("INSERT INTO sales (product_id, date, quantity_sold) VALUES (?, ?, ?)", 
                       (pid, str(datetime.date.today()), qty))
        cursor.execute("UPDATE inventory SET quantity = quantity - ? WHERE product_id = ?", (qty, pid))
        conn.commit()
        messagebox.showinfo("Success", "Sale recorded!")
    else:
        messagebox.showerror("Error", "Enter numeric Product ID and Quantity.")

# Show alerts
def show_alerts():
    alerts = check_restock()
    if alerts:
        messagebox.showwarning("Restock Alerts", "\n".join(alerts))
    else:
        messagebox.showinfo("All Good", "No restocking needed.")

# Buttons
ttk.Button(frame, text="Add Inventory", command=add_inventory).grid(row=2, column=0, pady=10)
ttk.Button(frame, text="Record Sale (PID in Name)", command=record_sale).grid(row=2, column=1, pady=10)
ttk.Button(frame, text="Check Restock Alerts", command=show_alerts).grid(row=3, column=0, columnspan=2, pady=10)

root.mainloop()