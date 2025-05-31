from flask import Flask, render_template, request, redirect, url_for
import mysql.connector
from datetime import datetime

app = Flask(__name__)

# 配置 MySQL 数据库
db = mysql.connector.connect(
    host="localhost",
    user="root",        
    password="*WXrxj2681", 
    database="book_management"
)
cursor = db.cursor()

@app.route('/')
def index():
    cursor.execute("SELECT * FROM books WHERE available = 1")
    books = cursor.fetchall()
    return render_template('index.html', books=books)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        user = cursor.fetchone()
        if user:
            return redirect(url_for('dashboard', user_id=user[0]))
        else:
            return "登录失败！用户名或密码错误"
    return render_template('login.html')

@app.route('/dashboard/<int:user_id>')
def dashboard(user_id):
    cursor.execute("SELECT * FROM books")
    books = cursor.fetchall()
    cursor.execute("SELECT * FROM borrowed_books WHERE user_id = %s", (user_id,))
    borrowed_books = cursor.fetchall()
    return render_template('dashboard.html', books=books, borrowed_books=borrowed_books, user_id=user_id)

@app.route('/borrow/<int:user_id>/<int:book_id>')
def borrow_book(user_id, book_id):
    cursor.execute("SELECT available FROM books WHERE id = %s", (book_id,))
    book = cursor.fetchone()
    if book and book[0] == 1:
        cursor.execute("UPDATE books SET available = 0 WHERE id = %s", (book_id,))
        cursor.execute("INSERT INTO borrowed_books (user_id, book_id, borrow_date) VALUES (%s, %s, %s)", 
                       (user_id, book_id, datetime.now().date()))
        db.commit()
        return redirect(url_for('dashboard', user_id=user_id))
    return "书籍已被借出"

@app.route('/return/<int:user_id>/<int:book_id>')
def return_book(user_id, book_id):
    cursor.execute("UPDATE books SET available = 1 WHERE id = %s", (book_id,))
    cursor.execute("UPDATE borrowed_books SET return_date = %s WHERE user_id = %s AND book_id = %s AND return_date IS NULL",
                   (datetime.now().date(), user_id, book_id))
    db.commit()
    return redirect(url_for('dashboard', user_id=user_id))

if __name__ == '__main__':
    app.run(debug=True)
