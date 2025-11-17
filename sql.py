import sqlite3 as sq

connection = sq.connect('saber.db')
cursor = connection.cursor()
# передает SQL запрос
# NOT NULL -- обязан содержать данные
cursor.execute("""CREATE TABLE IF NOT EXISTS users (
    name TEXT NOT NULL,
    sex INTEGER NOT NULL DEFAULT 1,
    old INTEGER,
    score INTEGER
)""")
cursor.execute("""CREATE TABLE IF NOT EXISTS games (
    user_id INTEGER NOT NULL,
    score INTEGER NOT NULL DEFAULT 0,
    time INTEGER
)""")
connection.close()


# INSERT INTO <table_name> VALUES (<value1>, <value2>, …)

# SELECT * FROM users WHERE score >12
# SELECT * FROM users WHERE score BETWEEN 500 AND 1000
# SELECT * FROM users WHERE old = 19
# SELECT * FROM users WHERE old IN(19, 32) AND score > 300 OR sex = 1

# ORDER BY -- отсортировать по стобцу
# ORDER BY " " DESC -- тсортировать по убыванию

# выведет стобцы name, old, score из таблицы users
# SELECT name, old, score FROM users


# указываем, что число очков должно быть не менее 100
# затем, данные сортируются по убыванию очков и отбираются первые пять записей:
# SELECT * FROM users WHERE score > 100 ORDER BY score DESC LIMIT 5

# параметр OFFSET позволяет пропускать несколько первых записей
# LIMIT 5 OFFSET 2 == LIMIT 2, 5


def draw(string):
    with sq.connect("saber.db") as connection:
        cursor = connection.cursor()
        cursor.execute(string)

        for result in cursor:
            print(*result)
        print()


def q(table,rowid=False):
    with sq.connect("saber.db") as connection:
        cursor = connection.cursor()
        if rowid:
            cursor.execute(f"SELECT rowid, * FROM {table}")
        else:
            cursor.execute(f"SELECT * FROM {table}")
        print()
        print("q()-------------------------------------")
        for result in cursor:
            print(result)
        print("q()-------------------------------------")
        print()


def f(string):
    with sq.connect("saber.db") as connection:
        cursor = connection.cursor()
        cursor.execute(string)


def add_person(name, sex, age, score):
    with sq.connect("saber.db") as connection:
        curs = connection.cursor()
        curs.execute(f"INSERT INTO users VALUES('{name}', {sex}, {age}, {score})")


# автоматически сохраняет данные в БД - вызывает метод commit()
with sq.connect("saber.db") as connection:
    cur = connection.cursor()

    cur.execute("SELECT * FROM users WHERE score > 10 ORDER BY score DESC LIMIT 5")

    # для получения результатов отбора SQL-запроса
    # result = cur.fetchall()
    # result будет ссылаться на упорядоченный список, состоящий из кортежей с данными таблицы
    # print(*result,sep='\n')

    # ля экономии памяти
    for result in cur:
        print(result)

    # cur.fetchone() # --  возвращает первую запись
    # fetchmany(size) # -– возвращает число записей не более size

    # delete table
    # cur.execute("DROP TABLE users")
    cur.execute("DELETE FROM users")
    cur.execute("DELETE FROM games")
    # cur.execute("DROP TABLE games")


# -------------------
# UPDATE имя_таблицы SET имя_столбца = новое_значение WHERE условие
# -------------------


add_person("John", 2, 35, 35)
add_person('bro',1,20,400)
add_person('Lena',1,18,950)
add_person('Misha',1,22,80)
add_person('Egor',1,20,900)
f("INSERT INTO users VALUES('Даша', 2, 24, 1200)")
q('users',True)


print("deleted №2, №5:")
# DELETE FROM имя_таблицы WHERE условие
# rowid - скрытый индекс строки
# rowid №2, №5
f('DELETE FROM users WHERE rowid IN(2, 5)')
q('users',True)

f("UPDATE users SET score = score+500 WHERE sex = 1")
q('users')

# игрокам, у которых имя начинается с буквы М будет добавлено 100 очков
f("UPDATE users SET score = score+100 WHERE name LIKE 'M%'")
f("UPDATE users SET score = score+100 WHERE name LIKE 'J_hn%'")
q('users')

f("UPDATE users SET score = score+333, old = 98 WHERE old > 30")
q('users')

f("INSERT INTO games VALUES(1, 100, 300)")
f("INSERT INTO games VALUES(2, 30, 17)")
f("INSERT INTO games VALUES(3, 90, 70)")
q('games')

print("считаем все user_id = 1:")
# коmaнда count() для переменной count
draw("SELECT count() as count FROM games WHERE user_id = 1")

print("считаем все уникальные (DISTINCT) user_id:")
draw("SELECT count(DISTINCT user_id) as count FROM games")

print("Выводим очки первого ID:")
draw("SELECT score FROM games WHERE user_id=1")

print("Выводим очки скрытого индекса rowid:")
draw("SELECT score FROM games WHERE rowid=2")

print("sum of all scores:")
draw("SELECT sum(score) as sum FROM games")

q('games',True)
