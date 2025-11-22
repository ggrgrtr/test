import sqlite3 as sq


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
    print("-"*10, 'drawed',"-"*10)
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
        print(table.upper(),':\n',sep='')
        column_names = [description[0] for description in cursor.description]
        print(*column_names)
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
    # result будет ссылаться на упорядоченный список,
    # состоящий из кортежей с данными таблицы
    # print(*result,sep='\n')

    # ля экономии памяти
    for result in cur:
        print(result)

    # cur.fetchone() # --  возвращает первую запись
    # fetchmany(size) # -– возвращает число записей не более size

    # delete table
    cur.execute("DELETE FROM users")
    cur.execute("DELETE FROM games")
    cur.execute("DROP TABLE if exists users")
    cur.execute("DROP TABLE if exists games")
    # cur.execute("DROP TABLE games")


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

# add to table games
f("INSERT INTO games VALUES(1, 100, 300)")
f("INSERT INTO games VALUES(2, 30, 17)")
f("INSERT INTO games VALUES(3, 90, 70)")
f("INSERT INTO games VALUES(2, 203, 170)")
f("INSERT INTO games VALUES(2, 10, 40)")
f("INSERT INTO games VALUES(4, 104, 11)")
f("INSERT INTO games VALUES(1, 99, 34)")
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

print("группировка по id и суммирование очков games:")
# группирует записи по указанному столбцу
# сортируем сумму по убыванию
draw("""
SELECT user_id, sum(score) as sum FROM games 
WHERE time>50
GROUP BY user_id
ORDER BY sum DESC
""")
# LIMIT 1 -- ограничение отбираемых записей

q('games',False)

q('users',True)


draw(
    """
    SELECT name, sex, games.score FROM games 
    JOIN users ON games.user_id = users.rowid
    """
)

print("объединение сыгранных игр и пользователей")
draw("""
select name, sex, games.score from games, users
""")
print()

print("таблица games с дополнениями из users:")
draw("""
SELECT name, sex, games.score FROM games
LEFT JOIN users ON games.user_id = users.rowid
""")

print('best players for all rounds:')
draw(
    """
    select name, sex, sum(games.score) as score FROM games
    join users on games.user_id = users.rowid
    group by user_id
    order by score desc 
    """
)

print("concotinuation:")
# оператор UNION оставляет только уникальные значения записей
draw(
    """
    SELECT time, user_id, games.score FROM games
    UNION SELECT name, sex, old FROM users    
    """
)

print("renamed concotinuation with column 'tbl':")
draw(
    """
SELECT games.score, 'table 1' as tbl FROM games
UNION SELECT old, 'table 2' FROM users
order by games.score desc 
    """
)

# добавление нового столбца
f("alter table users add column lastname TEXT")
q('users')
f("insert into users values('Egor', 1,30,4000, 'Mishuchkov')")
q('users')
