**репозиторий с коммитами других проектов**

-

-

-

-


SQLite:


C:\Users\Egor\PycharmProjects\SQL_and\.venv\Scripts\python.exe C:\Users\Egor\PycharmProjects\SQL_and\file1.py 
('Lena', 1, 18, 1450, None)
('Даша', 2, 24, 1200, None)
('Misha', 1, 22, 680, None)
('John', 2, 98, 468, None)

q()-------------------------------------
USERS:

rowid name sex old score
(1, 'John', 2, 35, 35)
(2, 'bro', 1, 20, 400)
(3, 'Lena', 1, 18, 950)
(4, 'Misha', 1, 22, 80)
(5, 'Egor', 1, 20, 900)
(6, 'Даша', 2, 24, 1200)
q()-------------------------------------

deleted №2, №5:

q()-------------------------------------
USERS:

rowid name sex old score
(1, 'John', 2, 35, 35)
(3, 'Lena', 1, 18, 950)
(4, 'Misha', 1, 22, 80)
(6, 'Даша', 2, 24, 1200)
q()-------------------------------------


q()-------------------------------------
USERS:

name sex old score
('John', 2, 35, 35)
('Lena', 1, 18, 1450)
('Misha', 1, 22, 580)
('Даша', 2, 24, 1200)
q()-------------------------------------


q()-------------------------------------
USERS:

name sex old score
('John', 2, 35, 135)
('Lena', 1, 18, 1450)
('Misha', 1, 22, 680)
('Даша', 2, 24, 1200)
q()-------------------------------------


q()-------------------------------------
USERS:

name sex old score
('John', 2, 98, 468)
('Lena', 1, 18, 1450)
('Misha', 1, 22, 680)
('Даша', 2, 24, 1200)
q()-------------------------------------


q()-------------------------------------
GAMES:

user_id score time
(1, 100, 300)
(2, 30, 17)
(3, 90, 70)
(2, 203, 170)
(2, 10, 40)
(4, 104, 11)
(1, 99, 34)
q()-------------------------------------

считаем все user_id = 1:
2
---------- drawed ----------

считаем все уникальные (DISTINCT) user_id:
4
---------- drawed ----------

Выводим очки первого ID:
100
99
---------- drawed ----------

Выводим очки скрытого индекса rowid:
30
---------- drawed ----------

sum of all scores:
636
---------- drawed ----------

группировка по id и суммирование очков games:
2 203
1 100
3 90
---------- drawed ----------


q()-------------------------------------
GAMES:

user_id score time
(1, 100, 300)
(2, 30, 17)
(3, 90, 70)
(2, 203, 170)
(2, 10, 40)
(4, 104, 11)
(1, 99, 34)
q()-------------------------------------


q()-------------------------------------
USERS:

rowid name sex old score
(1, 'John', 2, 98, 468)
(3, 'Lena', 1, 18, 1450)
(4, 'Misha', 1, 22, 680)
(6, 'Даша', 2, 24, 1200)
q()-------------------------------------

John 2 100

Lena 1 90

Misha 1 104

John 2 99

---------- drawed ----------

объединение сыгранных игр и пользователей

John 2 100

Lena 1 100

Misha 1 100

Даша 2 100

John 2 30

Lena 1 30

Misha 1 30

Даша 2 30

John 2 90

Lena 1 90

Misha 1 90

Даша 2 90

John 2 203

Lena 1 203

Misha 1 203

Даша 2 203

John 2 10

Lena 1 10

Misha 1 10

Даша 2 10

John 2 104

Lena 1 104

Misha 1 104

Даша 2 104

John 2 99

Lena 1 99

Misha 1 99

Даша 2 99

---------- drawed ----------


таблица games с дополнениями из users:

John 2 100

None None 30

Lena 1 90

None None 203

None None 10

Misha 1 104

John 2 99

---------- drawed ----------

best players for all rounds:

John 2 199

Misha 1 104

Lena 1 90

---------- drawed ----------


concotinuation:

11 4 104

17 2 30

34 1 99

40 2 10

70 3 90

170 2 203

300 1 100

John 2 98

Lena 1 18

Misha 1 22

Даша 2 24

---------- drawed ----------


renamed concotinuation with column 'tbl':

203 table 1

104 table 1

100 table 1

99 table 1

98 table 2

90 table 1

30 table 1

24 table 2

22 table 2

18 table 2

10 table 1

---------- drawed ----------


q()-------------------------------------

USERS:

name sex old score lastname

('John', 2, 98, 468, None)

('Lena', 1, 18, 1450, None)

('Misha', 1, 22, 680, None)

('Даша', 2, 24, 1200, None)

q()-------------------------------------


q()-------------------------------------

USERS:

name sex old score lastname

('John', 2, 98, 468, None)

('Lena', 1, 18, 1450, None)

('Misha', 1, 22, 680, None)

('Даша', 2, 24, 1200, None)

('Egor', 1, 30, 4000, 'Mishuchkov')

q()-------------------------------------


Process finished with exit code 0
