# Class for managing training results
import sqlite3
import csv

class ResultsManager:

    ### Constructor and Destructor
    def __init__(self, database_name, battery_name):
        self.__database_name = database_name
        self.__battery_name = battery_name
        self.__connection = sqlite3.connect(self.__database_name)

    def __del__(self):
        self.__connection.close()


    ### Database methods
    def setup(self):
        cursor = self.__connection.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS
                        {} (id INTEGER PRIMARY KEY, train_acc REAL, test_acc REAL);'''.format(self.__battery_name))
        self.__connection.commit()

    def add(self, train_acc, test_acc):
        cursor = self.__connection.cursor()
        cursor.execute('''INSERT INTO
                        {} (id, train_acc, test_acc)
                        VALUES (NULL, {}, {});'''.format(self.__battery_name, train_acc, test_acc))
        self.__connection.commit()

    def export(self):
        cursor = self.__connection.cursor()

        results_list = []
        for row in cursor.execute('''SELECT * FROM {} ORDER BY test_acc DESC;'''.format(self.__battery_name)):
            results_list.append(row)

        with open('results.csv', 'w') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONE)
            writer.writerow(["ID", "TRAIN ACCURACY", "TEST ACCURACY"])
            writer.writerows(results_list)


    ### Getters
    def get_database_name(self):
        return self.__database_name

    def get_battery_name(self):
        return self.__battery_name