# Class for managing training results

import sqlite3
import csv

class ResultsManager(object):

    # Constructor and Destructor
    def __init__(self, database_name, battery_name):
        self.database_name = database_name
        self.battery_name = battery_name
        self.connection = sqlite3.connect(self.database_name)

        cursor = self.connection.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS
                        {} (id INTEGER PRIMARY KEY, train_acc REAL, test_acc REAL);'''.format(self.battery_name))
        self.connection.commit()

    def __del__(self):
        self.connection.close()

    # Add training results to the database
    # todo include network and training parameters
    def add(self, train_acc, test_acc):
        cursor = self.connection.cursor()
        cursor.execute('''INSERT INTO
                        {} (id, train_acc, test_acc)
                        VALUES (NULL, {}, {});'''.format(self.battery_name, train_acc, test_acc))
        self.connection.commit()

    # Export database as csv
    def export(self):
        cursor = self.connection.cursor()

        results_list = []
        for row in cursor.execute('''SELECT * FROM {} ORDER BY test_acc DESC;'''.format(self.battery_name)):
            results_list.append(row)

        file_name = '{}-{}.csv'.format(self.database_name, self.battery_name)
        with open(file_name, 'w') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONE)
            writer.writerow(["ID", "TRAIN ACCURACY", "TEST ACCURACY"])
            writer.writerows(results_list)
