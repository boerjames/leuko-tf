# Class for managing training results

import sqlite3
import csv
import os

class ResultsManager(object):

    # Constructor and Destructor
    def __init__(self, database_name, battery_name, save_path):
        self.database_name = database_name
        self.battery_name = battery_name
        self.save_path = save_path
        self.connection = sqlite3.connect(self.database_name)

        cursor = self.connection.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS
                        {} (exp_id INTEGER, train_acc REAL, test_acc REAL);'''.format(self.battery_name))
        self.connection.commit()

    def __del__(self):
        self.connection.close()

    # Add training results to the database
    # todo include network and training parameters
    def add(self, exp_id, train_acc, test_acc):
        cursor = self.connection.cursor()
        cursor.execute('''INSERT INTO
                        {} (exp_id, train_acc, test_acc)
                        VALUES ({}, {}, {});'''.format(self.battery_name, exp_id, train_acc, test_acc))
        self.connection.commit()

    # Export database as csv
    def export(self):
        cursor = self.connection.cursor()

        results_list = []
        for row in cursor.execute('''SELECT * FROM {} ORDER BY test_acc DESC;'''.format(self.battery_name)):
            results_list.append(row)

        # prepare save data path
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        file_name = os.path.join(self.save_path, '{}-{}.csv'.format(self.database_name, self.battery_name))
        with open(file_name, 'w') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONE)
            writer.writerow(["EXPERIMENT ID", "TRAIN ACCURACY", "TEST ACCURACY"])
            writer.writerows(results_list)
