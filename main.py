# The program entry point
from ResultsManager import *

def main():
    results = ResultsManager('results.db', 'experiment_name')
    print("dbname = {}, batname = {}".format(results.get_database_name(), results.get_battery_name()))
    results.setup()

    for i in range(10):
        for j in range(10):
            results.add(i, j)

    results.export()

    del results

if __name__ == '__main__':
    main()