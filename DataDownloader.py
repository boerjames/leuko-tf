import psycopg2
import argparse

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', nargs=1, help='where to store the images', required=True)
    parser.add_argument('--password', nargs=1, help='the database password', required=True)

    args = vars(parser.parse_args())
    _path = args['path'][0]
    _password = args['password'][0]
    _host = 'facetag-db'
    _dbname = 'facetag'
    _user = 'facetag'

    return

if __name__ == '__main__':
    run()