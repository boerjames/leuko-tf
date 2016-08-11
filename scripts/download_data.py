# This file is used to download images from the facetag database
# Usage: python download_data.py --path PATH --password PASSWORD

# todo get connected to facetag and make it work, use old torch code as inspiration
# todo use safe queries

import argparse

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path',  nargs=1, required=True,     help='where to store the images')
    parser.add_argument('--password',   nargs=1, required=True,     help='the database password')

    args = vars(parser.parse_args())
    _path = args['save_path'][0]
    _password = args['password'][0]
    _host = 'facetag-db'
    _dbname = 'facetag'
    _user = 'facetag'

    return

if __name__ == '__main__':
    run()