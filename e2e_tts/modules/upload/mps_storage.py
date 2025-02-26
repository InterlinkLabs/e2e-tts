import os
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '../')
import requests
import subprocess

try:
    from .config.mps_config import mps_config, tts_config
except:
    from config.mps_config import mps_config, tts_config


class DefaultMPS:
    def __init__(self):
        self.host = mps_config['host']
        self.namespace = mps_config['namespace']
        self.secret_key = mps_config['secret_key']
        self.file_host = mps_config['file_host']

    def ls(self, dir):
        API_ENDPOINT = 'http://' + self.host + \
            '/_/ls?secret_key=' + self.secret_key + '&dirpath=' + dir
        ls = requests.get(url=API_ENDPOINT).json()
        return ls

    def upload(self, file):
        path = file
        API_ENDPOINT = self.host + '/_/upload'
        query = 'curl {} ' \
                '-F convert=false ' \
                '-F filename={} ' \
                '-F secret_key={} ' \
                '-F filedata=@{} ' \
                '-F overwrite=true'.format(API_ENDPOINT,
                                           path, self.secret_key, file)

        res = subprocess.call(query, shell=True)
        return 'https://' + self.file_host + '/' + path


class ServiceMPS:
    def __init__(self):
        self.host = tts_config['host']
        self.namespace = tts_config['namespace']
        self.secret_key = tts_config['secret_key']
        self.file_host = tts_config['file_host']

    def ls(self, dir):
        is_exists = False
        path, fn = os.path.dirname(dir), os.path.basename(dir)

        API_ENDPOINT = 'http://' + self.host + \
            '/_/ls?secret_key=' + self.secret_key + '&dirpath=' + path
        ls = requests.get(url=API_ENDPOINT).json()

        if fn in ls['files']:
            is_exists = True

        if is_exists is False and ls['page'] != '1/1':
            total_pages = int(ls['page'].split('/')[-1])
            for i in range(1, total_pages):
                ls = requests.get(url=API_ENDPOINT + '&page=' + str(i + 1)).json()
                if fn in ls['files']:
                    is_exists = True
                    break

        return is_exists

    def upload(self, file):
        path = '/'.join(file.split('/')[2:])
        API_ENDPOINT = self.host + '/_/upload'
        query = 'curl {} ' \
                '-F convert=false ' \
                '-F filename={} ' \
                '-F secret_key={} ' \
                '-F filedata=@{} ' \
                '-F overwrite=true'.format(API_ENDPOINT,
                                           path, self.secret_key, file)

        subprocess.call(query, shell=True)

        return 'https://' + self.file_host + '/' + path

    def delete(self, file):
        API_ENDPOINT = self.host + '/_/delete?'
        query = 'curl -X DELETE {}' \
                'secret_key={}&filepath={}'.format(API_ENDPOINT, self.secret_key, file)

        print(query)
        res = subprocess.call(query, shell=True)

        return res
