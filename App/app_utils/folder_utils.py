import os
from pathlib import Path
from sys import path
import shutil
import requests

class FolderUtils:
    '''Utilities supporting files and folders'''

    def __init__(self):
        return None

    [staticmethod]
    def RemoveFilesOfType(dir, ext):
        files_in_directory = os.listdir(dir)
        filtered_files = [
            file for file in files_in_directory if file.endswith(ext)]
        for file in filtered_files:
            path_to_file = os.path.join(dir, file)
            os.remove(path_to_file)

    [staticmethod]
    def CopyFolders(src, dest):
        ''' This is a helper function to copy entire folders'''
        if os.path.exists(dest): 
            shutil.rmtree(dest) 
        shutil.copytree(src,dest)
        print(f'Copied folder {src} --> {dest}')


    [staticmethod]
    def GetFileFromWeb(url, dest):
        dir = os.path.dirname(dest)
        file_data = requests.get(url).content
        if not os.path.exists(dir): 
            os.mkdir(dir)

        with open(dest, 'wb') as handler:
            handler.write(file_data)