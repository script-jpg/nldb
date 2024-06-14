from .data_loader import *
from .transformer import *

from tqdm import tqdm
import json
import os



class DataUtils:

    def get_default():
        return DataUtils(loaders=[MarkdownDataLoader()], transformer=BertTransformer())

    def __init__(self,
                 loaders: list[DataLoader] = [],
                 transformer: Transformer = BertTransformer()) -> None:
        self.loaders = loaders
        self.transformer = transformer


    def load(self, dir: str = './Documents/utils.json', default = None):
        # Load data from the utils file, create the file if it does not exist

        if default is None:
            default = dict()
    
        # create a default utils file if the file is created yet
        if not os.path.exists(dir):
            with open(dir, 'w') as utils_file:
                utils_file.write(json.dumps(default, indent=4))

        with open(dir, 'r') as utils_file:
            self.utils = json.loads(utils_file.read())


    def update(self, document_folder_dir: str = './Documents'):
        # Update self.utils and save the json file,
        #   call after a call to self.load()
        self.document_folder_dir = document_folder_dir

        def update_file(file_dir, loader: DataLoader):
            # updata utils for a file, return a structure
            #   {'mtime': str, 'content': list[[str, list[float]]]}
            mtime = os.path.getmtime(file_dir)
            if file_dir not in self.utils or\
                mtime != self.utils[file_dir]['mtime']:
                print(f'updating {file_dir} ...')
                file_content = [[line, self.transformer.encoded(line).tolist()]
                                for line in tqdm(loader.load_file(file_dir))]
                self.utils[file_dir] = {'mtime': mtime, 'content': file_content}


        # store list of files in dir
        files = os.listdir(document_folder_dir)
        
        # sort all files in dir by type
        sorted_files = dict()
        for file in files:
            file_type = file.split('.')[-1]
            if file_type not in sorted_files.keys():
                sorted_files[file_type] = list()
            sorted_files[file_type].append(file)
        
        for loader in self.loaders:
            file_type = loader.file_type()
            if file_type in sorted_files.keys():
                for file in sorted_files[file_type]:
                    update_file(f'{document_folder_dir}/{file}', loader)

        # find files in self.utils not in sorted files and delete them
        files_in_dir = {f"{document_folder_dir}/{file}" for file_list in sorted_files.values() for file in file_list}
        files_in_utils = self.utils.keys()

        for file in files_in_utils - files_in_dir:
            del self.utils[file]

    

    def save(self, dir: str = './Documents', default = dict()):
        # Save self.utils

        if default is None:
            default = dict()
        
        if self.utils is None:
            self.utils = default
        with open(dir, 'w') as utils_file:
            utils_file.write(json.dumps(self.utils, indent=4))

    def find_relevant(self, query: str):
        query_is_null_or_whitespace = query is None or (isinstance(query, str) and query.strip() == '')
        if query_is_null_or_whitespace:
            raise ValueError("query is either null or whitespace. Please give a valid query.")

        relevant_doc_names = self.transformer.find_relevant(query, self.utils)
        relevant_doc_data = []
        for doc in relevant_doc_names:
            with open(doc, 'r') as doc_data:
                file_content = doc_data.read()
                relevant_doc_data.append(file_content)
        return relevant_doc_data
   
if __name__ == "__main__":
    utils = DataUtils.get_default()
    utils.load('./Documents/utils.json', dict())
    utils.update('./Documents')
    utils.save('./Documents/utils.json')
