import os
import re

PATH_DATA = './output'
alphabet = '^[ _abcdeghiklmnopqrstuvxy0123456789áàảãạâấầẩẫậăắằẳẵặóòỏõọôốồổỗộơớờởỡợéèẻẽẹêếềểễệúùủũụưứừửữựíìỉĩịýỳỷỹỵđ!\"\',\-\.:;?_\(\)]+$'

folders = os.listdir(PATH_DATA)
# print(folders)
for folder in folders:
    path_folder = os.path.join(PATH_DATA, folder)
    # print(path_folder)
    list_files = os.listdir(path_folder)
    for file in list_files:
        with open(os.path.join(path_folder, file), 'r') as f:
            contents = f.read()
            contents = re.sub("(\s)+", r"\1", contents)
            contents = contents.split("\n")
            for content in contents:
                try:
                    content = eval(content)
                except:
                    continue
                lines = content['text'].split('\n')
                with open("train_data.txt", 'a') as f_w:
                    for line in lines[1:]:
                        if len(line.split(' ')) > 3 and re.match(alphabet, line.lower()):
                            f_w.write(line + '\n')

with open('train_data.txt') as f:
    lines = f.read().split('\n')

