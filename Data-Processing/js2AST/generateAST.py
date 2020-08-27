import esprima
import os
import json

def extractAST(dic, dic_list, flag):
    new_dic = {}
    new_list = []

    if type(dic) == dict:
        for value in dic.values():
            if type(value) == list or type(value) == dict:
                new_list.append(value)
        # print(new_list)

    if new_list != []:

        for key, value in dic.items():
            if key == 'type':
                new_dic['id'] = flag
                new_dic['type'] = value
                flag += 1
            if key == 'value' or key == 'name':
                new_dic['value'] = value

        children = []

        for child in new_list:
            if type(child) == dict:
                child_dic, dic_list, flag = extractAST(child, dic_list, flag)
                children.append(child_dic['id'])
            if type(child) == list:
                for i in range(len(child)):
                    child_dic, dic_list, flag = extractAST(child[i], dic_list, flag)
                    children.append(child_dic['id'])
        new_dic['children'] = children
        dic_list.append(new_dic)

        return new_dic, dic_list, flag

    else:
        for key,value in dic.items():
            if key == 'type':
                new_dic['id'] = flag
                new_dic['type'] = value
                flag += 1
            if key == 'value' or key == 'name':
                new_dic['value'] = value
        dic_list.append(new_dic)
        return new_dic, dic_list, flag






if __name__ == '__main__':
    path1 = 'dataset'
    files = os.listdir('dataset')
    for file in files:
        if not file.startswith('.'):
            f = os.listdir(path1+'/'+file)
            print(f)
    # f = open('test.js', 'r')
    # code_str = ''
    # for line in f.readlines():
    #     code_str += str(line)
    # print(code_str)
    # tree = esprima.parseScript(code_str)
    # tree_dic = tree.toDict()
    # print(tree)
    # ast_dic = []
    # flag = 0
    # dic, final_list, flag = extractAST(tree_dic, ast_dic, flag)
    # AST_list = sorted(final_list, key=lambda e: e.__getitem__('id'), reverse=False)
    # f1 = open('AST.json', 'a')
    # AST_list_json = json.dumps(AST_list)
    # f1.write(AST_list_json)
    # f1.write('\n')
    # f1.close()
    # print(AST_list_json)
    # print(type(AST_list_json))
