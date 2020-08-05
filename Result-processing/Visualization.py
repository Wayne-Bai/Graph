import graphviz
import extractInfo

def Visualize(name, num, total_feature):
    # path = 'graphs15/'
    prefix = 'GraphRNN_RNN_AST50_4_128_'
    # prefix = 'GraphRNN_RNN_AST_4_128_'
    format = '.dat'
    # file = path+prefix+str(name)+format
    file = prefix+str(name)+format
    node_list, edge_list, node_type_list = extractInfo.extract(file, num, total_feature)

    dot = graphviz.Graph(comment='Result')

    # Add node
    flag = 0
    print(len(node_list))
    print(len(node_type_list))
    for i in node_list:
        node_id = i
        node = node_type_list[flag]
        print(node_id, node)
        node_p = '[{}]'.format((node_id)) +': ' + node
        dot.node(str(node_id), node_p)
        flag += 1

    for i in edge_list:
        m, n = i[0], i[1]
        dot.edge(str(m), str(n))

    dot.render("process/test-output/%s.gv" %(file), view=True)

if __name__ == '__main__':
    # for i in range(31):
    #     flag = 5000 + i*100
    #     name = 'pred_%d_1'%(flag)
    #     Visualize(name, 0, 28)
    Visualize('pred_5000_1',15,28)

