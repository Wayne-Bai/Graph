import graphviz
import extractInfo

def Visualize(name, num, total_feature):
    prefix = 'GraphRNN_RNN_AST_4_128_'
    format = '.dat'
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
        dot.node(str(node_id), node)
        flag += 1

    for i in edge_list:
        m, n = i[0], i[1]
        dot.edge(str(m), str(n))

    dot.render("test-output/%s.gv" %(file), view=True)

if __name__ == '__main__':
    Visualize('test_0', 0, 18)
