import graphviz
import extractInfo

def Visualize(name):
    prefix = 'GraphRNN_RNN_AST_4_128_'
    format = '.dat'
    file = prefix+str(name)+format
    node_list, edge_list, node_type_list = extractInfo.extract(file)

    dot = graphviz.Graph(comment='Result')

    # Add node
    flag = 0
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
    Visualize('train_0')
