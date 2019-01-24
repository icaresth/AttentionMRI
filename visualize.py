from graphviz import Digraph
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

def make_dot(var, params=None):
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2'
                     )

    dot = Digraph(node_attr=node_attr, graph_attr=dict(size='12,12'))

    seen = set()

    def size_to_str(size):
        return '(' + (',').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n%s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot


# display
cpu = torch.device('cpu')
def cvt2numpy(tensor):
    t = tensor.to(cpu)
    t = t.numpy()
    t = t[:,:,:][0]
    return t

def display(data):
    data = cvt2numpy(data)
    c, x, y= data.shape
    for i in range(c):
        img = data[i, :, :]
        # img = data[i, x//2, :, :]
        # path = cae3d.save_path + '_' + name + '_channel{}'.format(c) + '.png'
        plt.imshow(img, cmap='gray')

        # plt.savefig(path)
        plt.show()

