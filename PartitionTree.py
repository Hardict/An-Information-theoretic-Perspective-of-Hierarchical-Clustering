class PartitionTreeNode:
    """
    param:
    @parent:树节点父亲->PartionTreeNode
    @children:节点孩子集合->set
    @volume:树节点叶子构成子图的体积->double
    @g_val:树节点叶子构成子图向外连边权值和->double
    @cut_val:树节点孩子构成划分的cut值->double
    @height:树节点叶子高度(自身到孩子的距离)->int，默认值为1
    @node_set:节点对应图节点集合(key为节点id，value为邻接表)->dict
    @origin_node_set:node_set会还有中间超节点，该set存储原始图的节点
    """

    def __init__(self, _id: int = -1, _parent=None, _children: set = None, _volume: float = 0, _g_val: float = 0,
                 _cut_val: float = 0,
                 _height: int = 1, _node_set: dict = None, _origin_node_set: set = None, _origin_children: set = None):
        self.id = _id
        self.parent: PartitionTreeNode = _parent
        self.children: Set[PartitionTreeNode] = _children
        self.volume = _volume
        self.g_val = _g_val
        self.cut_val = _cut_val
        self.height = _height
        self.node_set = _node_set
        self.origin_node_set = _origin_node_set
        self.origin_children = _origin_children

    def __str__(self):
        # return "{" + "{}:{}".format(self.__class__.__name__, ",".join("{}={}"
        #                                                               .format(k, getattr(self, k))
        #                                                               for k in self.__dict__.keys())) + "}"
        str = "{"
        str += self.__class__.__name__ + ":"
        str += ", id={}".format(self.id)
        pid = -1
        if self.parent is not None:
            pid = self.parent.id
        str += ", parent={}".format(pid)
        str += ", height={}".format(self.height)
        if self.children is not None:
            str += ", children_size={}".format(len(list(ch.id for ch in self.children)))
            str += ", children={}".format(list(ch.id for ch in self.children))
        else:
            str += ", children={None}"
        if self.origin_children is not None:
            str += ", origin_children_size={}".format(len(list(ch.id for ch in self.origin_children)))
            str += ", origin_children={}".format(list(ch.id for ch in self.origin_children))
        else:
            str += ", origin_children={None}"
        str += ", volume={}".format(self.volume)
        str += ", origin_node_set={}".format(sorted(list(self.origin_node_set)))
        str += "}"
        return str

    def __lt__(self, other):
        """
        优先队列比较(确保唯一性即可)
        """
        return self.id < other.id
