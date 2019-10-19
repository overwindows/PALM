# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def generateTrees(self, n: int) -> List[TreeNode]:
        
        def _genTrees(start,end) -> List[TreeNode]:
            #print(start,end)
            if start == end:
                return [TreeNode(start)]
            ret = []
            
            for i in range(start,end+1):
                if i+1 <= end:
                    right = _genTrees(i+1, end)
                else:
                    right = None
                if i-1 >= start:
                    left = _genTrees(start, i-1)
                else:
                    left = None
                node = TreeNode(i)
                
                if not left:
                    for r in right:
                        #_node = node
                        _node = TreeNode(node.val)
                        #assert _node.val < r.val
                        _node.right = r
                        ret.append(_node)
                elif not right:
                    for l in left:
                        #_node = node
                        _node = TreeNode(node.val)
                        #assert _node.val > l.val, (_node.val,l.val)
                        _node.left = l
                        ret.append(_node)
                else:
                    assert left and right
                    for l in left:
                        for r in right:
                            _node = TreeNode(node.val)
                             #_node = node
                            _node.left = l
                            _node.right = r
                            ret.append(_node)
            #print(ret)
            return ret
        
        res = _genTrees(1,n)
        return res