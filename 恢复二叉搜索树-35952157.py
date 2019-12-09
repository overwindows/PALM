# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
import functools
class Solution:
    def recoverTree(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
    
        """
        
        def inOrder(root:TreeNode) -> List[TreeNode]:
            if not root.left and not root.right:
                return [root]
            
            ret = []
            
            if root.left:
                left_array = inOrder(root.left)
                ret.extend(left_array)
            
            ret.append(root)
            
            if root.right:
                right_array = inOrder(root.right)
                ret.extend(right_array)
            
            return ret
        
        ret0 = inOrder(root)
        ret1 = copy.deepcopy(ret0)
        
        def cmp(a,b):
            return a.val - b.val
        
        ret1.sort(key=functools.cmp_to_key(cmp))    
        
        for i in range(len(ret0)):
            #print(ret0[i].val,ret1[i].val)
            if ret0[i].val != ret1[i].val:
                ret0[i].val = ret1[i].val
        
        
            