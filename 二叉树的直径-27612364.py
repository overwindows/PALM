# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def _diameterOfBinaryTree(self, root):
        if not root:
            return 0,0
        
        if not root.left and not root.right:
            #print(root.val)
            return 0,0
        
        lf_max_path = 0
        lf_max_edge = 0
        rt_max_path = 0
        rt_max_edge = 0
        
        cnt = 0
        
        if root.left:
            lf_max_path, lf_max_edge = self._diameterOfBinaryTree(root.left)
            #print(lf_max_path,lf_max_edge,root.left.val)
            cnt +=1
        
        if root.right:
            rt_max_path, rt_max_edge = self._diameterOfBinaryTree(root.right)
            #print(rt_max_path,rt_max_edge,root.right.val)
            cnt += 1
        
        return max(rt_max_path, lf_max_path, lf_max_edge+rt_max_edge+cnt), max(lf_max_edge, rt_max_edge)+1
        
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        max_path, _ = self._diameterOfBinaryTree(root)
        return max_path
         