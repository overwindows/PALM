# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        #print(preorder,inorder)
        if len(preorder) == 0:
            return None
        
        root = TreeNode(preorder[0])
        if len(preorder) == 1:
            return root
        
        split = 0
        
        for i in range(len(inorder)):
            if root.val == inorder[i]:
                split = i
                break       
        
        root.left = self.buildTree(preorder[1:split+1],inorder[:split])
        root.right = self.buildTree(preorder[split+1:],inorder[split+1:])
        
        return root
            