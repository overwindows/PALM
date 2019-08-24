# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
      if not root:
        return True
      
      queue = []
      queue.append(root)
      queue.append('SEP')
      check = []
      
      while queue:
        # check queue
        node = queue.pop(0)
        if node == 'SEP':
            #print(check,len(check))
            if len(check)%2:
              return False
            else:
              n = len(check)
              for i in range(n//2):
                if check[i] != check[n-i-1]:
                  #print(i, n-i-1)
                  return False          
            if queue:
              queue.append('SEP')
            check.clear()
        else:
            if node:
              if node.left:
                queue.append(node.left)
                check.append(node.left.val)
              else:
                check.append(None)
              if node.right:
                queue.append(node.right)
                check.append(node.right.val)
              else:
                check.append(None)
      return True