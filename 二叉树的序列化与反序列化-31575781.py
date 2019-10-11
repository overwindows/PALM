# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:
    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        BFS = []
        ret = []
        
        if root:
            BFS.append(root)
        else:
            return None
                
        while BFS:
            node = BFS.pop(0)
            if node:
                ret.append(node.val)
            else:
                ret.append(None)

            if node:
                BFS.append(node.left)
                BFS.append(node.right)
        
        while ret[-1] == None:
            ret.pop()
        #print(ret)
        return ret
        
    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        #print(data)
        if not data:
            return None
        lens = len(data)
        if lens == 1:
            return TreeNode(data[0])
        #n = len(data)+1
        #depth = 0
        
        #while n>1:
        #    depth +=1
        #    n = n//2
        
        root = TreeNode(data[0])
        Node = []
        Node.append(root)
        ix = 1
        
        while True:
            node = Node.pop(0)
            if node: 
                if data[ix] is not None:
                    node.left = TreeNode(data[ix])
                else:    
                    node.left = None    
                Node.append(node.left)
                ix += 1
                if ix >= lens:
                    break
                
                if data[ix] is not None:
                    node.right = TreeNode(data[ix])
                else:
                    node.right = None
                ix += 1
                if ix >= lens:
                    break
                Node.append(node.right)
            
            
            
        
        '''
        n = len(data)+1
        for i in range(n//2-1):
            if len(Node) < 1:
                break
            node = Node.pop(0)
            #print(node)
            if node == None:
                Node.append(None)
                Node.append(None)
                continue
            #print(data[i],data[2*i+1],data[2*(i+1)])
            if data[2*i+1] is None:
                node.left = None
            else:
                node.left = TreeNode(data[2*i+1])
            
            if data[2*(i+1)] is None:
                node.right = None
            else:
                node.right = TreeNode(data[2*(i+1)])
            
            Node.append(node.left)
            Node.append(node.right)
        '''
        return root
        

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))