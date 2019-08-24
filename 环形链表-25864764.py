# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        step1 = head
        if head and head.next and head.next.next:
            step2 = head.next.next
        else:
            return False
        
        while step1.next and step2.next and step2.next.next:
            if step1 == step2:
                return True
            else:
                step1 = step1.next
                step2 = step2.next.next
        return False
        
        