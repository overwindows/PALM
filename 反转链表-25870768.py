# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reverse(self, head):

        if head.next == None:
            return (head, head)
        else:
            new_head, tail = self.reverse(head.next)
            tail.next = head
            head.next = None
            return new_head, head 
    
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head == None:
            return None
        new_head, _ = self.reverse(head)
        
        return new_head