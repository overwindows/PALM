# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        p1 = head
        p2 = head
        pp1 = None
        
        for _ in range(n-1):
            p2 = p2.next
        
        while p2.next:
            pp1 = p1
            p1 = p1.next
            p2 = p2.next
        
        if pp1:
            pp1.next = p1.next
        elif n > 1: 
            head = head.next
        else:
            return None
        
        return head