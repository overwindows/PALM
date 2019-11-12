# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        lead = head
        follow = head
        ptr = head
        n = 0
        while ptr:
            ptr = ptr.next
            n+=1
        if n == 0:
            return None
        if n == 1:
            return head
        k = k%n
        if k == 0:
            return head
        for _ in range(k-1):
            follow  = follow.next
        
        while follow.next.next:
            follow = follow.next
            lead = lead.next
        
        new_head = lead.next
        lead.next = None
        
        if new_head.next == None:
            new_head.next = head    
        else:
            follow.next.next = head

        return new_head
        
        
        
        
            