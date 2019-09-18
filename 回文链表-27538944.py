# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        ptr0 = head
        ptr1 = head
        if not head:
            return True
        
        if ptr1.next and ptr1.next.next:
            ptr1 = ptr1.next.next
        else:
            if ptr1.next:
                return ptr1.val == ptr1.next.val
            else:
                return True
        
        while ptr1.next and ptr1.next.next:
            ptr1 = ptr1.next.next
            ptr0 = ptr0.next
        
        sec_head = ptr0.next.next
        
        if ptr1.next:
            # 1 2 3 4 5 6
            ptr0.next.next = None
        else:
            # 1 2 3 4 5 
            ptr0.next = None
        
        ptr = sec_head
        revert_head = None
        
        while ptr:
            tmp = ptr.next
            ptr.next = revert_head
            revert_head = ptr
            ptr = tmp
        
        while revert_head and head and revert_head.val == head.val:
            revert_head = revert_head.next
            head = head.next
        
        if not revert_head and not head:
            return True
        else:
            return False
                        
        
            
            
            
            
            