# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        ptr1 = head
        if head == None or head.next == None or head.next.next == None:
            return None
        ptr2 = head.next.next
        
        while ptr1 != ptr2:
            if ptr1.next:
                ptr1 = ptr1.next
            else:
                return None
            
            if ptr2.next and ptr2.next.next:
                ptr2 = ptr2.next.next
            else:
                return None
        
        while head:
            #print(head.val)
            ptr = ptr1
            ptr = ptr.next
            while ptr != ptr1:
                if ptr == head:
                    return head
                else:
                    ptr = ptr.next
            if ptr == head:
                return head
            head = head.next    
        
        return None
        
        
        