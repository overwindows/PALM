# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if l1 is None:
            return l2
        
        if l2 is None:
            return l1
        
        head = None
        ptr = None
        while l1 and l2:
            if head == None:
                if l1.val < l2.val:
                    head = l1
                    ptr = l1
                    l1 = l1.next
                else:
                    head = l2
                    ptr = l2
                    l2 = l2.next
            else:
                if l1.val < l2.val:
                    ptr.next = l1
                    l1 = l1.next
                else:
                    ptr.next = l2
                    l2 = l2.next
                ptr = ptr.next
        
        while l1:
            ptr.next = l1
            l1 = l1.next
            ptr = ptr.next
        
        while l2:
            ptr.next = l2
            l2 = l2.next
            ptr = ptr.next  
        
        ptr = None
        
        return head
        