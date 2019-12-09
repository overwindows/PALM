# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        if head == None:
            return head
        if head.next == None:
            return head
        if head.next.next == None:
            ptr1 = head
            ptr2 = head.next
            if ptr1.val <= ptr2.val:
                return head
            else:
                ptr2.next = ptr1
                ptr1.next = None
                return ptr2
                
        mid = head
        tail = head
        while tail.next and tail.next.next:
            mid = mid.next
            tail = tail.next.next
        head2 = mid.next
        mid.next = None
        
        sort_head1 = self.sortList(head)
        sort_head2 = self.sortList(head2)
        
        sort_head = None
        ptr = None
        while sort_head1 and sort_head2:
            if sort_head1.val > sort_head2.val:
                if sort_head == None:
                    sort_head = sort_head2
                    ptr = sort_head
                else:
                    ptr.next = sort_head2
                    ptr = ptr.next
                sort_head2 = sort_head2.next
            else:
                if sort_head == None:
                    sort_head = sort_head1
                    ptr = sort_head
                else:
                    ptr.next = sort_head1
                    ptr = ptr.next
                sort_head1 = sort_head1.next
                
        while sort_head1:
            ptr.next = sort_head1
            ptr = ptr.next
            sort_head1 = sort_head1.next
        while sort_head2:
            ptr.next = sort_head2
            ptr = ptr.next
            sort_head2 = sort_head2.next
            
        return sort_head
            
        
        
            
            
        