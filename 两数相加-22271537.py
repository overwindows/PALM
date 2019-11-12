# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        is_plus_one = False
        _sum_reverse = ListNode(0)
        _pivot = _sum_reverse
        
        while l1 and l2:
            
            _sum_reverse.val += l1.val + l2.val
            
            if _sum_reverse.val >= 10:
                is_plus_one = True
                _sum_reverse.val -= 10
            
            l1 = l1.next
            l2 = l2.next
            
            if l1 and l2:    
                _sum_reverse.next = ListNode(0)                    
                _sum_reverse = _sum_reverse.next
                if is_plus_one:
                    _sum_reverse.val = 1
                    is_plus_one = False
            else:
                break
        
        if is_plus_one:
            _sum_reverse.next = ListNode(1)
            _sum_reverse = _sum_reverse.next
            
        while l1:
            if not is_plus_one:
                if _sum_reverse.next is None:
                    _sum_reverse.next =  ListNode(0)
                _sum_reverse = _sum_reverse.next
            else:
                is_plus_one = False
                
            _sum_reverse.val += l1.val
            if _sum_reverse.val >= 10:
                _sum_reverse.val -= 10
                _sum_reverse.next =  ListNode(1)
                #_sum_reverse = _sum_reverse.next
            l1 = l1.next
        
        while l2:
            if not is_plus_one:
                if _sum_reverse.next is None:
                    _sum_reverse.next =  ListNode(0)
                _sum_reverse = _sum_reverse.next
            else:
                is_plus_one = False
                
            _sum_reverse.val += l2.val
            if _sum_reverse.val >= 10:
                _sum_reverse.val -= 10
                _sum_reverse.next =  ListNode(1)
                #_sum_reverse = _sum_reverse.next
            l2 = l2.next
        
        _sum = []
        
        while _pivot:
            _sum.append( _pivot.val)
            _pivot = _pivot.next
            
        return _sum
            
            