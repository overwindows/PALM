# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        lenA = 0
        _headA = headA
        while _headA:
            lenA += 1
            _headA = _headA.next
        
        lenB = 0
        _headB = headB
        while _headB:
            lenB += 1
            _headB = _headB.next
            
        if lenB > lenA:
            dlt = lenB-lenA
            #print(dlt)
            for i in range(dlt):
                headB = headB.next
        else:
            dlt = lenA - lenB
            for i in range(dlt):
                headA = headA.next
        
        while headA and headB:
            if headA == headB:
                return headA
            else:
                headA = headA.next
                headB = headB.next
        