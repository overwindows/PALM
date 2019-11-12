# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        h = []
        #init
        for i in range(len(lists)):
            if lists[i]:
                heapq.heappush(h,(lists[i].val,i))
        if not h:
            return None
        _, x = heapq.heappop(h)
        head = lists[x]
        lists[x] = lists[x].next
        if lists[x]:
            heapq.heappush(h,(lists[x].val,x))
        
        ptr = head
        while h:
            _, k = heapq.heappop(h)
            ptr.next = lists[k]
            lists[k] = lists[k].next
            if lists[k]:
                heapq.heappush(h,(lists[k].val,k))
    
            ptr = ptr.next
            ptr.next = None
            
        return head
            
            
            
        
        