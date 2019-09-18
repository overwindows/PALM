class Solution:  
    def reconstructQueue(self, people):
        def queue_cmp(x,y):
            if x[0] > y[0]:
                return -1
            if x[0] == y[0]:
                if x[1] > y[1]:
                    return 1
                else:
                    return -1
            return 1
        people.sort(cmp=queue_cmp)
        re_people = []
        #print(people)
        for h,k in people:
            re_people.insert(k,[h,k])            
            
        return re_people