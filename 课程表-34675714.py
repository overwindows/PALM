class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        course_all = set()
        course_dep = {}
        
        if numCourses == 1:
            return True
        
        if len(prerequisites) == 0:
            return True
        
        for c, pre_c in prerequisites:
            if c == course_dep:
                return False
            
            if c in course_dep:
                course_dep[c].add(pre_c)
            else:
                course_dep[c] = set()
                course_dep[c].add(pre_c)
        
            course_all.add(c)
            course_all.add(pre_c)
        
        dep_keys = course_dep.keys()
        course_indep = course_all - dep_keys
        
        if len(course_indep) == 0:
            return False
        
        #print(course_all, dep_keys, course_indep)
        while course_indep:
            t = course_indep.pop()
            tbd = []
            for k,v in course_dep.items():
                if t in v:
                    v.remove(t)
                    if not v:
                        course_indep.add(k)
                        tbd.append(k)
                        #del course_dep[k]
            for k in tbd:
                del course_dep[k]
            '''
            while course_dep:
                k,v = course_dep.popitem()
                if t in v:
                    v.remove(t)
                    if len(v) == 0:
                        course_indep.add(k)
                    else:
                        course_dep[k] = v
                else:
                    course_dep[k] = v
            '''      
        if course_dep:
            return False
        else:
            return True
        
        #return False
            