# Data

This directory contains data used for the **computer education** project.


- **Daily Submission Data** (ranging from 2015-10-8 to 2020-4-14)

    **File**：`results_daily.csv`  
    **Attributes**：time, uid, pid, cnt  
    **Item Number**：277799  

- **Contest Submission Data** (ranging from 2015-10-8 to 2020-4-14)

    **File**：`results_contest.csv`    
    **Attributes**：time, uid, pid, cid, cnt    
    **Item Number**：821258    

- **Daily Feedback Data** (ranging from 2015-10-8 to 2020-4-14)   

    **File**：results_daily.csv    
    **Attributes**：time, uid, pid, detail    
    **Item Number**：277799    

- **Contest Feedback Data** (ranging from 2015-10-8 to 2020-4-14)  

    **File**：`results_contest.csv`  
    **Attributes**：time, uid, pid, cid, detail( [(timestamp，result id), ......] )  
    **Item Number**：821259  

- **Problem & Knowledge Point Data** (Not all problems have knowledge points)   

    **File**：`problem_knowledgePoints.csv`  
    **Attributes**：pid, kid  
    **Item Number**：2706  

- **Problem & Contest Data**

    **File**：`problem_contest.csv`  
    **Attributes**：pid, cid  
    **Item Number**：2984  

- **Feedback Status Data**

    **File**：`results.csv`  
    **Attributes**：rid, rname  
    **Item Number**：12  

- **Knowledge Point Data**  

    **File**：`knowledgePoints.csv`  
    **Attributes**：kid, kname  
    **Item Number**：100  

- **User Profile Data** (Users whose submission number is zero are eliminated) 

    **File**：`users.csv`  
    **Attributes**：uid, grade(ended by 'y' means graduate students), cid(college)  
    **Item Number**：16532  


- **Problem Data**（Not all problems have 'difficulty' attribute） 

    **File**：`problems.csv`  
    **Attributes**：pid, pname, difficulty, creator_id  
    **Item Number**：2923  

- **Contest Data**  

    **File**：`contest.csv`  
    **Attributes**：cid, stime(start time), etime(end time)  
    **Item Number**：398  
