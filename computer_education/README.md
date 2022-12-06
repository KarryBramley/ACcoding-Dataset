# Computer Education Dataset

This repository contains all materials related to the **computer education dataset** project.




## Online Judge Platforms
**Differences beween beihang online judge platforms and others**
- Most online judge platforms are designed for ACM competitors(i.e. codeforces, POJ, HDU, etc) or job hunters(i.e. leetcode). 
- Most online judge platforms don't provide open-source datasets for computer education research. Although codeforces platform has provide a dataset called Code4Bench, Code4Bench is a benchmark dataset which is only utilized for reproducible research, such as program analysis and software testing.

## Why construct graph data set?
- **Application perspective**: Incorporating such a graph-structured nature to the knowledge tracing model as a relational inductive bias can improve performance.
- **Data Structure perspective**: Use knowledge graph structure can represent potential non-Euclidean structure in the data set.

## Educational Datasets
Here we provide some educational datasets.

**Programming Data sets**: 

- **Hour of Code Data Set**:　[\[Description\]](https://code.org/research)　[\[Download\]](https://web.archive.org/web/20150503235202/https://code.org/files/anonymizeddata.zip)  
  Note that the data set download link in the description webpage has been expired. So we request the author for a new download link.

  **Difference between this data set and our data set**:  
  The problems are all elementary programming problems about maze puzzles. Furthermore, problem codes in this data set are only block-based code fragments which contain simple program operation blocks, such as for-loop block, if-else block, etc.
  
  Papers using this data set:  
  - [Learning to Represent Student Knowledge on Programming Exercises Using Deep Learning](http://educationaldatamining.org/EDM2017/proc_files/papers/paper_129.pdf)
  - [Deep Knowledge Tracing on Programming Exercises](https://dl.acm.org/doi/10.1145/3051457.3053985)
  - [Autonomously Generating Hints by Inferring Problem Solving Policies](https://web.stanford.edu/~cpiech/bio/papers/inferringProblemSolvingPolicies.pdf)
  - [Zero Shot Learning for Code Education Rubric Sampling with Deep Learning Inference](https://arxiv.org/pdf/1809.01357.pdf)

- **Code4Bench Data Set**:　[\[Paper\]](https://www.sciencedirect.com/science/article/pii/S1045926X18302489)　[\[Download\]](https://zenodo.org/record/2582968/files/code4bench.rar)　[\[Github\]](https://github.com/code4bench/Code4Bench)  
  xxxxx

- **CodeChef Data Set**:　[\[Description\]](https://www.kaggle.com/arjoonn/codechef-competitive-programming)　[\[Download\]](https://www.kaggle.com/arjoonn/codechef-competitive-programming/download)  
  It contains about 1000 problem statements and a little over 1 million solutions in total to these problems in various languages, which is used for **code generation learning**.
  

**Other subjects Data sets**:
- [ASSISTments Data Set](https://sites.google.com/site/assistmentsdata/)
- [AICFE Data Set](http://www.bnu-ai.cn/data)



## Useful Dataset Website

- [DataShop, the world's largest repository of learning interaction data](https://pslcdatashop.web.cmu.edu/index.jsp?datasets=public)
- [Aminer Science Knowledge Graph Data Set](https://www.aminer.org/scikg)  
- [Microsoft Concept Graph For Short Text Understanding](https://concept.research.microsoft.com/Home/Introduction)


## Educational Data Mining Applications
Our project focus on two educational data mining applications, **knowledge tracing** and **educational recommendation**.

### Knowledge Tracing
Here we provide some knowledge tracing methods used in our project.

- **Bayesian Knowledge Tracing(BKT)**:　[\[Paper\]](http://act-r.psy.cmu.edu/wordpress/wp-content/uploads/2012/12/893CorbettAnderson1995.pdf)　[\[Code\]](https://github.com/CAHLR/pyBKT)
- **Deep Knowledge Tracing(DKT)**:　[\[Paper\]](https://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf)　[\[Lua Code\]](https://github.com/chrispiech/DeepKnowledgeTracing)　[\[Python Code\]](https://github.com/lccasagrande/Deep-Knowledge-Tracing)
- **Dynamic Key-Value Memory Networks(DKVMN)**:　[\[Paper\]](https://arxiv.org/pdf/1611.08108.pdf)　[\[Code\]](https://github.com/jennyzhang0215/DKVMN)
- **Graph-based Knowledge Tracing(GKT)**:　[\[Paper\]](https://dl.acm.org/doi/10.1145/3350546.3352513)　[\[Code\]](xxxx)

### Educational Recommendation
Here we provide some educational recommendation methods used in our project.
