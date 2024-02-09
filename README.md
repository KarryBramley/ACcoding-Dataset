### ACcoding: A graph-based dataset for online judge programming

### Dataset

The dataset, named ACcoding, was built by a university teaching group.  The latest release of ACcoding dataset is available at https://zenodo.org/record/6522395. By the time of constructing the dataset (May 6, 2022), the dataset contains 4,046,652 task-solving records submitted by 27,444 students on 4,559 programming tasks over a span of 6 years. The large size of the dataset, combined with its rich functional features, empowers educators to trace students’programming progress and choose appropriate programming tasks for specific training purposes. 

### Load Dataset

You can download the ACcoding dataset from https://zenodo.org/record/6522395, which includes 6 SQL files. Assuming that you have already created a MySQL database, you can load the data files by executing the command 

```bash
source xxx.sql
```

 in MySQL. Alternatively, you can use tools like Navicat or DataGrip to load the SQL files. 

### Data Format

ACcoding dataset is constructed and stored in a relational database in the form of tables. Note that in the table below, names in lower cases are entities, and those in upper cases are relations between the entities.

|     Name      |             Description             |               Attributes               | Quantity |
| :-----------: | :---------------------------------: | :------------------------------------: | :------: |
|     Users     |         undergraduate users         |             id, last_login             |  27444   |
|   Problems    |          programming tasks          |  difficulty, time_limit, memory_limit  |   4559   |
|   Contests    |        regular ACM contests         |          start_time, end_time          |   606    |
| Problem-tags  | knowledge point (KP) tags of tasks  |       tag_id, problem_id, weight       |   100    |
|  Submissions  |   source code submitted by users    | code, language, time_cost, memory_cost | 4046652  |
|  **FINISH**   | links between users and submissions |                   -                    | 4046652  |
|  **SUBMIT**   | links between submissions and tasks |              time, result              | 4046652  |
|  **CONTAIN**  |     links between tasks and KPs     |                   -                    |   4397   |
| **BELONG TO** |  links between contests and tasks   |                 order                  |   4756   |

Users:

|            |   type   |       description       |
| :--------: | :------: | :---------------------: |
|     id     |   int    |         use ID          |
| last_login | datetime | user latests login time |

Problems:

|              |      type      |                description                 |
| :----------: | :------------: | :----------------------------------------: |
|      id      |      int       |                 problem ID                 |
| access_level |      enum      |     3 levels: private, protect, public     |
|  difficulty  | decimal(10, 5) | difficulty of problem, ranges from 0 to 10 |
|  updated_at  |    datetime    |     last revision time of the problem      |
|  creator_id  |      int       |          author ID of the problem          |

Contests:

|              |   type   |            description             |
| :----------: | :------: | :--------------------------------: |
|      id      |   int    |             contest ID             |
|  start_time  | datetime |     start time of the contest      |
|   end_time   | datetime |      end time of the contest       |
|  updated_at  | datetime | last revision time of the problem  |
| access_level |   enum   | 3 levels: private, protect, public |

Problem-tags:

|            | type |            description             |
| :--------: | :--: | :--------------------------------: |
|   tag_id   | int  | tag ID associated with the problem |
| problem_id | int  |             problem ID             |

Submissions:

|             | type  |                         description                          |
| :---------: | :---: | :----------------------------------------------------------: |
|     id      |  int  |                          contest ID                          |
|    lang     | enum  |        'c++','c','python','java','python2','python3'         |
|   result    | enum  | 'WT','JG','AC','WA','CE','REG','MLE','REP','PE','TLE','IFNR','OFNR','EFNR','OE' |
|    score    | float |          score of the submission, ranges from 0~100          |
|  time_cost  |  int  |                 runtime consumed by the code                 |
| memory_cost |  int  |             memory consumed by running the code              |
| code_length |  int  |                      length of the code                      |
|   detail    | text  |                 compiler result of the code                  |
| creator_id  |  int  |                 author ID of the submission                  |
| problem_id  |  int  |                          problem ID                          |
| contest_id  |  int  |                          contest ID                          |

### Dataset Statistics

Figure (a), (b) and (c) are the distribution of knowledge points, feedback result types and submission languages, respectively.

### ![Dataset statistics](./dataset%20statistics.png)

### Code Usage

#### Example

The following example is about visualizing embedding results for programming task difficulty, AC ratios, and heats (popularity) through ACcoding dataset.

```python
cd data_process/experiment/visualize/
python visual_transx.py
```
### ![Visualization of task embeddings](./embedding%20visualization.jpg)

