import numpy as np
import math



class DATA(object):
    def __init__(self, n_question, seqlen, separate_char, name="data", binary=False):
        # In the ASSISTments2009 dataset:
        # param: n_queation = 110
        #        seqlen = 200
        self.separate_char = separate_char
        self.n_question = n_question
        """
        self.seqlen = seqlen+1
        """
        self.seqlen = seqlen
        self.binary = binary

    ### data format
    ### 15
    ### 1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
    ### 0,1,1,1,1,1,0,0,1,1,1,1,1,0,0
    def load_data(self, path):
        f_data = open(path , 'r')
        q_data = []
        qa_data = []
        for lineID, line in enumerate(f_data):
            line = line.strip( )
            # lineID starts from 0
            if lineID % 3 == 1:
                Q = line.split(self.separate_char)
                if len( Q[len(Q)-1] ) == 0:
                    Q = Q[:-1]
                #print(len(Q))
            elif lineID % 3 == 2:
                A = line.split(self.separate_char)
                if len( A[len(A)-1] ) == 0:
                    A = A[:-1]
                #print(len(A),A)

                # start split the data
                n_split = 1
                #print('len(Q):',len(Q))
                if len(Q) > self.seqlen:
                    n_split = math.floor(len(Q) / self.seqlen)
                    if len(Q) % self.seqlen:
                        n_split = n_split + 1
                #print('n_split:',n_split)
                for k in range(n_split):
                    question_sequence = []
                    answer_sequence = []
                    if k == n_split - 1:
                        endINdex  = len(A)
                    else:
                        endINdex = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(Q[i]) > 0 :
                            if self.binary:
                                # int(A[i]) is in {0,1}
                                Xindex = int(Q[i]) * 2 + int(A[i])
                            else:
                                # int(A[i]) is in {1,2,3,4,5,6,7,8,9,10,11,12}
                                Xindex = int(Q[i]) * 12 + int(A[i]) - 1
                            question_sequence.append(int(Q[i]))
                            answer_sequence.append(Xindex)
                        else:
                            print(Q[i])
                    #print('instance:-->', len(instance),instance)
                    q_data.append(question_sequence)
                    qa_data.append(answer_sequence)
        f_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        ### convert data into ndarrays for better speed during training
        q_dataArray = np.zeros((len(q_data), self.seqlen))
        for j in range(len(q_data)):
            dat = q_data[j]
            q_dataArray[j, :len(dat)] = dat

        qa_dataArray = np.zeros((len(qa_data), self.seqlen))
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_dataArray[j, :len(dat)] = dat
        # dataArray: [ array([[],[],..])] Shape: (3633, 200)
        return q_dataArray, qa_dataArray