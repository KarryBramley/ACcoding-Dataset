f_train = open("mini/train2id_mini.txt", "r")
f_test = open("mini/test2id_mini.txt", "r")
f_valid = open("mini/valid2id_mini.txt", "r")
f_train_new = open("mini/train2id.txt", "w")
f_test_new = open("mini/test2id.txt", "w")
f_valid_new = open('mini/valid2id.txt', 'w')

train_len = 200
train_data = f_train.readlines()
test_data = f_test.readlines()
valid_data = f_valid.readlines()
h_list = []
r_list = []
t_list = []

for tr in train_data[1:train_len]:
    h, t, r = tr.split()
    print(h, t, r)
    h_list.append(h)
    r_list.append(r)
    t_list.append(t)
    f_train_new.write(h + '\t' + t + '\t' + r+'\n')

for te in test_data[1:]:
    p, h, t, r = te.split()
    if h in h_list and r in r_list and t in t_list:
        f_test_new.write(p+'\t'+h+'\t'+t+'\t'+r+'\n')

for va in valid_data[1:]:
    h, t, r = va.split()
    if h in h_list and r in r_list and t in t_list:
        f_valid_new.write(h+'\t'+t+'\t'+r+'\n')