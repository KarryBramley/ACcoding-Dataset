import argparse
import tensorflow as tf
import deepkt, data_util, metrics
dataset, length, nb_features, nb_skills = data_util.load_dataset(fn="/data/Assistments_skill_builder_data.csv",
                                                                     batch_size=32,
                                                                     shuffle=True)

train_set, test_set, val_set = data_util.split_dataset(dataset=dataset,
                                                           total_size=length,
                                                           test_fraction=.2,
                                                           val_fraction=.2)

#print(list(test_set.as_numpy_iterator()))
print(length, nb_features, nb_skills)
print("[----- COMPILING  ------]")
model = deepkt.DKTModel(nb_features=nb_features,
                            nb_skills=nb_skills,
                            hidden_units=100,
                            dropout_rate=.3)
model.compile(
        optimizer='adam',
        metrics=[
            metrics.BinaryAccuracy(),
            metrics.AUC(),
            metrics.Precision(),
            metrics.Recall()
        ])

print(model.summary())
print("\n[-- COMPILING DONE  --]")
model.load_weights("/project/knowledge_tracing/DKT/test/weight/bestmodel")
#models.evaluate(dataset=test_set, verbose=1)

result = []
for test_batch in test_set:
    r = model.predict(test_batch, verbose=1)
    result.append(r)
#(result)
print('test_set:')
for i,t in enumerate(test_set):
    print(i, t)
print('predict:')
for i,r in enumerate(result):
    print(i, r.shape, r)
# 搞不懂predict输出的是啥诶，每个batch里面的形状和test_set还不一样。。。



