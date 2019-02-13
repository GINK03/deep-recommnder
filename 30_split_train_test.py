import pandas as pd
import random

df = pd.read_csv('works/dataset/preprocess.csv')

userIds = list( df['userId'].unique() )
testIds  = set(random.sample(userIds, len(userIds)//5))
trainIds = set(userId for userId in userIds if userId not in testIds)

df[df['userId'].apply(lambda x:x in testIds)].to_csv('works/dataset/test.csv', index=None)
df[df['userId'].apply(lambda x:x in trainIds)].to_csv('works/dataset/train.csv', index=None)

