
import os
import pickle
import locale
import hashlib
import datetime
import pycountry
import itertools
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as ss
import scipy.spatial.distance as ssd

from collections import defaultdict
from matplotlib import pylab as plt
from sklearn.preprocessing import normalize
from sklearn.model_selection._split import check_cv,KFold
from Model_ import XGBClassifyCV,LGBClassifyCV,SGDClassifyCV,RandomForestClassifyCV,LRClassifyCV,StackingModel
from utils_ml import PATH,VALIDATION_SIZE,SEED,STOP_ROUNDS,BASE_LINE,TRAIN_CHUNK,TEST_CHUNK,IS_CHUNK,loc,plot,CV,features,sgd_params,xgb_params, \
                    fit_params,rf_params,lgb_params,lr_params,trans_test_split



'''
数据清洗
整体处理国家，地区等数据，用python工具处理，小写，拼接时间等问题。
'''
class DataCleaner(object):
    def __init__(self,isClean = True):
        print("数据清洗初始化开始...")
        if isClean == False:
            raise ImportError("不进行数据清洗会导致结果不准确!请改成True或不赋值成False")
        self.localIdMap = defaultdict(int)
        '''
        根据业务要求，需要将美国，加拿大编号，按照如下仿照别人写的格式即可。
        '''
        self.countryIdMap = defaultdict(int)
        self.genderIdMap = defaultdict(int,{'male':1,'female':2})
        for i, l in enumerate(locale.locale_alias.keys()):
            self.localIdMap[l] = i + 1
            ctyIdx = defaultdict(int)
            for i, c in enumerate(pycountry.countries):
                self.countryIdMap[c.name.lower()] = i + 1
                if c.name.lower() == "usa":
                    ctyIdx["US"] = i
                if c.name.lower() == "canada":
                    ctyIdx["CA"] = i
            for cc in ctyIdx.keys():
                for s in pycountry.subdivisions.get(country_code=cc):
                    self.countryIdMap[s.name.lower()] = ctyIdx[cc] + 1
        print("数据清洗初始化结束...\n\n{}\n".format("*" * 200))
    '''
    获取本地ID
    '''
    def get_local_id(self,local:str):
        return self.localIdMap[local.lower()]
    '''
    获取性别ID
    '''
    def get_gender_id(self,gender:str):
        return self.genderIdMap[gender]
    '''
    获取城市ID
    '''
    def get_country_id(self,location):
        if isinstance(location,str) and len(location) > 0 and location.rfind(" ") > -1:
            return self.countryIdMap[location[location.rindex(" ")+2:].lower()]
        else:
            return 0
    '''
    获取并拼接年份和月份
    '''
    def get_join_year_of_month(self,data:str):
        time_total = datetime.datetime.strptime(data,"%Y-%m-%dT%H:%M:%S.%fZ")
        return ''.join([str(time_total.year),str(time_total.month)])
    '''
    获取生日年份并转为Int
    '''
    def get_birth_year_of_int(self,birthYear):
        try:
            return 0 if birthYear == "None" else int(birthYear)
        except:
            return 0
    '''
    获取时间并转为int
    '''
    def get_timezone_of_int(self, timezone):
        try:
            return int(timezone)
        except:
            return 0
    '''
    获取特征的哈希值
    '''
    def get_feature_hash(self, value):
        if len(value.strip()) == 0:
            return -1
        else:
            return int(hashlib.sha224(value).hexdigest()[0:4], 16)
    '''
    获取数值并转为float类型
    '''
    def get_float_value(self, value):
        if len(value.strip()) == 0:
            return 0.0
        else:
            return float(value)


'''
构建相似度矩阵
'''

'''
构建用户-事件评分矩阵
为了节约内存，我们用open方式读取
'''
class UserOfEventCleaner(object):
    def __init__(self,isClean = True):
        print("构建用户-事件评分矩阵初始化开始...")
        if isClean == False:
            raise ImportError("不进行构建用户-事件评分矩阵会导致结果不准确!请改成True或不赋值成False")
        self.user_unique = set()                # 去重后的用户
        self.event_unique = set()               # 去重后的事件
        self.uniqueUserPairs = set()            # 参与计算的用户(必须保证至少参加俩次事件，否则无法计算相似度)
        self.uniqueEventPairs = set()           # 参与计算的事件(必须保证至少被俩个用户参加))
        self.userIndex = dict()                 # 用户坐标
        self.eventIndex = dict()                # 事件坐标
        self.eventForUser = defaultdict(set)    # 事件—用户评分矩阵
        self.userForEvent = defaultdict(set)    # 用户-事件评分矩阵
        for name in ['train.csv','test.csv']:
            '''
            为了节约内存，用open方式进行处理，然后将每列相应位置特征数值放入去重后的用户事件列表中。
            再用用户-事件字典分别存储相应的用户/事件。
            '''
            with open(PATH+name,'r') as reader:
                for r in reader:
                    cols = str(reader.readline().strip()).split(',')
                    if len(cols) > 2:
                        self.user_unique.add(cols[0])
                        self.event_unique.add(cols[1])
                        self.eventForUser[cols[0]].add(cols[1])
                        self.userForEvent[cols[1]].add(cols[0])
                    else:
                        continue

        '''
        初始化用户事件评分矩阵
        '''
        self.userEventScores = ss.dok_matrix((len(self.user_unique),len(self.event_unique)))
        '''
        初始化用户/事件坐标表
        '''
        for i,u in enumerate(self.user_unique):
            self.userIndex[u] = i
        for i,e in enumerate(self.event_unique):
            self.eventIndex[e] = i

        '''
        构建用户-事件评分表
        '''
        print("构建用户-事件评分表开始...")
        with open(PATH + 'train.csv','r') as reader_train:
            for m in reader_train:
                cols = str(reader_train.readline().strip()).split(',')
                if len(cols) > 2 and len(cols) < 7:
                    i = self.userIndex[cols[0]]
                    j = self.eventIndex[cols[1]]
                    self.userEventScores[i,j] = int(cols[4]) - int(cols[5])
                else:
                    continue
        print("构建用户-事件评分表结束...\n\n{}\n".format("*"*200))
        '''
        保存用户-事件评分矩阵
        '''
        sio.mmwrite("User_Events_Scores",self.userEventScores)
        '''
        计算相似度，是至少一个事件有俩个用户评分。
        或一个用户至少对俩个事件评分。所以清除掉一个事件对应一个用户这种无关计算的数据
        '''
        for event in self.event_unique:
            user = self.userForEvent[event]
            if len(user) > 2:
                self.uniqueUserPairs.update(itertools.combinations(user,2))
        for user in self.user_unique:
            event = self.eventForUser[user]
            if len(event) > 2:
                self.uniqueEventPairs.update(itertools.combinations(event,2))
        print(self.userIndex)
        print(self.eventIndex)
        '''
        这种dict模式只能用dump存储。不能用mmwrite
        '''
        pickle.dump(self.userIndex,open('User_Index.pkl','wb'))
        pickle.dump(self.eventIndex,open('Event_index.pkl','wb'))
        print("构建用户-事件评分矩阵初始化结束...\n\n{}\n".format("*" * 200))


'''
构建用户-用户相似度矩阵,我们用ssd.correlation皮尔逊相似度
'''
class UserSimilar(object):
    def __init__(self,isUser = True,programEntities = None, sim=ssd.correlation):
        print("构建用户-用户相似度矩阵初始化开始...")
        if isUser == False:
            raise ImportError("不进行构建用户-事件评分矩阵会导致结果不准确!请改成True或不赋值成False")
        self.data_cleaner = DataCleaner()
        self.programEntities = programEntities
        self.user_num = len(programEntities.userIndex.keys())
        with open(PATH + 'users.csv', 'r') as reader:
            col_names = str(reader.readline().strip()).split(',')
            self.userMatrix = ss.dok_matrix((self.user_num, len(col_names) - 1))
            self.userSimilarMatrix = ss.dok_matrix((self.user_num, self.user_num))
            for line in reader:
                cols = line.strip().split(',')
                '''
                python3放弃了dict的has_key,直接用in 判断是否在字典里即可
                '''
                if cols[0] in self.programEntities.userIndex:
                    i = self.programEntities.userIndex[cols[0]]
                    self.userMatrix[i, 0] = self.data_cleaner.get_local_id(cols[1])
                    self.userMatrix[i, 1] = self.data_cleaner.get_birth_year_of_int(cols[2])
                    self.userMatrix[i, 2] = self.data_cleaner.get_gender_id(cols[3])
                    self.userMatrix[i, 3] = self.data_cleaner.get_join_year_of_month(cols[4])
                    self.userMatrix[i, 4] = self.data_cleaner.get_country_id(cols[5])
                    self.userMatrix[i, 5] = self.data_cleaner.get_timezone_of_int(cols[6])
        '''
        构建完直接归一化，用l1正则，帮助我们成稀疏矩阵。然后保存
        '''
        self.userMatrix = normalize(self.userMatrix, norm='l1', axis=0, copy=False)
        sio.mmwrite('User_Matrix', self.userMatrix)
        '''
        计算用户相似度矩阵
        '''
        for i in range(0, self.user_num):
            self.userSimilarMatrix[i, i] = 1.0
        for u1, u2 in self.programEntities.uniqueUserPairs:
            i = self.programEntities.userIndex[u1]
            j = self.programEntities.userIndex[u2]
            if (i, j) not in self.userSimilarMatrix:
                user_sim = sim(self.userMatrix.getrow(i).todense(), self.userMatrix.getrow(j).todense())
                self.userSimilarMatrix[i, j] = user_sim
                self.userSimilarMatrix[j, i] = user_sim
        sio.mmwrite('User_Sim_Matrix', self.userMatrix)
        print("构建用户-用户相似度矩阵初始化结束...\n\n{}\n".format("*" * 200))


'''
构建事件-事件相似度矩阵
注意这里有2种相似度：
（1）由用户-event行为，类似协同过滤算出的相似度
（2）由event本身的内容(event信息)计算出的event-event相似度
scipy.spatial.distance :  计算相似度
(1)correlation：距离相似度。常用做用户行为分析的数值
(2)cosine；余弦相似度 常用做文本分析，所以这里用户信息我们用余弦相似度
'''
class EventSimilar(object):
    def __init__(self,programEntities,isEvent = True, pro_sim=ssd.correlation,cont_sim = ssd.cosine):
        print("构建事件-事件相似度矩阵初始化开始...")
        if isEvent == False:
            raise ImportError("不进行构建事件-事件相似度矩阵会导致结果不准确!请改成True或不赋值成False")
        self.data_cleaner = DataCleaner()
        self.programEntities = programEntities
        self.event_num = len(programEntities.eventIndex.keys())
        self.eventPropMatrix = ss.dok_matrix((self.event_num, 7))
        self.eventContMatrix = ss.dok_matrix((self.event_num, 100))
        self.eventPropSim = ss.dok_matrix((self.event_num, self.event_num))
        self.eventContSim = ss.dok_matrix((self.event_num, self.event_num))
        with open(PATH + 'events.csv', 'rb') as reader:
            reader = str(reader.readline()).strip().split(',')
            for line in reader:
                cols = line.strip().split(',')
                if cols[0] in self.programEntities.eventIndex:
                    i = self.programEntities.eventIndex[cols[0]]
                    self.eventPropMatrix[i, 0] = self.data_cleaner.get_join_year_of_month(cols[2])
                    self.eventPropMatrix[i, 1] = self.data_cleaner.get_feature_hash(cols[3])
                    self.eventPropMatrix[i, 2] = self.data_cleaner.get_feature_hash(cols[4])
                    self.eventPropMatrix[i, 3] = self.data_cleaner.get_feature_hash(cols[5])
                    self.eventPropMatrix[i, 4] = self.data_cleaner.get_feature_hash(cols[6])
                    self.eventPropMatrix[i, 5] = self.data_cleaner.get_float_value(cols[7])
                    self.eventPropMatrix[i, 6] = self.data_cleaner.get_float_value(cols[8])
        self.eventPropMatrix = normalize(self.eventPropMatrix, norm='l1', axis=0, copy=False)
        sio.mmwrite('Event_Prop_Matrix', self.eventPropMatrix)
        self.eventContMatrix = normalize(self.eventContMatrix, norm='l1', axis=0, copy=False)
        sio.mmwrite('Event_Cont_Matrix', self.eventContMatrix)
        for i in range(self.event_num):
            self.eventPropSim[i, i] = 1.0
            self.eventContSim[i, i] = 1.0
        for e1, e2 in self.programEntities.uniqueEventPairs:
            i = self.programEntities.eventIndex[e1]
            j = self.programEntities.eventIndex[e2]
            if (i,j) not in self.eventPropSim:
                event_sim_pro = pro_sim(self.eventPropMatrix.getrow(i).todense(),
                                        self.eventPropMatrix.getrow(j).todense())
                self.eventPropSim[i, j] = event_sim_pro
                self.eventPropSim[j, i] = event_sim_pro
            if (i, j) not in self.eventContSim:
                event_sim_con = cont_sim(self.eventContMatrix.getrow(i).todense(),
                                         self.eventContMatrix.getrow(j).todense())
                self.eventContSim[i, j] = event_sim_con
                self.eventContSim[j, i] = event_sim_con
        sio.mmwrite("Event_Prop_Sim_Matrix", self.eventPropSim)
        sio.mmwrite("Event_Cont_Sim_Matrix", self.eventContSim)
        print("构建事件-事件相似度矩阵结束...\n\n{}\n".format("*" * 200))


'''
构建用户社交评分矩阵 (Friends数据)
我们先按照朋友多外向，喜欢参加活动，或者喜欢参加一些外向的活动，当然这不一定准确，当做一个中级特征来挖掘。
'''
class UserFriends(object):
    def __init__(self, programEntities = None,isClean = True):
        print("构建用户社交评分矩阵初始化开始...")
        if isClean == False:
            raise ImportError("不进行构特征挖掘--用户社交会导致结果不准确!请改成True或不赋值成False")
        self.programEntities = programEntities
        self.user_num = len(programEntities.userIndex.keys())
        self.numFriends = np.zeros((self.user_num))
        self.userFriends = ss.dok_matrix((self.user_num, self.user_num))
        with open(PATH + 'user_friends.csv', 'rb') as reader:
            reader = str(reader.readline())
            ln = 0
            for line in reader:
                if ln % 200 == 0:
                    cols = line.strip().split(",")
                    user = cols[0]
                    if user in self.programEntities.userIndex:
                        friends = cols[1].split(" ")
                        i = self.programEntities.userIndex[user]
                        self.numFriends[i] = len(friends)
                        for friend in friends:
                            if friend in self.programEntities.userIndex:
                                j = self.programEntities.userIndex[friend]
                                eventsForUser = self.programEntities.userEventScores.getrow(j).todense()
                                score = eventsForUser.sum() / np.shape(eventsForUser)[1]
                                self.userFriends[i, j] += score
                                self.userFriends[j, i] += score
                ln += 1
            sumNumFriends = self.numFriends.sum(axis=0)
            self.numFriends = self.numFriends / sumNumFriends
            '''
            我们保存俩种，一种是归一化前，一种是归一化后。
            '''
            sio.mmwrite("Num_Friends", np.matrix(self.numFriends))
            self.userFriends = normalize(self.userFriends, norm="l1", axis=0, copy=False)
            sio.mmwrite("User_Friends", self.userFriends)
            print("特征挖掘--用户社交初始化结束...\n\n{}\n".format("*" * 200))




'''
构建活跃度矩阵 -- 依旧参加人数来判断是否活跃
构建列为1维即可，主要看是否活跃
'''
class EventAttend(object):
    def __init__(self, programEntities = None,isClean = True):
        print("统计活跃度初始化开始...")
        if isClean == False:
            raise ImportError("不进行统计活跃度会导致结果不准确!请改成True或不赋值成False")
        self.programEntities = programEntities
        self.num_events = len(self.programEntities .eventIndex.keys())
        self.eventPopularity = ss.dok_matrix((self.num_events, 1))
        with open('event_attendees.csv','rb') as reader:
            reader = str(reader.readline())
            for line in reader:
                cols = line.strip().split(",")
                eventId = cols[0]
                if eventId in self.programEntities.eventIndex:
                    i = self.programEntities.eventIndex[eventId]
                    self.eventPopularity[i, 0] = len(cols[1].split(" ")) - len(cols[4].split(" "))
        self.eventPopularity = normalize(self.eventPopularity, norm="l1",axis=0, copy=False)
        sio.mmwrite("Event_Popularity", self.eventPopularity)
        print("统计活跃度结束...\n\n{}\n".format("*"*200))


'''
特征工程
'''
class FeatureEngineering(object):
    def __init__(self,isFeature = True):
        print("特征工程初始化开始...")
        if isFeature == False:
            raise ImportError("不进行特征工程会导致结果不准确!请改成True或不赋值成False")
        with open('User_Index.pkl','rb') as reader_user:
            self.userIndex = pickle.load(reader_user)
        with open('Event_index.pkl','rb') as reader_event:
            self.eventIndex = pickle.load(reader_event)
        self.userEventScores = sio.mmread("User_Events_Scores").todense()
        self.userSimMatrix = sio.mmread("User_Sim_Matrix").todense()
        self.eventPropSim = sio.mmread("Event_Prop_Sim_Matrix").todense()
        self.eventContSim = sio.mmread("Event_Cont_Sim_Matrix").todense()
        self.numFriends = sio.mmread("Num_Friends")
        self.userFriends = sio.mmread("User_Friends").todense()
        self.eventPopularity = sio.mmread("Event_Popularity").todense()
        print(self.userIndex)
        print("*"*200)
        print(self.eventIndex)
        print("特征工程初始化结束...\n\n{}\n".format("*"*200))

    '''
    基于用户协同过滤，得到event的推荐度，我们把这个当做一个feature
    返回的是当前位置的相似度评分-自身评分即可。
    '''
    def user_recommend(self,userId,eventId):
        if userId is None or eventId is None:
            raise ValueError("输入的用户/事件ID不能为空")
        i = self.userIndex[userId]
        j = self.eventIndex[eventId]
        '''
        userEventScores的横坐标是user,纵坐标是event,
        因为我们是基于用户，所以我们取所有用户对同一事件的得分，
        '''
        score = self.userEventScores[:,j]
        '''
        userSimMatrix 横坐标是用户，纵坐标是相似度，我们用哪个用户，就选取哪个用户。这样就形成这个用户对这个事件的评分。
        '''
        sim = self.userSimMatrix[i,:]
        pre = score*sim
        try:
            return pre[0,0] - self.userEventScores[i,j]
        except IndexError:
            return 0

    '''
    基于事件的协同过滤，得到event的推荐度，我们把这个当做一个feature.同理
    '''
    def event_recommend(self,userId,eventId):
        if userId is None or eventId is None:
            raise ValueError("输入的用户/事件ID不能为空")
        i = self.userIndex[userId]
        j = self.eventIndex[eventId]
        '''
        同理，基于事件，就取所有事件,取俩个相似度。
        '''
        score =self.userEventScores[i,:]
        sim_con = self.eventContSim[:,j]
        sim_pro = self.eventPropSim[:,j]
        pro_score = sim_pro * score
        con_score = sim_con * score
        pre_con = 0
        pre_pro = 0
        try:
            pre_con = con_score[0,0] - self.eventContSim[i,j]
        except IndexError:
            pass
        try:
            pre_pro = pro_score[0, 0] - self.eventPropSim[i, j]
        except IndexError:
            pass
        return pre_con,pre_pro
    '''
    基于用户的朋友个数来推断用户的社交程度
    主要的考量是如果用户的朋友非常多，可能会更倾向于参加各种社交活动
    '''
    def user_pop(self, userId):
        if userId in self.userIndex:
            i = self.userIndex[userId]
            try:
                return self.numFriends[0, i]
            except IndexError:
                return 0
        else:
            return 0

    '''
    朋友对用户的影响
    主要考虑用户所有的朋友中，有多少是非常喜欢参加各种社交活动/event的
    用户的朋友圈如果都积极参与各种event，可能会对当前用户有一定的影响
    '''
    def friendInfluence(self, userId):
        num_users = np.shape(self.userFriends)[1]
        i = self.userIndex[userId]
        return (self.userFriends[i, :].sum(axis=0) / num_users)[0, 0]

    '''
    本活动本身的热度
    主要是通过参与的人数来界定的
    '''
    def eventPop(self, eventId):
        i = self.eventIndex[eventId]
        return self.eventPopularity[i, 0]

    def get_variable_name(variable):
        for key in loc:
            if loc[key] == variable:
                return key

    """
    把前面user-based协同过滤 和 item-based协同过滤，以及各种热度和影响度作为特征组合在一起
    生成新的训练数据，用于分类器分类使用
    """

    def rewriteData(self, start=1, train=True, header=True):
        fn = "train.csv" if train else "test.csv"
        with open(fn, 'rb') as fin:
            if header:
                if os.path.exists('data_'+fn) == True:
                    os.remove('data_'+fn)
                with open('data_'+fn,'w') as writer:
                    ocolnames = ["invited", "user_reco", "evt_p_reco","evt_c_reco", "user_pop", "frnd_infl", "evt_pop"]
                    ocolnames.append("interested")
                    ocolnames.append("not_interested")
                    columns_end = ",".join(ocolnames)+"\n"
                    writer.write(columns_end)
                    ln = 0
                    for line in fin:
                        ln += 1
                        if ln < start:
                            continue
                        cols = str(line).strip().split(",")
                        userId = cols[0][2:]
                        eventId = cols[1]
                        invited = cols[2]
                        if ln % 500 == 0:
                           print("%s:%d (userId, eventId)=(%s, %s)" % (fn, ln, userId, eventId))
                        #print(userId," ",eventId," ",invited)
                        if userId in self.userIndex and eventId in self.eventIndex:
                            user_reco = self.user_recommend(userId, eventId)
                            evt_p_reco, evt_c_reco = self.event_recommend(userId, eventId)
                            user_pop = self.user_pop(userId)
                            frnd_infl = self.friendInfluence(userId)
                            evt_pop = self.eventPop(eventId)
                            ocols = [invited, user_reco, evt_p_reco,evt_c_reco, user_pop, frnd_infl, evt_pop]

                            if train:
                                ocols.append(cols[4])  # interested
                                ocols.append(cols[5][0])  # not_interested
                            writer.write(",".join(map(lambda x: str(x), ocols)) + "\n")

        def rewriteTrainingSet(self):
            self.rewriteData(True)

        def rewriteTestSet(self):
            self.rewriteData(False)







def main():
    DataCleaner()
    programEntities = UserOfEventCleaner()
    UserSimilar(programEntities = programEntities)
    EventSimilar(programEntities = programEntities)
    UserFriends(programEntities = programEntities)
    EventAttend(programEntities = programEntities)

    dr = FeatureEngineering()
    print("生成训练数据...\n")
    dr.rewriteData(train=True, start=2, header=True)
    print("生成预测数据...\n")
    dr.rewriteData(train=False, start=2, header=True)

    '''
    LGB模型
    '''
    lgb_model = LGBClassifyCV(lgb_params = lgb_params)
    trainDf = pd.read_csv("data_train.csv")
    testDf = pd.read_csv("data_test.csv")
    train_X = np.matrix(pd.DataFrame(trainDf, index=None,
                                     columns=features))
    trainDf.fillna(0, inplace=True)
    testDf.fillna(0, inplace=True)
    train_X = np.matrix(pd.DataFrame(trainDf, index=None,
                               columns=features))
    test_X = np.matrix(pd.DataFrame(testDf,index = None,columns = features))
    y = trainDf.interested
    lgb_model.fit(train_X,y)

    print("LGB模型最优参数为:",lgb_model.get_params)
    print("LGB模型Auc值为:",lgb_model.cv_scores_)
    print("LGB模型最佳Auc值为:",lgb_model.cv_score_)
    print("模型开始保存...")
    try:
        with open('LgbModel1.pickle', 'wb') as writer:
            pickle.dump(lgb_model, writer)
    except IOError:
        print("模型保存失败...")
    print("模型保存结束...\n\n{}\n".format("*" * 200))

    print("模型开始读取...")
    with open('LgbModel1.pickle', 'rb') as reader:
        lgb_model = pickle.load(reader)

    print("模型读取结束...\n\n{}\n".format("*" * 200))

    '''
    XGB模型
    '''
    xgb_model = XGBClassifyCV(xgb_params = xgb_params,fit_params = fit_params)
    trainDf = pd.read_csv("data_train.csv")
    testDf = pd.read_csv("data_test.csv")
    trainDf.fillna(0, inplace=True)
    testDf.fillna(0, inplace=True)
    train_X = np.matrix(pd.DataFrame(trainDf, index=None,
                               columns=features))
    test_X = np.matrix(pd.DataFrame(testDf,index = None,columns = features))
    y = np.array(trainDf.interested)
    xgb_model.fit(train_X,y)

    print("XGB模型最优参数为:",xgb_model.get_params)
    print("XGB模型Auc值为:",xgb_model.cv_scores_)
    print("XGB模型最佳Auc值为:",xgb_model.cv_score_)
    print("模型开始保存...")
    try:
        with open('XgbModel1.pickle', 'wb') as writer:
            pickle.dump(xgb_model, writer)
    except IOError:
        print("模型保存失败...")
    print("模型保存结束...\n\n{}\n".format("*" * 200))

    print("模型开始读取...")
    with open('XgbModel1.pickle', 'rb') as reader:
        xgb_model = pickle.load(reader)

    print("模型读取结束...\n\n{}\n".format("*" * 200))

    print("预测结果开始...")
    predict = xgb_model.predict(test_X)
    print(predict)
    print("预测结果结束...\n\n{}\n".format("*" * 200))

    print("可视化特征开始...")
    if plot:
        with open('xgb.fmap', 'w') as writer:
            i = 0
            for feat in features:
                writer.write('{0}\t{1}\tq\n'.format(i, feat))
                i = i + 1
        importance = xgb_model.feature_importances_
        print(importance)
        importance = zip(features,importance)
        importance = sorted(importance)
        df = pd.DataFrame(importance, columns=['feature', 'score'])
        df['score'] = df['score'] / df['score'].sum()
        plt.figure()
        df.plot()
        df.plot(kind='barh', x='feature', y='score', legend=False, figsize=(25, 15))
        plt.title('XGBoost_Feature_Importance')
        plt.xlabel('Feature_Importance')
        plt.gcf().savefig('推荐系统—特征重要程度可视化-XGBoost.png')
    print("可视化特征结束...\n\n{}\n".format("*" * 200))

    '''
    SGD模型
    '''
    sgd_model = SGDClassifyCV(sgd_params = sgd_params)

    trainDf = pd.read_csv("data_train.csv")
    testDf = pd.read_csv("data_test.csv")
    trainDf.fillna(0,inplace = True)
    testDf.fillna(0, inplace = True)
    train_X = np.matrix(pd.DataFrame(trainDf, index=None,
                                     columns=features))
    test_X = np.matrix(pd.DataFrame(testDf, index=None, columns=features))
    y = np.array(trainDf.interested)
    sgd_model.fit(train_X, y)
    print("SGD模型最优参数为:", sgd_model.get_params)
    print("SGD模型Auc值为:", sgd_model.cv_scores_)
    print("SGD模型最佳Auc值为:", sgd_model.cv_score_)

    print("模型开始保存...")
    try:
        with open('SgdModel1.pickle', 'wb') as writer:
            pickle.dump(sgd_model, writer)
    except IOError:
        print("模型保存失败...")
    print("模型保存结束...\n\n{}\n".format("*" * 200))
    print("模型开始读取...")
    with open('SgdModel1.pickle', 'rb') as reader:
        sgd_model = pickle.load(reader)
    print("模型读取结束...\n\n{}\n".format("*" * 200))

    '''
    RF模型
    '''
    rf_model = RandomForestClassifyCV(rf_params = rf_params)
    trainDf = pd.read_csv("data_train.csv")
    testDf = pd.read_csv("data_test.csv")
    trainDf.fillna(0, inplace=True)
    testDf.fillna(0, inplace=True)
    train_X = np.matrix(pd.DataFrame(trainDf, index=None,
                                     columns=features))
    test_X = np.matrix(pd.DataFrame(testDf, index=None, columns=features))
    y = np.array(trainDf.interested)
    rf_model.fit(train_X, y)
    print("随机森林模型最优参数为:", rf_model.get_params)
    print("随机森林模型Auc值为:", rf_model.cv_scores_)
    print("随机森林模型最佳Auc值为:", rf_model.cv_score_)
    print("模型开始保存...")
    try:
        with open('RfModel1.pickle', 'wb') as writer:
            pickle.dump(rf_model, writer)
    except IOError:
        print("模型保存失败...")
    print("模型保存结束...\n\n{}\n".format("*" * 200))
    print("模型开始读取...")
    with open('RfModel1.pickle', 'rb') as reader:
        rf_model = pickle.load(reader)
    print("模型读取结束...\n\n{}\n".format("*" * 200))

    '''
    LR模型
    '''
    lr_params = {
        'solver':'sag',
        'class_weight':'balanced',
        'random_state':SEED,
        'n_jobs':-1
    }
    lr_model = LRClassifyCV(lr_params = lr_params)
    lr_model.fit(train_X, y)
    print("逻辑回归模型最优参数为:", lr_model.get_params)
    print("逻辑回归模型Auc值为:", lr_model.cv_scores_)
    print("逻辑回归模型最佳Auc值为:", lr_model.cv_score_)
    print("模型开始保存...")
    try:
        with open('LrModel1.pickle', 'wb') as writer:
            pickle.dump(lr_model, writer)
    except IOError:
        print("模型保存失败...")
    print("模型保存结束...\n\n{}\n".format("*" * 200))
    print("模型开始读取...")
    with open('LrModel1.pickle', 'rb') as reader:
        lr_model = pickle.load(reader)
    print("模型读取结束...\n\n{}\n".format("*" * 200))

    '''
    模型融合
    '''
    trainDf = pd.read_csv("data_train.csv")
    testDf = pd.read_csv("data_test.csv")
    trainDf.fillna(0, inplace=True)
    testDf.fillna(0, inplace=True)
    train_X = np.matrix(pd.DataFrame(trainDf, index=None,
                                     columns=features))
    test_X = np.matrix(pd.DataFrame(testDf, index=None, columns=features))
    y = np.array(trainDf.interested)

    '''
    模型加载
    '''
    with open('LrModel1.pickle','rb') as reader_lr:
        lr_model = pickle.load(reader_lr)
    with open('RfModel1.pickle','rb') as reader_rf:
        rf_model = pickle.load(reader_rf)
    with open('XgbModel1.pickle','rb') as reader_xgb:
        xgb_model = pickle.load(reader_xgb)
    with open('SgdModel1.pickle','rb') as reader_sgd:
        sgd_model = pickle.load(reader_sgd)
    with open('LgbModel1.pickle','rb') as reader_lgb:
        lgb_model = pickle.load(reader_lgb)

    print("开始模型融合...")
    total_model = StackingModel(mod=(xgb_model,sgd_model,lgb_model,rf_model),meta_model = lr_model)
    print("模型融合初始化结束...")
    cv = check_cv(CV, y, classifier=True)
    scores = []
    estimators_ = []
    for train, valid in cv.split(train_X, y):
        score1 = 0
        test = len(y[valid])
        print("融合模型开始拟合")
        clf = total_model.fit(train_X[train],y[train])
        print("融合模型拟合结束")
        for i in range(0, test):
            yt = clf.predict(train_X[valid][i, :])
            if yt == y[valid][i]:
                score1 += 1
        score1 = score1 / test
        print(score1)
        scores.append(score1)
        estimators_.append(clf)

    score = sum(scores) / len(scores)
    print("模型融合的最佳AUC值为:",score)




if __name__ == '__main__':
    main()
