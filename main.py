
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from utils import *
from sklearn.cross_validation import train_test_split,StratifiedKFold
from optparse import OptionParser
import scipy.io as sio

parser = OptionParser()
parser.add_option("-d", "--d", default=1024, help="The embedding dimension d")  #定义嵌入的变量表示维度
parser.add_option("-n","--n",default=1, help="global norm to b e clipped")
parser.add_option("-k","--k",default=479,help="The dimension of project matrices k")
parser.add_option("-t","--t",default = "o",help="Test scenario")
parser.add_option("-r","--r",default = "all",help="positive negative ratio")

(opts, args) = parser.parse_args()


def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)

def row_normalize(a_matrix, substract_self_loop):
    if substract_self_loop == True:                #要不要检测矩阵填充0
        np.fill_diagonal(a_matrix,0)
    a_matrix = a_matrix.astype(float)
    row_sums = a_matrix.sum(axis=1)+1e-12
    new_matrix = a_matrix / row_sums[:, np.newaxis]
    new_matrix[np.isnan(new_matrix) | np.isinf(new_matrix)] = 0.0
    return new_matrix
a = 1#主要是两种相似性的融合
#load network
network_path = '../data_drug/'

drug_drug = np.loadtxt(network_path+'DrugSim.txt')

drug_drug2 = sio.loadmat(network_path+'dru.mat')
drug_drug = drug_drug * a + drug_drug2["dd"] * (1-a)
print('loaded drug drug', check_symmetric(drug_drug), np.shape(drug_drug))

disease_drug = np.loadtxt(network_path+'DiDrA.txt')
print('loaded disease_drug', np.shape(disease_drug))
drug_disease = disease_drug.T

disease_disease = np.loadtxt(network_path+'DiseaseSim.txt')
disease_disease2 = sio.loadmat(network_path+'dis.mat')


disease_disease = disease_disease * a + disease_disease2["LL"] * (1-a)
print('loaded disease disease', check_symmetric(disease_disease), np.shape(disease_disease))

#normalize network for mean pooling aggregation
drug_drug_normalize = row_normalize(drug_drug,True)
disease_disease_normalize = row_normalize(disease_disease,True)

drug_disease_normalize = row_normalize(drug_disease,False)
disease_drug_normalize = row_normalize(disease_drug,False)

#define computation graph
num_drug = len(drug_drug_normalize)
num_disease = len(disease_disease_normalize)
# num_protein = len(protein_protein_normalize)
# num_sideeffect = len(sideeffect_drug_normalize)

dim_drug = int(opts.d)
# dim_protein = int(opts.d)
dim_disease = int(opts.d)
# dim_sideeffect = int(opts.d)
dim_pred = int(opts.k)
dim_pass = int(opts.d)

class Model(object):
    def __init__(self):
        self._build_model()

    def _build_model(self):
        #inputs
        self.drug_drug = tf.placeholder(tf.float32, [num_drug, num_drug])
        self.drug_drug_normalize = tf.placeholder(tf.float32, [num_drug, num_drug])

        self.drug_disease = tf.placeholder(tf.float32, [num_drug, num_disease])
        self.drug_disease_normalize = tf.placeholder(tf.float32, [num_drug, num_disease])

        self.disease_disease = tf.placeholder(tf.float32, [num_disease, num_disease])
        self.disease_disease_normalize = tf.placeholder(tf.float32, [num_disease, num_disease])

        self.disease_drug = tf.placeholder(tf.float32, [num_disease, num_drug])
        self.disease_drug_normalize = tf.placeholder(tf.float32, [num_disease, num_drug])


        self.drug_disease_mask = tf.placeholder(tf.float32, [num_drug, num_disease])

        #features
        self.drug_embedding = weight_variable([num_drug,dim_drug])
        self.disease_embedding = weight_variable([num_disease, dim_disease])


        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(self.drug_embedding))
     # tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(self.protein_embedding))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(self.disease_embedding))
        # tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(self.sideeffect_embedding))



        #feature passing weights (maybe different types of nodes can use different weights)
        W0 = weight_variable([dim_pass, dim_drug])
        b0 = bias_variable([dim_drug])
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W0))

        #passing 1 times (can be easily extended to multiple passes)

        print('')
        drug_vector1 = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(
            tf.concat([tf.matmul(self.drug_drug_normalize, a_layer(self.drug_embedding, dim_pass)) + \
            tf.matmul(self.drug_disease_normalize, a_layer(self.disease_embedding, dim_pass)) + \
            self.drug_embedding], axis=1), W0)+b0),dim=1)

        disease_vector1 = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(
            tf.concat([tf.matmul(self.disease_disease_normalize, a_layer(self.disease_embedding, dim_pass)) + \
            tf.matmul(self.disease_drug_normalize, a_layer(self.drug_embedding, dim_pass)) + \
            self.disease_embedding], axis=1), W0)+b0),dim=1)




        self.drug_representation = drug_vector1
        # self.protein_representation = protein_vector1
        self.disease_representation = disease_vector1
        # self.sideeffect_representation = sideeffect_vector1

        #reconstructing networks
        self.drug_drug_reconstruct = bi_layer(self.drug_representation,self.drug_representation, sym=True, dim_pred=dim_pred)
        self.drug_drug_reconstruct_loss = tf.reduce_sum(tf.multiply((self.drug_drug_reconstruct-self.drug_drug), (self.drug_drug_reconstruct-self.drug_drug)))

        self.disease_disease_reconstruct = bi_layer(self.disease_representation,self.disease_representation, sym=True, dim_pred=dim_pred)
        self.disease_disease_reconstruct_loss = tf.reduce_sum(tf.multiply((self.disease_disease_reconstruct-self.disease_disease), (self.disease_disease_reconstruct-self.disease_disease)))


        self.drug_disease_reconstruct = bi_layer(self.drug_representation,self.disease_representation, sym=False, dim_pred=dim_pred)
        tmp = tf.multiply(self.drug_disease_mask, (self.drug_disease_reconstruct-self.drug_disease))
        self.drug_disease_reconstruct_loss = tf.reduce_sum(tf.multiply(tmp, tmp))

        self.l2_loss = tf.add_n(tf.get_collection("l2_reg"))

        self.loss = self.drug_disease_reconstruct_loss + 1.0*(self.drug_drug_reconstruct_loss+
                                                            self.disease_disease_reconstruct_loss
                                                            ) + self.l2_loss

graph = tf.get_default_graph()
with graph.as_default():
    model = Model()
    learning_rate = tf.placeholder(tf.float32, [])
    total_loss = model.loss
    dti_loss = model.drug_disease_reconstruct_loss
    optimize = tf.train.AdamOptimizer(learning_rate)
    gradients, variables = zip(*optimize.compute_gradients(total_loss))
    gradients, _ = tf.clip_by_global_norm(gradients, int(opts.n))
    optimizer = optimize.apply_gradients(zip(gradients, variables))
    eval_pred = model.drug_disease_reconstruct

def train_and_evaluate(DTItrain, DTIvalid, DTItest, graph, verbose=True, num_steps = 800):
    drug_disease = np.zeros((num_drug, num_disease))
    mask = np.zeros((num_drug, num_disease))

    for ele in DTItrain:
        drug_disease[ele[1],ele[0]] = ele[2]
        mask[ele[1],ele[0]] = 1
    disease_drug = drug_disease.T

    drug_disease_normalize = row_normalize(drug_disease,False)
    disease_drug_normalize = row_normalize(disease_drug,False)

    lr = 0.001

    best_valid_aupr = 0
    best_valid_auc = 0
    test_aupr = 0
    test_auc = 0

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        for i in range(num_steps):
            _, tloss, dtiloss, results = sess.run([optimizer,total_loss,dti_loss,eval_pred], \
                                        feed_dict={model.drug_drug:drug_drug, model.drug_drug_normalize:drug_drug_normalize,\
                                        model.drug_disease:drug_disease, model.drug_disease_normalize:drug_disease_normalize,\
                                        model.disease_disease:disease_disease, model.disease_disease_normalize:disease_disease_normalize,\
                                        model.disease_drug:disease_drug, model.disease_drug_normalize:disease_drug_normalize,\
                                        model.drug_disease_mask:mask,\
                                        learning_rate: lr})
            #every 25 steps of gradient descent, evaluate the performance, other choices of this number are possible
            if i % 20 == 0 and verbose == True:
                print('step',i,'total and dtiloss',tloss, dtiloss)

                pred_list = []
                ground_truth = []
                for ele in DTIvalid:
                    pred_list.append(results[ele[1],ele[0]])
                    ground_truth.append(ele[2])
                valid_auc = roc_auc_score(ground_truth, pred_list)
                valid_aupr = average_precision_score(ground_truth, pred_list)
                if valid_aupr >= best_valid_aupr:
                    best_valid_aupr = valid_aupr
                    best_valid_auc = valid_auc
                    pred_list = []
                    ground_truth = []
                    for ele in DTItest:
                        pred_list.append(results[ele[1],ele[0]])
                        ground_truth.append(ele[2])
                    test_auc = roc_auc_score(ground_truth, pred_list)
                    test_aupr = average_precision_score(ground_truth, pred_list)
                print('valid auc aupr,', valid_auc, valid_aupr, 'test auc aupr', test_auc, test_aupr)
    return best_valid_auc, best_valid_aupr, test_auc, test_aupr

test_auc_round = []
test_aupr_round = []
for r in range(10):
    print('sample round', r+1)
    if opts.t == 'o':
        dti_o = np.loadtxt(network_path+'DiDrA.txt')
    else:
        dti_o = np.loadtxt(network_path+'DiDrA'+opts.t+'.txt')

    whole_positive_index = []
    whole_negative_index = []
    for i in range(np.shape(dti_o)[0]):
        for j in range(np.shape(dti_o)[1]):
            if int(dti_o[i][j]) == 1:
                whole_positive_index.append([i, j])
            elif int(dti_o[i][j]) == 0:
                whole_negative_index.append([i, j])


    if opts.r == 'ten':
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),size=10*len(whole_positive_index),replace=False)
    elif opts.r == 'all':
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),size=len(whole_negative_index),replace=False)
    else:
        print('wrong positive negative ratio')
        break

    data_set = np.zeros((len(negative_sample_index)+len(whole_positive_index), 3), dtype=int)
    count = 0
    print('np.shape(whole_positive_index)', np.shape(whole_positive_index))
    print('np.shape(data_set)', np.shape(data_set))

    for i in whole_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1
    print(count)
    for i in negative_sample_index:
        data_set[count][0] = whole_negative_index[i][0]
        data_set[count][1] = whole_negative_index[i][1]
        data_set[count][2] = 0
        count += 1
    print('np.shape(data_set)', np.shape(data_set))

    if opts.t == 'unique':
        whole_positive_index_test = []
        whole_negative_index_test = []
        for i in (np.shape(dti_o)[0]):
            for j in range(np.shape(dti_o)[1]):
                if int(dti_o[i][j]) == 3:
                    whole_positive_index_test.append([i, j])
                elif int(dti_o[i][j]) == 2:
                    whole_negative_index_test.append([i, j])

        if opts.r == 'ten':
            negative_sample_index_test = np.random.choice(np.arange(len(whole_negative_index_test)),size=10*len(whole_positive_index_test),replace=False)
        elif opts.r == 'all':
            negative_sample_index_test = np.random.choice(np.arange(len(whole_negative_index_test)),size=whole_negative_index_test,replace=False)
        else:
            print('wrong positive negative ratio')
            break
        data_set_test = np.zeros((len(negative_sample_index_test)+len(whole_positive_index_test),3),dtype=int)
        count = 0
        for i in whole_positive_index_test:
            data_set_test[count][0] = i[0]
            data_set_test[count][1] = i[1]
            data_set_test[count][2] = 1
            count += 1
        for i in negative_sample_index_test:
            data_set_test[count][0] = whole_negative_index_test[i][0]
            data_set_test[count][1] = whole_negative_index_test[i][1]
            data_set_test[count][2] = 0
            count += 1
        DTItrain = data_set
        print('np.shape(DTItrain)', np.shape(DTItrain))
        DTItest = data_set_test
        rs = np.random.randint(0,1000,1)[0]
        DTItrain, DTIvalid =  train_test_split(DTItrain, test_size=0.05, random_state=rs)
        print('np.shape(DTItrain)', np.shape(DTItrain))
        v_auc, v_aupr, t_auc, t_aupr = train_and_evaluate(DTItrain=DTItrain, DTIvalid=DTIvalid, DTItest=DTItest, graph=graph, num_steps=3000)

        test_auc_round.append(t_auc)
        test_aupr_round.append(t_aupr)
        np.savetxt('test_auc', test_auc_round)
        np.savetxt('test_aupr', test_aupr_round)

    else:
        test_auc_fold = []
        test_aupr_fold = []
        rs = np.random.randint(0,1000,1)[0]
        kf = StratifiedKFold(data_set[:,2], n_folds=10, shuffle=True, random_state=rs)

        for train_index, test_index in kf:
            DTItrain, DTItest = data_set[train_index], data_set[test_index]
            DTItrain, DTIvalid =  train_test_split(DTItrain, test_size=0.05, random_state=rs)

            v_auc, v_aupr, t_auc, t_aupr = train_and_evaluate(DTItrain=DTItrain, DTIvalid=DTIvalid, DTItest=DTItest, graph=graph, num_steps=3000)
            test_auc_fold.append(t_auc)
            test_aupr_fold.append(t_aupr)

        test_auc_round.append(np.mean(test_auc_fold))
        test_aupr_round.append(np.mean(test_aupr_fold))
        np.savetxt('test_auc', test_auc_round)
        np.savetxt('test_aupr', test_aupr_round)
