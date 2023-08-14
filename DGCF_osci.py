import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
from utility.helper import *
from utility.batch_test import *
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

tf.random.get_seed(1)

class DGCF(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings
        self.model_type = args.model_type
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type

        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.n_fold = 100

        self.norm_adj = data_config['norm_adj']
        self.cross_adj = data_config['cross_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)

        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.local_factor = args.local_factor

        self.verbose = args.verbose

        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        # dropout: node dropout (adopted on the ego-networks);
        #          ... since the usage of node dropout have higher computational cost,
        #          ... please use the 'node_dropout_flag' to indicate whether use such technique.
        #          message dropout (adopted on the convolution operations).
        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._init_weights()

        """
        *********************************************************
        Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
        Different Convolutional Layers:
            1. ngcf: defined in 'Neural Graph Collaborative Filtering', SIGIR2019;
            2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
            3. gcmc: defined in 'Graph Convolutional Matrix Completion', KDD2018;
        """
        if self.alg_type in ['lightgcn']:
            self.ua_embeddings, self.ia_embeddings = self._create_lightgcn_embed()

        elif self.alg_type in ['dgcf']:
            self.ua_embeddings, self.ia_embeddings = self._create_dgcf_embed()

        elif self.alg_type in ['dgcf-wochp']:
            self.ua_embeddings, self.ia_embeddings = self._create_dgcf_embed_wochp()

        elif self.alg_type in ['dgcf-wola']:
            self.ua_embeddings, self.ia_embeddings = self._create_dgcf_embed_wola()

        elif self.alg_type in ['dgcf_final']: # only keep the last layers
            print('using final embeddings ')
            self.ua_embeddings, self.ia_embeddings = self._create_dgcf_embed_final()    

        elif self.alg_type in ['gcn']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcn_embed()

        elif self.alg_type in ['gcmc']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcmc_embed()

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)
        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)

        """
        *********************************************************
        Inference for the testing phase.
        """
        self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)

        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss = self.mf_loss + self.emb_loss # + self.reg_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
#         self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _init_weights(self):
        all_weights = dict()

        initializer = initializer = tf.keras.initializers.glorot_uniform()

        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
            print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'][:,0:self.emb_dim], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'][:,0:self.emb_dim], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            print('using pretrained embeddings')

        self.weight_size_list = [self.emb_dim] + self.weight_size
        for k in range(self.n_layers):
            all_weights['locality_%d' %k] = tf.Variable(initializer([self.n_users + self.n_items, 1]), name='locality_%d' %k)

        return all_weights

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _create_dgcf_embed(self):
        # Generate a set of adjacency sub-matrix.
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj + self.cross_adj)
            del self.norm_adj, self.cross_adj
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj + self.cross_adj)
            del self.norm_adj, self.cross_adj

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings =  ego_embeddings / (self.n_layers+1)

        for k in range(0, self.n_layers):

            temp_embed = []
            residual_embed = []
            ego_embeddings = tf.multiply(tf.nn.sigmoid(self.weights['locality_%d' % k]), ego_embeddings)
            for f in range(self.n_fold): # save memory
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings)) # sum messages of neighbors.
            ego_embeddings = tf.concat(temp_embed, 0)
            all_embeddings += ego_embeddings / (self.n_layers+1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)

        return u_g_embeddings, i_g_embeddings

    def _create_dgcf_embed_wochp(self):
        # Generate a set of adjacency sub-matrix.
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
            del self.norm_adj, self.cross_adj
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)
            del self.norm_adj, self.cross_adj

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings =  ego_embeddings / (self.n_layers+1)

        for k in range(0, self.n_layers):

            temp_embed = []
            residual_embed = []
            ego_embeddings = tf.multiply(tf.nn.sigmoid(self.weights['locality_%d' % k]), ego_embeddings)
            for f in range(self.n_fold): # save memory
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings)) # sum messages of neighbors.
            ego_embeddings = tf.concat(temp_embed, 0)
            all_embeddings += ego_embeddings / (self.n_layers+1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)

        return u_g_embeddings, i_g_embeddings

    def _create_dgcf_embed_wola(self):
        # Generate a set of adjacency sub-matrix.
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
            del self.norm_adj, self.cross_adj
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)
            del self.norm_adj, self.cross_adj

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings =  ego_embeddings / (self.n_layers+1)

        for k in range(0, self.n_layers):

            temp_embed = []
            residual_embed = []
            # ego_embeddings = tf.multiply(tf.nn.sigmoid(self.weights['locality_%d' % k]), ego_embeddings)
            for f in range(self.n_fold): # save memory
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings)) # sum messages of neighbors.
            ego_embeddings = tf.concat(temp_embed, 0)
            all_embeddings += ego_embeddings / (self.n_layers+1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)

        return u_g_embeddings, i_g_embeddings


    def _create_dgcf_embed_final(self):
        # Generate a set of adjacency sub-matrix.
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj + self.cross_adj)
            del self.norm_adj, self.cross_adj
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj + self.cross_adj)
            del self.norm_adj, self.cross_adj

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings =  ego_embeddings / (self.n_layers+1)

        for k in range(0, self.n_layers):

            temp_embed = []
            residual_embed = []
            ego_embeddings = tf.multiply(tf.nn.sigmoid(self.weights['locality_%d' % k]), ego_embeddings)
            for f in range(self.n_fold): # save memory
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings)) # sum messages of neighbors.
            ego_embeddings = tf.concat(temp_embed, 0)
            # all_embeddings += ego_embeddings / (self.n_layers+1)
        u_g_embeddings, i_g_embeddings = tf.split(ego_embeddings, [self.n_users, self.n_items], 0)

        return u_g_embeddings, i_g_embeddings


    def _create_lightgcn_embed(self):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
            del self.norm_adj
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)
            del self.norm_adj
        
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        
        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings=tf.stack(all_embeddings,1)
        all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings


    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        
        regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(
                self.pos_i_g_embeddings_pre) + tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        regularizer = regularizer / self.batch_size
        
        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        

        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)

def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embeddings')
    print(pretrain_path)
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data
