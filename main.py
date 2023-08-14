import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import sys
from DGCF_osci import DGCF, load_pretrained_data
from utility.helper import *
from utility.batch_test import *
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

tf.random.get_seed(1)





if __name__ == '__main__':

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, norm_adj_no_eye, cross_adj = data_generator.get_adj_mat(args.low, args.high)
    # cross_adj.setdiag(0.0)
    # cross_adj.eliminate_zeros()

    if args.adj_type == 'plain':
        config['norm_adj'] = plain_adj
        print('use the plain adjacency matrix')

    elif args.adj_type == 'laplacian':
        config['norm_adj'] = norm_adj
        print('use the laplacian adjacency matrix')

    elif args.adj_type == 'laplacian_no_eye':
        config['norm_adj'] = norm_adj_no_eye
        print('use the laplacian adjacency matrix without eye')

    else:
        config['norm_adj'] = norm_adj + sp.eye(norm_adj.shape[0])
        print('use the mean adjacency matrix')

    config['cross_adj'] = cross_adj
    t0 = time()

    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None

    model = DGCF(data_config=config, pretrain_data=pretrain_data)
    del config, cross_adj

    """
    *********************************************************
    Save the model parameters.
    """
    saver = tf.train.Saver()
    layer = '-'.join([str(l) for l in eval(args.layer_size)])

    if args.save_flag == 1:
        weights_save_path = '%sweights/%s/%s/%s/l%s_r%s_d%s_low%s_high%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]), str(eval(args.node_dropout)[0]),
                                                            str(args.low), str(args.high))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    """
    *********************************************************
    Reload the pretrained model parameters.
    """
    if args.pretrain == 1:
        pretrain_path = '%sweights/%s/%s/%s/l%s_r%s_d%s_low%s_high%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]), str(eval(args.node_dropout)[0]),
                                                            str(args.low), str(args.high))

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load the pretrained model parameters from: ', pretrain_path)

            # *********************************************************
            # get the performance from pretrained model.
            if args.report != 1:
                users_to_test = list(data_generator.test_set.keys())
                ret = test(sess, model, users_to_test, drop_flag=True)
                cur_best_pre_0 = ret['recall'][0]

                pretrain_ret = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]"% \
                 ('\t'.join(['%.5f' % r for r in ret['recall']]),
                  '\t'.join(['%.5f' % r for r in ret['precision']]),
                  '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                  '\t'.join(['%.5f' % r for r in ret['ndcg']]))
                print(pretrain_ret)
        else:
            sess.run(tf.global_variables_initializer())
            cur_best_pre_0 = 0.
            print('without pretraining.')

    else:
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        print('without pretraining weights.')

    """
    *********************************************************
    Get the performance w.r.t. different sparsity levels.
    """
    if args.report == 1:
        assert args.test_flag == 'full'
        users_to_test_list, split_state = data_generator.get_sparsity_split()
        users_to_test_list.append(list(data_generator.test_set.keys()))
        split_state.append('all')

        report_path = '%sreport/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
        ensureDir(report_path)
        f = open(report_path, 'w')
        f.write(
            'embed_size=%d, lr=%.4f, layer_size=%s, keep_prob=%s, regs=%s, loss_type=%s, adj_type=%s\n'
            % (args.embed_size, args.lr, args.layer_size, args.keep_prob, args.regs, args.loss_type, args.adj_type))

        for i, users_to_test in enumerate(users_to_test_list):
            ret = test(sess, model, users_to_test, drop_flag=True)

            final_perf = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                         ('\t'.join(['%.5f' % r for r in ret['recall']]),
                          '\t'.join(['%.5f' % r for r in ret['precision']]),
                          '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                          '\t'.join(['%.5f' % r for r in ret['ndcg']]))
            print(final_perf)

            f.write('\t%s\n\t%s\n' % (split_state[i], final_perf))
        f.close()
        exit()

    """
    *********************************************************
    Train.
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    time_loger = []
    stopping_step = 0
    should_stop = False

    for epoch in range(args.epoch):
        t1 = time()
        loss = 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            _, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss = sess.run([model.opt, model.loss],
                               feed_dict={model.users: users, model.pos_items: pos_items,
                                          model.node_dropout: eval(args.node_dropout),
                                          model.mess_dropout: eval(args.mess_dropout),
                                          model.neg_items: neg_items})
            loss += batch_loss/n_batch
            #mf_loss += batch_mf_loss/n_batch
            #emb_loss += batch_emb_loss/n_batch
            #reg_loss += batch_reg_loss/n_batch

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        # print the validation metrics each 10 epochs; pos:neg = 1:10.
        test_epoch = args.test_epoch
        loss_loger.append(loss)
        if epoch  % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f]' % (
                    epoch, time() - t1, loss)
                print(perf_str)
            continue
        perf_str = 'Epoch %d [%.1fs]: train==[%.5f]' % (
                    epoch, time() - t1, loss)
        print(perf_str)
        t2 = time()
        time_loger.append(time() - t1)
        print("---------start validation--------------")
        users_to_valid = list(set(data_generator.train_items.keys()).intersection(set(data_generator.valid_set.keys())))
        ret = validate(sess, model, users_to_valid, drop_flag=True)

        t3 = time()

        
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]"% \
                 ('\t'.join(['%.5f' % r for r in ret['recall']]),
                  '\t'.join(['%.5f' % r for r in ret['precision']]),
                  '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                  '\t'.join(['%.5f' % r for r in ret['ndcg']]))
            print(perf_str)
            

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=args.stop_step)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            print('---------find the best validation------------')
            # print("---------start testing--------------")
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            print('save the weights in path: ', weights_save_path)
            perf_str = "-----Best!-----recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]-------Best!------"% \
                 ('\t'.join(['%.5f' % r for r in ret['recall']]),
                  '\t'.join(['%.5f' % r for r in ret['precision']]),
                  '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                  '\t'.join(['%.5f' % r for r in ret['ndcg']]))
            u_embed, i_embed = sess.run([model.u_g_embeddings, model.pos_i_g_embeddings], 
                        feed_dict={model.users:list(range(model.n_users)), 
                        model.pos_items:list(range(model.n_items)), model.node_dropout: [0.]*len(eval(args.layer_size)),
                                                                    model.mess_dropout: [0.]*len(eval(args.layer_size))})
            for k in range(model.n_layers): # store the locality weight for checking
                locality_weights = sess.run(model.weights['locality_%d' %k], feed_dict={model.users:list(range(1)), 
                        model.pos_items:list(range(1)), model.node_dropout: [0.]*len(eval(args.layer_size)),
                                                                    model.mess_dropout: [0.]*len(eval(args.layer_size))})
                locality_file_name = weights_save_path + '/locality_%d.npz'%k
                np.savez(locality_file_name, locality_weights=locality_weights)
                print('save the locality weights at layer %d' %k)
            embed_file_name = weights_save_path + '/embeddings.npz'
            np.savez(embed_file_name, user_embed=u_embed, item_embed=i_embed)
            print('save the embeddings in path:',embed_file_name)
            print(perf_str)
    # begin to test
    print('------------start testing!------------')
    if args.save_flag == 0:
        weights_save_path = '%sweights/%s/%s/%s/l%s_r%s_d%s_low%s_high%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]), str(eval(args.node_dropout)[0]),
                                                            str(args.low), str(args.high))
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(weights_save_path + '/checkpoint'))
    saver.restore(sess, ckpt.model_checkpoint_path)
    users_to_test = list(set(data_generator.train_items.keys()).intersection(set(data_generator.test_set.keys())))
    ret = test(sess, model, users_to_test, drop_flag=True)
    final_perf = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]"% \
         ('\t'.join(['%.5f' % r for r in ret['recall']]),
          '\t'.join(['%.5f' % r for r in ret['precision']]),
          '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
          '\t'.join(['%.5f' % r for r in ret['ndcg']]))
    print(final_perf)

    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')
    f.write(
        'embed_size=%d, lr=%.4f, layer_size=%s, , low=%.4f, high=%.4f, node_dropout=%s, mess_dropout=%s, regs=%s, adj_type=%s\n\t%s\n'
        % (args.embed_size, args.lr, args.layer_size, args.low, args.high, args.node_dropout, args.mess_dropout, args.regs,
           args.adj_type, final_perf))
    f.close()
    out_path = '%strain/%s/%s.train%s' % (args.proj_path, args.dataset, model.model_type, time())
    ensureDir(out_path)
    with open(out_path, 'a') as o_f:
        o_f.write('loss:')
        loss_string = str(loss_loger)
        o_f.write(loss_string)
        o_f.write('\n')
        o_f.write('recall:')
        recall_string = str(rec_loger)
        o_f.write(recall_string)
        o_f.write('\n')
        o_f.write('time:')
        time_string = str(time_loger)
        o_f.write(time_string)
        o_f.write('\n')
