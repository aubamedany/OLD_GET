import time
import os
import torch
import gc
from Models.FCWithEvidences.basic_fc_model import BasicFCModel
import torch_utils
from setting_keywords import KeyWordSettings
from Models.BiDAF.wrapper import GGNN, GGNN_with_GSL, Linear
from thirdparty.two_branches_attention import *
import numpy as np
import torch_utils as my_utils
from scipy.spatial.distance import cosine
import torch.nn as nn
import numpy as np
import torch_utils
from Models import base_model
import time
import json
import interactions
from handlers.tensorboard_writer import TensorboardWrapper
from setting_keywords import KeyWordSettings
from Fitting.FittingFC.multi_level_attention_composite_fitter import MultiLevelAttentionCompositeFitter
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score
import sklearn
import torch.nn.functional as F
from handlers import output_handler, mz_sampler
from torch_geometric.nn.conv import GATConv, GATv2Conv
import torch.optim as optim

class Loss_func():
    def cross_entroy(self,predictions: torch.Tensor, labels: torch.tensor):
        assert predictions.shape[0] == labels.shape[0]
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(F.log_softmax(predictions,dim=1), labels.long())
class PropagationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat_conv1 = GATv2Conv(2, 2, heads=8, edge_dim=1)
        self.bn_gat_conv1 = nn.BatchNorm1d(2 * 8)
        self.gat_conv2 = GATv2Conv(16, 2, heads=8, edge_dim=1)
        self.bn_gat_conv2 = nn.BatchNorm1d(2 * 8)
        self.dropout = nn.Dropout(p=0.4) 
        self.linear = nn.Linear(16,2)
        self.bn_linear = nn.BatchNorm1d(2)
        self.linear.apply(my_utils.init_weights)
    def forward(self,x,edge_index,old_id):
        # x = self.gat_conv(x,edge_index)
        # x = x[old_id.long(), :] 
        # x = self.dropout(x)
        # x = F.relu(x)
        # x = self.linear(x)

        num_node = x.size(0)//11
        x = self.gat_conv1(x,edge_index)
        x = self.bn_gat_conv1(x)
        x = self.gat_conv2(x,edge_index)
        x = self.bn_gat_conv2(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.linear(x)
        x = self.bn_linear(x)
        x = x.view(num_node,11,2)
        x = torch.mean(x,dim=1)
        return x

    def predict(self,x,edge_index,old_id):
        self.train(False)
        x=  self(x,edge_index,old_id)
        # probs = F.softmax(x,dim=-1)
        return x
class FittingModel():
    def __init__(self, net: PropagationModel,batch_size:int,n_iter:int,idx_fold:int,threshold:float,data="snope",use_cuda=0,logfolder=None,seed=None,early_stopping= 10,**kargs):
                
        # super(FittingModel, self).__init__(net,batch_size,n_iter,idx_fold,threshold,data="snope",**kargs)
        self.output_handler = kargs["output_handler_fact_checking"]
        self._net = net
        self._batch_size = batch_size
        self._n_iter = n_iter
        self.idx_fold = idx_fold
        self._threshold = threshold
        self.name_dir = "embed_probagation/"
        self._testing_epochs = 5
        self.data = data
        self._sampler = mz_sampler.Sampler()
        self.output_size = 2
        self._use_cuda = use_cuda
        self.logfolder = logfolder
        self.saved_model = os.path.join(logfolder, "saved_model_%s" % seed)
        self._early_stopping_patience = early_stopping
        self._optimizer = optim.Adam(
            self._net.parameters(),
            weight_decay = 0.001,
            lr = 0.001)
        self._loss_func = Loss_func()
        if use_cuda:
            self._net = self._net.to("cuda:0")


        self.output_handler.myprint("Using: " + str(self._loss_func))
        self.train_outputs = my_utils.gpu(torch.load(f'{self.name_dir}output_{self.data}_train_{self.idx_fold}.pt'), self._use_cuda)
        self.train_logits = my_utils.gpu(torch.load(f'{self.name_dir}logit_{self.data}_train_{self.idx_fold}.pt'),self._use_cuda)
        self.valid_outputs = my_utils.gpu(torch.load(f'{self.name_dir}output_{self.data}_valid.pt'), self._use_cuda)
        self.valid_logits = my_utils.gpu(torch.load(f'{self.name_dir}logit_{self.data}_valid.pt'), self._use_cuda)
        self.test_outputs = my_utils.gpu(torch.load(f'{self.name_dir}output_{self.data}_test_{self.idx_fold}.pt'), self._use_cuda)
        self.test_logits = my_utils.gpu(torch.load(f'{self.name_dir}logit_{self.data}_test_{self.idx_fold}.pt'), self._use_cuda)

        self.train_similarity_dict = torch.load(f'{self.name_dir}similar_dict_{self.data}_train_{self.idx_fold}.pt')
        self.test_similarity_dict = torch.load(f'{self.name_dir}similar_dict_{self.data}_test_{self.idx_fold}.pt')
        self.valid_similarity_dict = torch.load(f'{self.name_dir}similar_dict_{self.data}_valid.pt')
    def fit(self,
            train_interactions: interactions.ClassificationInteractions,
            verbose=True,  # for printing out evaluation during training
            topN = 10,
            val_interactions: interactions.ClassificationInteractions = None, 
            test_interactions: interactions.ClassificationInteractions = None
            ):
        
        best_val_f1_macro, best_epoch = 0, 0
        test_results_dict = None
        iteration_counter = 0
        count_patience_epochs = 0

        
        for epoch_num in range(self._n_iter):
            # query_ids, left_contents, left_lengths, query_sources, query_char_sources, query_adj, \
            # evd_docs_ids, evd_docs_contents, evd_docs_lens, evd_sources, evd_cnt_each_query, evd_char_sources, \
            # pair_labels, evd_docs_adj = self._sampler.get(train_interactions,100)
            self._net.train(True)
            query_ids,pair_labels = self._sampler.get_instance_new(train_interactions,100)
            # del left_contents, left_lengths, query_sources, query_char_sources, query_adj, \
            # evd_docs_ids, evd_docs_contents, evd_docs_lens, evd_sources, evd_cnt_each_query, evd_char_sources, evd_docs_adj
            # gc.collect()
            # torch.cuda.empty_cache() 
            query_2_vector = {} 
            for i,id in enumerate(query_ids):
                query_2_vector[id] = i
            id_b = np.array([query_2_vector[i] for i in query_ids ])
            id_b,query_ids, pair_labels = my_utils.shuffle(id_b,query_ids, pair_labels)
            # all_labels = pair_labels.flatten().astype(int).tolist()

            # pair_labels = pair_labels.tolist()
            epoch_loss, total_pairs = 0.0, 0
            t1 = time.time()
            for (minibatch_num,
                    (batch_id,batch_label)) \
                    in enumerate(my_utils.minibatch(id_b,pair_labels, batch_size=self._batch_size)):
                batch_label = my_utils.gpu(torch.from_numpy(batch_label), self._use_cuda)
                batch_id = my_utils.gpu(torch.from_numpy(batch_id), self._use_cuda)
                # self._optimizer.zero_grad()

                loss = self._get_prediction("train",batch_id,batch_label)
                epoch_loss += loss.item()
                # print(loss.item())
                iteration_counter += 1
                # if iteration_counter % 2 == 0: break
                # TensorboardWrapper.mywriter().add_scalar("loss/minibatch_loss", loss.item(), iteration_counter)
                loss.backward()


                # grad_norm = 0
                # for p in self._net.parameters():
                #     if p.grad is not None:
                #         grad_norm+=(p.grad.view(-1) ** 2).sum()
                # grad_norm = torch.sqrt(grad_norm)
                # grad_norm = torch.sqrt(sum([torch.norm(p.grad)**2 for p in self._net.parameters()]))

                # Print the gradient norm
                # print("Gradient Norm:", grad_norm.item())
                self._optimizer.step()

                # TensorboardWrapper.mywriter().add_scalar("loss/epoch_loss_avg", epoch_loss, epoch_num)
                # print("Number of Minibatches: ", minibatch_num, "Avg. loss of epoch: ", epoch_loss)
                t2 = time.time()
                epoch_train_time = t2 - t1
            if verbose:  # validation after each epoch
                # print("Evaluating")
                # f1_macro_val = self._output_results_every_epoch( val_interactions, test_interactions,topN, epoch_num, epoch_train_time, epoch_loss)
                auc_val = self._output_results_every_epoch( val_interactions, test_interactions,topN, epoch_num, epoch_train_time, epoch_loss)
                if auc_val > best_val_f1_macro :
                    # if (hits + ndcg) > (best_hit + best_ndcg):
                    count_patience_epochs = 0
                    with open(self.saved_model, "wb") as f:
                        torch.save(self._net.state_dict(), f)
                    # test_results_dict = result_test
                    best_val_f1_macro, best_epoch = auc_val, epoch_num
                    # test_hit, test_ndcg = hits_test, ndcg_test
                else:
                    count_patience_epochs += 1
                if self._early_stopping_patience and count_patience_epochs > self._early_stopping_patience:
                    self.output_handler.myprint(
                        "Early Stopped due to no better performance in %s epochs" % count_patience_epochs)
                    break
            # self._net.train(False)
            # id_test = torch.tensor([0,1])
            # id_test = my_utils.gpu(id_test, self._use_cuda)
            # old_id,x,edge_index = self.get_graph("train",id_test,self._threshold)
            # probs = self._net(x,edge_index,old_id)
            # print(probs)
            if np.isnan(epoch_loss) or epoch_loss == 0.0:
                raise ValueError('Degenerate epoch loss: {}'.format(epoch_loss))
        self._flush_training_results(best_val_f1_macro, best_epoch)
    def _flush_training_results(self, best_val_f1_macro: float, best_epoch: int):
        self.output_handler.myprint("Closing tensorboard")
        self.output_handler.myprint('Best result: | vad F1_macro = %.5f | epoch = %d' % (best_val_f1_macro, best_epoch))
    def _output_results_every_epoch(self,val_interactions: interactions.ClassificationInteractions,
                                    test_interactions: interactions.ClassificationInteractions, topN: int,epoch_num: int, epoch_train_time: float, epoch_loss: float):
        t1 = time.time()
        result_val,loss_val = self.evaluate(testRatings=val_interactions,K=topN,type_set="valid")
        auc_val = result_val[KeyWordSettings.AUC_metric]
        f1_macro_val = result_val[KeyWordSettings.F1_macro]
        f1_micro_val = result_val[KeyWordSettings.F1_micro]
        # f1_val = result_val[KeyWordSettings.F1]
        # ndcg_val = result_val["ndcg"]
        t2 = time.time()
        valiation_time = t2 - t1

        if epoch_num and epoch_num % self._testing_epochs == 0:
            t1 = time.time()
            result_test,_ = self.evaluate(testRatings=test_interactions,K=topN,type_set="test")
            # auc_test = result_test[KeyWordSettings.AUC_metric]
            f1_macro_test = result_test[KeyWordSettings.F1_macro]
            f1_micro_test = result_test[KeyWordSettings.F1_micro]
            # f1_test = result_test[KeyWordSettings.F1]
            # ndcg_test = result_test["ndcg"]
            t2 = time.time()
            testing_time = t2 - t1
            # TensorboardWrapper.mywriter().add_scalar("auc/auc_test", auc_test, epoch_num)
            # TensorboardWrapper.mywriter().add_scalar("f1/f1_macro_test", f1_macro_test, epoch_num)
            # TensorboardWrapper.mywriter().add_scalar("f1/f1_micro_test", f1_micro_test, epoch_num)
            # TensorboardWrapper.mywriter().add_scalar("f1/f1_test", f1_test, epoch_num)

            # TensorboardWrapper.mywriter().add_scalar("ndcg/ndcg_test", ndcg_test, epoch_num)
            self.output_handler.myprint('|Epoch %03d | Test F1_macro = %.5f | Testing time: %04.1f(s)'
                                        % (epoch_num, f1_macro_test, testing_time))

        # TensorboardWrapper.mywriter().add_scalar("auc/auc_val", auc_val, epoch_num)
        # TensorboardWrapper.mywriter().add_scalar("f1/f1_macro_val", f1_macro_val, epoch_num)
        # TensorboardWrapper.mywriter().add_scalar("f1/f1_micro_val", f1_micro_val, epoch_num)
        # TensorboardWrapper.mywriter().add_scalar("f1/f1_val", f1_val, epoch_num)

        # TensorboardWrapper.mywriter().add_scalar("ndcg/ndcg_val", ndcg, epoch_num)
        self.output_handler.myprint('|Epoch %03d | Train time: %04.1f(s) | Train loss: %.3f| Val loss: %.3f'
                                    '| Val F1_macro = %.3f | Vad AUC = %.3f'
                                    '| Val F1_micro = %.3f | Validation time: %04.1f(s)'
                                    % (epoch_num, epoch_train_time, epoch_loss,loss_val, f1_macro_val, auc_val,
                                       f1_micro_val, valiation_time))
        return auc_val
    def evaluate(self,testRatings: interactions.ClassificationInteractions, K: int, output_ranking=False, type_set="valid",**kargs):

        
        all_final_preds = []
        all_final_probs = []
        list_error_analysis = []
        # query_ids, left_contents, left_lengths, query_sources, query_char_sources, query_adj, \
        #     evd_docs_ids, evd_docs_contents, evd_docs_lens, evd_sources, evd_cnt_each_query, evd_char_sources, \
        #     pair_labels, evd_docs_adj = self._sampler.get_train_instances_char_man(testRatings,100)
        
        query_ids, pair_labels = self._sampler.get_instance_new(testRatings,100)
        query_2_vector = {} 
        for i,id in enumerate(query_ids):
            query_2_vector[id] = i
        id_b = np.array([query_2_vector[i] for i in query_ids ])
        id_b,query_ids, pair_labels = my_utils.shuffle(id_b,query_ids, pair_labels)
        all_labels = pair_labels.flatten().astype(int).tolist()

        # pair_labels = pair_labels.tolist()

        epoch_loss = 0.0 
        for (minibatch_num,
                (batch_id,batch_label)) \
                in enumerate(my_utils.minibatch(id_b,pair_labels, batch_size=self._batch_size)):
        # padded_evd_contents = self._pad_evidences(evd_contents) # 1, 30, R
            # new_batch =  self.get_new_batch(type_set,batch_id)
            batch_label = my_utils.gpu(torch.from_numpy(batch_label), self._use_cuda)
            batch_id = my_utils.gpu(torch.from_numpy(batch_id), self._use_cuda)
            old_id,x,edge_index = self.get_graph(type_set,batch_id,self._threshold)
            probs = self._net.predict(x,edge_index,old_id)

            predictions = probs.argmax(dim=1)
            loss = self._loss_func.cross_entroy(probs, batch_label.float())
            epoch_loss += loss.item()
            # all_final_preds.append(float(my_utils.cpu(predictions).detach().numpy().flatten()))
            # all_final_probs.append(float(my_utils.cpu(probs[:, 1]).detach().numpy().flatten()))
            all_final_preds = all_final_preds + predictions.cpu().detach().numpy().flatten().astype(int).tolist()
            all_final_probs = all_final_probs + probs[:, 1].cpu().detach().numpy().flatten().tolist()

        results = self._computing_metrics(true_labels=all_labels, predicted_labels=all_final_preds, predicted_probs=all_final_probs)
        # print(f'predicted_probs{all_final_probs}')
        # print(f'predicted_labels: {all_final_preds}')
        # print(f'true_labels:{all_labels}')
        if output_ranking: return results, list_error_analysis  # sorted(list_error_analysis, key=lambda x: x["qid"])
        return results, epoch_loss

    def _computing_metrics(self, true_labels: List[int], predicted_labels: List[float], predicted_probs: List[float]):
        """
        Computing classifiction metrics for 3 category classification
        Parameters
        ----------
        true_labels: ground truth
        predicted_labels: predicted labels

        Returns
        -------

        """
        # print(f'Predicted {predicted_labels}')
        # print(f'TrueLabel {true_labels}')
        assert len(true_labels) == len(predicted_labels)
        results = {}

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(true_labels, predicted_probs, pos_label=1)
        auc = sklearn.metrics.auc(fpr, tpr)
        f1_macro = f1_score(true_labels, predicted_labels, average='macro')
        f1_micro = f1_score(true_labels, predicted_labels, average='micro')
        f1 = f1_score(true_labels, predicted_labels)

        # this is the normal precision and recall we seen so many times
        precision_true_class = precision_score(true_labels, predicted_labels, labels=[1], average=None)[0]
        recall_true_class = recall_score(true_labels, predicted_labels, labels=[1], average=None)[0]
        f1_true_class = f1_score(true_labels, predicted_labels, labels=[1], average=None)[0]

        precision_false_class = precision_score(true_labels, predicted_labels, labels=[0], average=None)[0]
        recall_false_class = recall_score(true_labels, predicted_labels, labels=[0], average=None)[0]
        f1_false_class = f1_score(true_labels, predicted_labels, labels=[0], average=None)[0]

        precision_mixed_class = precision_score(true_labels, predicted_labels, labels=[2], average=None)[0] \
            if self.output_size == 3 else 0.0
        recall_mixed_class = recall_score(true_labels, predicted_labels, labels=[2], average=None)[0] \
            if self.output_size == 3 else 0.0
        f1_mixed_class = f1_score(true_labels, predicted_labels, labels=[2], average=None)[0] \
            if self.output_size == 3 else 0.0

        results[KeyWordSettings.AUC_metric] = auc
        results[KeyWordSettings.F1_macro] = f1_macro
        results[KeyWordSettings.F1_micro] = f1_micro
        results[KeyWordSettings.F1] = f1

        results[KeyWordSettings.PrecisionTrueCls] = precision_true_class
        results[KeyWordSettings.RecallTrueCls] = recall_true_class
        results[KeyWordSettings.F1TrueCls] = f1_true_class  # this must be normal F1

        results[KeyWordSettings.PrecisionFalseCls] = precision_false_class
        results[KeyWordSettings.RecallFalseCls] = recall_false_class
        results[KeyWordSettings.F1FalseCls] = f1_false_class

        results[KeyWordSettings.PrecisionMixedCls] = precision_mixed_class
        results[KeyWordSettings.RecallMixedCls] = recall_mixed_class
        results[KeyWordSettings.F1MixedCls] = f1_mixed_class

        return results
    def load_best_model(self, val_interactions: interactions.ClassificationInteractions,
                        test_interactions: interactions.ClassificationInteractions, topN: int = 10):
        mymodel = self._net
        # print("Trained model: ", mymodel.out.weight)
        mymodel.load_state_dict(torch.load(self.saved_model))
        mymodel.train(False)
        my_utils.gpu(mymodel, self._use_cuda)

        
        result_val,loss_val = self.evaluate(val_interactions,topN,type_set="valid")
        auc_val = result_val[KeyWordSettings.AUC_metric]
        f1_val = result_val[KeyWordSettings.F1]
        f1_macro_val = result_val[KeyWordSettings.F1_macro]
        f1_micro_val = result_val[KeyWordSettings.F1_micro]

        assert len(test_interactions.dict_claims_and_evidences_test) in KeyWordSettings.ClaimCountTest
        result_test,loss_test = self.evaluate(test_interactions,topN,type_set="test")
        auc_test = result_test[KeyWordSettings.AUC_metric]
        f1_test = result_test[KeyWordSettings.F1]
        f1_macro = result_test[KeyWordSettings.F1_macro]
        f1_micro = result_test[KeyWordSettings.F1_micro]
        precision_true_class = result_test[KeyWordSettings.PrecisionTrueCls]
        recall_true_class = result_test[KeyWordSettings.RecallTrueCls]
        f1_true_class = result_test[KeyWordSettings.F1TrueCls]

        precision_false_class = result_test[KeyWordSettings.PrecisionFalseCls]
        recall_false_class = result_test[KeyWordSettings.RecallFalseCls]
        f1_false_class = result_test[KeyWordSettings.F1FalseCls]

        precision_mixed_class = result_test[KeyWordSettings.PrecisionMixedCls]
        recall_mixed_class = result_test[KeyWordSettings.RecallMixedCls]
        f1_mixed_class = result_test[KeyWordSettings.F1MixedCls]

        print(auc_val, auc_test)
        # self.output_handler.save_error_analysis_validation(json.dumps(error_analysis_val, sort_keys=True, indent=2))
        # self.output_handler.save_error_analysis_testing(json.dumps(error_analysis_test, sort_keys=True, indent=2))
        self.output_handler.myprint('Best Vad F1_macro = %.5f | Best Vad AUC = %.5f'
                                    '| Best Test F1_macro = %.5f | Best Test F1_micro = %.5f | Best Vad AUC = %.5f \n'
                                    '| Best Test Precision_True_class = %.5f | Best Test Recall_True_class = %.5f '
                                    '| Best Test F1_True_class = %.5f \n'
                                    '| Best Test Precision_False_class = %.5f | Best Test_Recall_False class = %.5f '
                                    '| Best Test F1_False_class = %.5f \n'
                                    '| Best Test Precision_Mixed_class = %.5f | Best Test_Recall_Mixed_class = %.5f '
                                    '| Best Test F1_Mixed_class = %.5f '
                                    % (f1_macro_val, auc_val, f1_macro, f1_micro, auc_test,
                                       precision_true_class, recall_true_class, f1_true_class,
                                       precision_false_class, recall_false_class, f1_false_class,
                                       precision_mixed_class, recall_mixed_class, f1_mixed_class))

        return result_val, result_test
    def _get_prediction(self,type_set,batch_id,batch_label):
        # new_batch =  self.get_new_batch(type_set,batch_id)
        old_id,x,edge_index = self.get_graph(type_set,batch_id,self._threshold)
        logits = self._net(x,edge_index,old_id)
        loss=self._loss_func.cross_entroy(logits, batch_label.float())
        return loss
    def get_new_batch(self,type_set,batch_id):
        batch_id = batch_id.tolist()
        old_id = []
        new_batch =[]
        if type_set == 'train':
            similarity_dict = self.train_similarity_dict
        elif type_set == 'test':
            similarity_dict = self.test_similarity_dict
        else :
            similarity_dict = self.valid_similarity_dict
        for i,id in enumerate(batch_id): #batch_id.: id of vector
            old_id.append(len(new_batch))
            new_batch.append(id)
            new_batch = new_batch + similarity_dict[id][:10]
        return old_id,new_batch
    def get_graph(self,type_set,batch_id,threshold):
        old_id,new_batch_id =  self.get_new_batch(type_set,batch_id)
        if type_set == 'train':
            outputs = self.train_outputs
            logits = self.train_logits
        elif type_set == 'test':
            outputs = self.test_outputs
            logits = self.test_logits
        else:
            outputs = self.valid_outputs
            logits = self.valid_logits
        output = outputs[new_batch_id]
        logit = logits[new_batch_id]
        # adjacency_matrix = np.zeros((len(new_batch_id), len(new_batch_id)))
        # for i in range(len(new_batch_id)):
        #     for j in range(len(new_batch_id)):
        #         # adjacency_matrix[i][j] =   1 - cosine(output[i], output[j]) 
        #         adjacency_matrix[i][j] = torch.nn.functional.cosine_similarity(output[i], output[j], dim=0)
        # edge_index = []
        # for i in range(len(new_batch_id)):
        #     for j in range(len(new_batch_id)):
        #         if adjacency_matrix[i,j] >threshold:
        #             edge_index.append([i,j])
        output_norm = torch.nn.functional.normalize(output, dim=1)
        adjacency_matrix = torch.matmul(output_norm,output_norm.transpose(0, 1))
        edge_index = (adjacency_matrix > threshold).nonzero().t()

        old_id = my_utils.gpu(torch.tensor(old_id), self._use_cuda)
        logit = my_utils.gpu(logit, self._use_cuda)
        # edge = torch.tensor(np.array(edge_index).transpose()).to(torch.int)
        edge_index = my_utils.gpu(edge_index, self._use_cuda)
        return old_id,logit,edge_index
    