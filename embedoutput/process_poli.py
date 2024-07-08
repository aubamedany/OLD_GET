import sys
sys.path.insert(0, '../GET')
sys.path.insert(0, '../../GET')

from Models.FCWithEvidences import graph_based_semantic_structure
from Fitting.FittingFC import char_man_fitter_query_repr1
import time
import json
from interactions import ClassificationInteractions
import matchzoo as mz
from handlers import cls_load_data
import argparse
import random
import numpy as np
import torch
import torch_utils
import numpy as np
import os
import datetime
from handlers.output_handler_FC import FileHandlerFC
from Evaluation import mzEvaluator as evaluator
from setting_keywords import KeyWordSettings
from matchzoo.embedding import entity_embedding
from Models.BiDAF.wrapper import GGNN, GGNN_with_GSL, Linear
from thirdparty.two_branches_attention import *
cuda = False
import interactions
import torch_utils as my_utils
import gc

def get_energy(interaction,fit_model):
      
  query_ids, left_contents, left_lengths, query_sources, query_char_sources, query_adj, \
            evd_docs_ids, evd_docs_contents, evd_docs_lens, evd_sources, evd_cnt_each_query, evd_char_sources, \
            pair_labels, evd_docs_adj = fit_model._sampler.get_train_instances_char_man(interaction,fit_model.fixed_num_evidences)

  logits = torch.tensor([])
  outputs = torch.tensor([])
  E = torch.tensor([])
  with torch.no_grad():

    for (minibatch_num,
                  (batch_query, batch_query_content, batch_query_len, batch_query_sources, batch_query_chr_src,
                    batch_query_adj, batch_evd_docs, batch_evd_contents, batch_evd_lens, batch_evd_sources,
                    # i.e. claim source
                    batch_evd_cnt_each_query, batch_evd_chr_src, batch_labels, batch_evd_docs_adj)) \
                      in enumerate(my_utils.minibatch(query_ids, left_contents, left_lengths, query_sources,
                                                      query_char_sources, query_adj,
                                                      evd_docs_ids, evd_docs_contents, evd_docs_lens, evd_sources,
                                                      evd_cnt_each_query, evd_char_sources, pair_labels, evd_docs_adj,
                                                      batch_size=32)):
        batch_query = my_utils.gpu(torch.from_numpy(batch_query), cuda)
        batch_query_content = my_utils.gpu(torch.from_numpy(batch_query_content), cuda)
        # batch_query_len = my_utils.gpu(torch.from_numpy(batch_query_len), self._use_cuda)
        batch_query_sources = my_utils.gpu(torch.from_numpy(batch_query_sources), cuda)
        batch_query_chr_src = my_utils.gpu(torch.from_numpy(batch_query_chr_src), cuda)
        batch_query_adj = my_utils.gpu(torch.from_numpy(batch_query_adj), cuda)

        batch_evd_docs = my_utils.gpu(torch.from_numpy(batch_evd_docs), cuda)
        batch_evd_contents = my_utils.gpu(torch.from_numpy(batch_evd_contents), cuda)
        # batch_evd_lens = my_utils.gpu(torch.from_numpy(batch_evd_lens), self._use_cuda)
        batch_evd_sources = my_utils.gpu(torch.from_numpy(batch_evd_sources), cuda)
        batch_evd_cnt_each_query = my_utils.gpu(torch.from_numpy(batch_evd_cnt_each_query), cuda)
        batch_evd_chr_src = my_utils.gpu(torch.from_numpy(batch_evd_chr_src), cuda)

        batch_labels = my_utils.gpu(torch.from_numpy(batch_labels), cuda)
        batch_evd_docs_adj = my_utils.gpu(torch.from_numpy(batch_evd_docs_adj), cuda)
        # total_pairs += self._batch_size * self.
        additional_data = {KeyWordSettings.EvidenceCountPerQuery: batch_evd_cnt_each_query,
                            KeyWordSettings.FCClass.QueryCharSource: batch_query_chr_src,
                            KeyWordSettings.FCClass.DocCharSource: batch_evd_chr_src,
                            KeyWordSettings.Query_Adj: batch_query_adj,
                            KeyWordSettings.Evd_Docs_Adj: batch_evd_docs_adj}
        n=30
        evd_count_per_query = batch_evd_cnt_each_query  # (B, )
        query_char_source = batch_query_chr_src
        doc_char_source = batch_evd_chr_src
        query_adj = batch_query_adj
        evd_docs_adj = batch_evd_docs_adj
        _, L = batch_query_content.size()
        batch_size = batch_query.size(0)
        # prunning at this step to remove padding\
        e_lens, e_conts, q_conts, q_lens, e_adj = [], [], [], [], []
        e_chr_src_conts = []
        expaned_labels = []
        for evd_cnt, q_cont, q_len, evd_lens, evd_doc_cont, evd_chr_src, label, evd_adj in \
                zip(evd_count_per_query, batch_query_content, batch_query_len,
                    batch_evd_lens, batch_evd_contents, doc_char_source, batch_labels, evd_docs_adj):
            evd_cnt = int(torch_utils.cpu(evd_cnt).detach().numpy())
            e_lens.extend(list(evd_lens[:evd_cnt]))
            e_conts.append(evd_doc_cont[:evd_cnt, :])  # stacking later
            e_adj.append(evd_adj[:evd_cnt])
            e_chr_src_conts.append(evd_chr_src[:evd_cnt, :])
            q_lens.extend([q_len] * evd_cnt)
            q_conts.append(q_cont.unsqueeze(0).expand(evd_cnt, L))
            expaned_labels.extend([int(torch_utils.cpu(label).detach().numpy())] * evd_cnt)
        # concat
        e_conts = torch.cat(e_conts, dim=0)  # (n1 + n2 + ..., R)
        e_chr_src_conts = torch.cat(e_chr_src_conts, dim=0)  # (n1 + n2 + ... , R)
        e_adj = torch.cat(e_adj, dim=0)     # (n1 + n2 + ..., R, R)
        e_lens = np.array(e_lens)  # (n1 + n2 + ..., )
        q_conts = torch.cat(q_conts, dim=0)  # (n1 + n2 + ..., R)
        q_lens = np.array(q_lens)
        assert q_conts.size(0) == q_lens.shape[0] == e_conts.size(0) == e_lens.shape[0]

        d_new_indices, d_old_indices = torch_utils.get_sorted_index_and_reverse_index(e_lens)
        e_lens = my_utils.gpu(torch.from_numpy(e_lens), cuda)
        x = batch_query_len
        q_new_indices, q_restoring_indices = torch_utils.get_sorted_index_and_reverse_index(x)
        x = my_utils.gpu(torch.from_numpy(x), cuda)
        # query_lens = my_utils.gpu(torch.from_numpy(query_lens), self._use_cuda)

        additional_paramters = {
            KeyWordSettings.Query_lens: x,  # 每一个query长度
            KeyWordSettings.Doc_lens: batch_evd_lens,
            KeyWordSettings.DocLensIndices: (d_new_indices, d_old_indices, e_lens),
            KeyWordSettings.QueryLensIndices: (q_new_indices, q_restoring_indices, x),
            KeyWordSettings.QuerySources: batch_query_sources,
            KeyWordSettings.DocSources: batch_evd_sources,
            KeyWordSettings.TempLabel: batch_labels,
            KeyWordSettings.DocContentNoPaddingEvidence: e_conts,
            KeyWordSettings.QueryContentNoPaddingEvidence: q_conts,
            KeyWordSettings.EvidenceCountPerQuery: evd_count_per_query,
            KeyWordSettings.FCClass.QueryCharSource: query_char_source,  # (B, 1, L)
            KeyWordSettings.FCClass.DocCharSource: e_chr_src_conts,
            KeyWordSettings.FIXED_NUM_EVIDENCES: n,
            KeyWordSettings.Query_Adj: query_adj,
            KeyWordSettings.Evd_Docs_Adj: e_adj                       # flatten->(n1 + n2 ..., R, R)
        }
        output,logit = fit_model._net.predict(batch_query_content, batch_evd_contents, **additional_paramters)
        logits = torch.cat((logits,logit),dim = 0)
        outputs = torch.cat((outputs,output),dim=0)
        del batch_query_content, batch_query_sources, batch_query_chr_src, batch_query_adj, batch_evd_docs, batch_evd_contents, batch_evd_lens, batch_evd_sources, batch_evd_cnt_each_query, batch_evd_chr_src, batch_labels, batch_evd_docs_adj, additional_paramters
        gc.collect()
        torch.cuda.empty_cache()
        
  return logits,outputs
def process_poli():
    dataset="PolitiFact"
    fixed_length_left=30
    fixed_length_right=100
    log="logs/testget"
    loss_type="cross_entropy"
    batch_size=32
    num_folds=5
    use_claim_source=1
    use_article_source=1
    path="formatted_data/declare/"
    hidden_size=300
    epochs=100
    num_att_heads_for_words=3
    num_att_heads_for_evds=1
    gnn_window_size=3
    lr=0.0001
    gnn_dropout=0.2
    gsl_rate=0.6
    fixed_length_left_src_chars = 20
    fixed_length_right_src_chars = 20
    seed = 123456
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    preprocessor = mz.preprocessors.CharManPreprocessor(fixed_length_left = fixed_length_left,
                                                    fixed_length_right = fixed_length_right,
                                                    fixed_length_left_src = fixed_length_left_src_chars,
                                                    fixed_length_right_src = fixed_length_right_src_chars)
    fixed_length_left=30
    fixed_length_right=100
    log="logs/get"
    batch_size=32
    gnn_window_size=3
    fold_idx=0
    if not os.path.exists(log):
        os.mkdir(log)
    secondary_log_folder = os.path.join(log, "log_results_%s" % (dataset))
    if not os.path.exists(secondary_log_folder):
        os.mkdir(secondary_log_folder)
    secondary_log_folder = secondary_log_folder
    root = os.path.join(os.path.join(path,dataset), "mapped_data")
    tx = time.time()
    kfold_dev_results, kfold_test_results = [], []
    list_metrics = KeyWordSettings.CLS_METRICS
    outfolder_per_fold = os.path.join(secondary_log_folder, "Fold_%s" % fold_idx)
    if not os.path.exists(outfolder_per_fold):
        os.mkdir(outfolder_per_fold)
    logfolder_result_per_fold = os.path.join(outfolder_per_fold, "result_%s.txt" % int(seed))
    file_handler = FileHandlerFC()
    file_handler.init_log_files(logfolder_result_per_fold)
    
    for i in range(5):
        predict_pack = cls_load_data.load_data(root + "/%sfold" % num_folds,  'test_%s' % i, kfolds = num_folds)
        train_pack = cls_load_data.load_data(root + "/%sfold" % num_folds, 'train_%sres' % i, kfolds = num_folds)
        
        # global additional_data
        additional_data = {KeyWordSettings.OutputHandlerFactChecking: file_handler,
                            KeyWordSettings.GNN_Window: gnn_window_size}

        print('parsing data')

        train_processed = preprocessor.fit_transform(train_pack)  # This is a DataPack
        predict_processed = preprocessor.transform(predict_pack)


        train_interactions = ClassificationInteractions(train_processed, **additional_data)
        predict_interactions = ClassificationInteractions(predict_processed, **additional_data)
        file_handler.myprint('done extracting')
        print("Loading word embeddings......")
        t1_emb = time.time()
        term_index = preprocessor.context['vocab_unit'].state['term_index']
        glove_embedding = mz.datasets.embeddings.load_glove_embedding_FC(dimension = 300,
                                                                        term_index = term_index, **additional_data)

        embedding_matrix = glove_embedding.build_matrix(term_index)
        entity_embs1 = entity_embedding.EntityEmbedding(128)
        claim_src_embs_matrix = entity_embs1.build_matrix(preprocessor.context['claim_source_unit'].state['term_index'])

        entity_embs2 = entity_embedding.EntityEmbedding(128)
        article_src_embs_matrix = entity_embs2.build_matrix(preprocessor.context['article_source_unit'].state['term_index'])

        t2_emb = time.time()
        print("Time to load word embeddings......", (t2_emb - t1_emb))
        match_params = {}
        match_params['embedding'] = embedding_matrix
        match_params["num_classes"] = 2
        match_params["fixed_length_right"] = 100
        match_params["fixed_length_left"] = 30

        # for claim source
        match_params["use_claim_source"] = use_claim_source
        match_params["claim_source_embeddings"] = claim_src_embs_matrix
        # for article source
        match_params["use_article_source"] = use_article_source
        match_params["article_source_embeddings"] = article_src_embs_matrix
        # multi-head attentionx
        match_params["cuda"] = 0
        match_params["num_att_heads_for_words"] = num_att_heads_for_words  # first level
        match_params["num_att_heads_for_evds"] = num_att_heads_for_evds  # second level


        match_params['dropout_gnn'] = 0.2
        match_params["dropout_left"] = 0.2
        match_params["dropout_right"] = 0.2
        match_params["hidden_size"] = hidden_size

        match_params["gsl_rate"] = 0.6

        match_params["embedding_freeze"] = True
        match_params["output_size"] = 2 # if args.dataset == "Snopes" else 3
        match_model = graph_based_semantic_structure.Graph_basedSemantiStructure(match_params)
        name_pickle = f'saved_model/Politifacts/poli_{i}'
        match_model.load_state_dict(torch.load(name_pickle,map_location=torch.device('cpu')))
        loss_type = 'cross_entropy'
        epochs = 100
        batch_size = 32
        lr = 0.001
        early_stopping = 10
        outfolder_per_fold = "logs/get/log_results_PolitiFact/Fold_10"
        curr_date = datetime.datetime.now().timestamp()
        fixed_num_evidences = 30
        # file_handler = FileHandlerFC()
        seed = 123456
        fit_model = char_man_fitter_query_repr1.CharManFitterQueryRepr1(net = match_model, loss = loss_type, n_iter = epochs,
                                                        batch_size = batch_size, learning_rate = lr,
                                                        early_stopping = early_stopping, use_cuda = 0,
                                                        logfolder = outfolder_per_fold, curr_date = curr_date,
                                                        fixed_num_evidences = fixed_num_evidences,
                                                        output_handler_fact_checking = file_handler, seed=seed,
                                                        output_size=match_params["output_size"],args="args")
        logits,outputs = get_energy(train_interactions, fit_model)
        # query_ids, left_contents, left_lengths, query_sources, query_char_sources, query_adj, \
        #         evd_docs_ids, evd_docs_contents, evd_docs_lens, evd_sources, evd_cnt_each_query, evd_char_sources, \
        #         pair_labels, evd_docs_adj = fit_model._sampler.get_train_instances_char_man(train_interactions,fit_model.fixed_num_evidences)
        output_name = f'embed_probagation/output_poli_train_{i}.pt'
        logit_name = f'embed_probagation/logit_poli_train_{i}.pt'
        torch.save(outputs, output_name)
        torch.save(logits, logit_name)

        logits,outputs = get_energy(predict_interactions, fit_model)
        output_name = f'embed_probagation/output_poli_test_{i}.pt'
        logit_name = f'embed_probagation/logit_poli_test_{i}.pt'
        torch.save(outputs, output_name)
        torch.save(logits, logit_name)
        if i ==4:
            valid_pack = cls_load_data.load_data(root, 'dev', kfolds = num_folds)
            valid_processed = preprocessor.transform(valid_pack)
            valid_interactions = ClassificationInteractions(valid_processed, **additional_data)
            logits,outputs = get_energy(valid_interactions, fit_model)
            output_name = f'embed_probagation/output_poli_valid.pt'
            logit_name = f'embed_probagation/logit_poli_valid.pt'
            torch.save(outputs, output_name)
            torch.save(logits, logit_name)
    

