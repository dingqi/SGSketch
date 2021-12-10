function [mrr_orders, recall_orders] = eva_link_sketch(nodes_set,graph_current, graph_edge_truth,top_K, embs_cell)
%EVA_CLASSIFICATION_GRAPH Summary of this function goes here
%   Detailed explanation goes here



mrr_orders = zeros(1,length(embs_cell));
recall_orders = zeros(length(top_K),length(embs_cell));

candidates = sub2ind(size(graph_current),...
    [graph_edge_truth(:,1);graph_edge_truth(:,2)],...
    [graph_edge_truth(:,2);graph_edge_truth(:,1)]);


% [embs_cell,~] = sgsketch_node_embs_fast(sparse(graph_current'), K_hash, Rand_beta, order, alpha,0);
% embs_cell = cellfun(@transpose,embs_cell,'UniformOutput',false); % transpose due to the interface between MATLAB and C

for kk=1:length(embs_cell)
    simMat = squareform(1-pdist(embs_cell{kk},@dist_hamming));
    simMat(eye(size(graph_current))==1)=NaN;
    [~,inds] = sort(simMat,'descend','MissingPlacement','last');
    [~,inds2] = sort(inds,'ascend');

    ranks = inds2(candidates);
    mrr_orders(kk) = mean(1./ranks);

    hitTopK = zeros(size(top_K));
    for tt=1:length(top_K)
        hitTopK(tt) = sum(ranks<=top_K(tt));
    end
    recall_orders(:,kk) = hitTopK./length(ranks);
end



end
