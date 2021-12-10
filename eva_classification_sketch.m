function [accuracy_orders] = eva_classification_sketch(nodes_set,graph_current,label_current, embs_cell)
%EVA_CLASSIFICATION_GRAPH Summary of this function goes here
%   Detailed explanation goes here

% [embs_cell,~] = sgsketch_node_embs_fast(sparse(graph_current'), K_hash, Rand_beta, order, alpha,0);
% embs_cell = cellfun(@transpose,embs_cell,'UniformOutput',false); % transpose due to the interface between MATLAB and C


flag = (label_current~=0);
accuracy_orders = zeros(length(embs_cell),1);

for kk=1:length(embs_cell)
    clf = fitcknn(embs_cell{kk}(flag,:),label_current(flag,:), 'NumNeighbors',5, 'Distance', @dist_hamming,'Kfold',5);    
    accuracy_orders(kk) = 1 - kfoldLoss(clf);
end



end

