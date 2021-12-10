clc;clear;
% embedding size
K_hash = 128;
% order of proximity
order = 4;
% order decay weight
alpha = [0.1];
% gradual forget decay weight
decay_weight = exp(-0.05);
num_node = 200;
num_streaming_edge = 20000;

% recall@10
top_K = [10];


% generate simulated streaming graphs
[edgestream] = streamgraph_simulator_abrupt(num_node, 10, 3, 1/3*2, 0.25, 0, num_streaming_edge);
% [edgestream] = streamgraph_simulator_gradual(num_node, 10, 3, 1/3*2, 0.25, 0.25, 0, num_streaming_edge);


%% simulate streaming graphs
label_current = zeros(num_node,1);
graph_current_forget = zeros(num_node,num_node);

% prepare sketching parameters
Rand_beta = -log(rand(K_hash,num_node)); % pre-generating hash value for efficiency

testing_period_link_prediction = 0.1; % next 10% streaming edges as link prediction ground truth
testing_frequency = 0.02; % evaluate every 2% streaming edges
testing_points = [0.1:testing_frequency:0.9]; % testing points on percentage
testing_points_streams = round(testing_points*size(edgestream,1)); % actual testing points over streams

accuracy_over_time = zeros(length(testing_points_streams), order); % classification accuracy
mrr_over_time = zeros(length(testing_points_streams), 1, order); % link prediction MRR
recall_over_time = zeros(length(testing_points_streams), length(top_K), order); % link prediction Recall

% for undirected graphs, rank edge so that n1<n2, to speedup link prediction task
edgestream(:,1:2) = sort(edgestream(:,1:2),2);

for ii=1:size(edgestream,1)
    edge_current = edgestream(ii,:);
    

    graph_current_forget(edge_current(1),:) = graph_current_forget(edge_current(1),:)*decay_weight;
    graph_current_forget(edge_current(2),:) = graph_current_forget(edge_current(2),:)*decay_weight;
    
    graph_current_forget(edge_current(1),edge_current(2))=1;
    graph_current_forget(edge_current(2),edge_current(1))=1;
    graph_current_forget(edge_current(1),edge_current(1))=1;
    graph_current_forget(edge_current(2),edge_current(2))=1;
    
    if edge_current(3)~=0
        label_current(edge_current(1))=edge_current(3);
    end
    if edge_current(4)~=0
        label_current(edge_current(2))=edge_current(4);
    end
    
    [flag,ind] = ismember(ii,testing_points_streams);
    if flag % evaluating at each testing point
        
        if ind==1 % embedding create 
            [embs,vals] = sgsketch_node_embs_fast(sparse(graph_current_forget'), K_hash, Rand_beta, order, alpha,0);   
            ii_last = ii;
        else % incremental embedding update
            edge_current = edgestream(ii_last+1:ii,:);
            [embs,vals] = sgupdate_node_embs_fast(sparse(graph_current_forget'), K_hash, Rand_beta, order, alpha, embs, vals, edge_current(:,1:2)',0);
            ii_last = ii;
        end
        embs_cell = cellfun(@transpose,embs,'UniformOutput',false); % transpose for evaluation tasks, due to the interface between MATLAB and C
        nodes_set = unique(edgestream(1:ii,1:2)); % nodes observed so far
        
        % evaluating on the node classification task
        if edgestream(1,3)~=0 % when node labels are not empty
            [accs_forget_sgsketch] =  eva_classification_sketch(nodes_set,graph_current_forget,label_current, embs_cell);
            accuracy_over_time(ind,:) = accs_forget_sgsketch;
        end
        
        
        % evaluating on the link prediction task
        graph_edge_truth = edgestream(ii+1:ii+floor(testing_period_link_prediction*size(edgestream,1)),1:2); % get ground truth as the next 10% treaming edges
        [mrr_forget_sgsketch,recall_forget_sgsketch] = eva_link_sketch(nodes_set,graph_current_forget, graph_edge_truth,top_K, embs_cell);
        mrr_over_time(ind,:,:) = mrr_forget_sgsketch;
        recall_over_time(ind,:,:) = recall_forget_sgsketch;
        
        disp(strcat('Evaluating-', num2str(ind),...
            '; Acc=', num2str(accs_forget_sgsketch(order)),...
            '; MRR=', num2str(mrr_forget_sgsketch(order)),...
            '; Rec10=', num2str(recall_forget_sgsketch(order))));
        
    end
    
    
end


