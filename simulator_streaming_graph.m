function [accuracy_over_time, mrr_over_time, recall_over_time] = simulator_streaming_graph(edgestream, num_node, K_hash,order,alpha,decay_weight,top_K)
%% random graph generation

% num_node = 500;
% [edgTestream, graph, labels] = randgraph_generator(num_node, 100, 3, 1/3*2, 0.2);
% [edgestream] = streamgraph_generator_abrupt(num_node, [50, 100], [3, 4], [1/3*2, 1/4*2], 0.2);
% [edgestream] = streamgraph_generator_gradual(num_node, [50, 100], [3, 4], [1/3*2, 1/4*2], 0.25, 0.2);
% [edgestream] = streamgraph_generator_continuous(num_node, 100, 3, 1/3*2, 0.25, 0.2);

% % updated graph with two clusters of node
% [edgestream,graph] = randgraph_generator(30, 5, 2, 0.8);

% nodes_test = randperm(num_node, round(0.2*num_node));
% nodes_train = setdiff([1:num_node],nodes_test);

% X_train = graph(nodes_train,:);
% Y_train = labels(nodes_train);
% X_test = graph(nodes_test,:);
% Y_test = labels(nodes_test);

% KNN test
% clf = fitcknn(X_train,Y_train, 'NumNeighbors',3, 'Distance', @dist_minmax);
% Y_pred = predict(clf,X_test);
% accs = sum(Y_pred==Y_test)./length(Y_pred);



%% streaming data evaluation
label_current = zeros(num_node,1);
graph_current = eye(num_node,num_node);
graph_current_forget = eye(num_node,num_node);

% prepare sketching parameters
Rand_beta = -log(rand(K_hash,num_node)); % pre-generating hash value for efficiency

testing_period_link_prediction = 0.1; % percentage
testing_frequency = 0.02;
testing_points = [0.1:testing_frequency:1]; % testing points on percentage
testing_points_streams = round(testing_points*size(edgestream,1)); % actual testing points over streams

num_method = 3;
accuracy_over_time = zeros(length(testing_points_streams), order, num_method); % classification accuracy
mrr_over_time = zeros(length(testing_points_streams)-testing_period_link_prediction/testing_frequency,1, order, num_method); % link prediction precision
recall_over_time = zeros(length(testing_points_streams)-testing_period_link_prediction/testing_frequency,length(top_K), order, num_method); % link prediction precision

% rank edge so that n1<n2, to speedup link prediction task
edgestream(:,1:2) = sort(edgestream(:,1:2),2);



for ii=1:size(edgestream,1)
    edge_current = edgestream(ii,:);
    graph_current(edge_current(1),edge_current(2))=1;
    graph_current(edge_current(2),edge_current(1))=1;

    %     graph_current_forget = graph_current_forget*decay_weight;
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
    if flag % evaluating at delai points
        %                 disp('evaluating');
        %                         break;
        nodes_set = unique(edgestream(1:ii,1:2));

        if edgestream(1,3)~=0
%             [accs, acces_sketch] =  eva_classification_graph(nodes_set,graph_current,label_current, order, alpha, true, K_hash, Rand_beta);
            [accs_forget, accs_forget_sketch] = eva_classification_graph(nodes_set,graph_current_forget,label_current, order, alpha, true, K_hash, Rand_beta);
%             [accs_sgsketch] =  eva_classification_sketch(nodes_set,graph_current,label_current, K_hash, Rand_beta, order, alpha);
            [accs_forget_sgsketch] =  eva_classification_sketch(nodes_set,graph_current_forget,label_current, K_hash, Rand_beta, order, alpha);
%             accuracy_over_time(ind,:,:) = [accs,accs_forget,accs_sgsketch,accs_forget_sgsketch,acces_sketch,accs_forget_sketch];
            accuracy_over_time(ind,:,:) = [accs_forget,accs_forget_sgsketch,accs_forget_sketch];
        end

        if ind<=size(mrr_over_time,1)
            %             graph_edge_truth = sub2ind(size(graph_current),...
            %                 edgestream(ii+1:ii+floor(testing_period_link_prediction*size(edgestream,1)),1),...
            %                 edgestream(ii+1:ii+floor(testing_period_link_prediction*size(edgestream,1)),2));

            graph_edge_truth = edgestream(ii+1:ii+floor(testing_period_link_prediction*size(edgestream,1)),1:2);
%             flag1 = ismember(graph_edge_truth,edgestream(1:ii,1:2),'rows');
%             flag2 = ismember(graph_edge_truth,nodes_set);
%             graph_edge_truth = graph_edge_truth(~flag1&flag2(:,1)&flag2(:,2),:);

%             [mrr, recall, mrr_sketch, recall_sketch] = eva_link_graph_dygnn(nodes_set,graph_current,graph_edge_truth,top_K, order, alpha, true, K_hash, Rand_beta);
            [mrr_forget, recall_forget, mrr_forget_sketch, recall_forget_sketch] = eva_link_graph_dygnn(nodes_set,graph_current_forget,graph_edge_truth,top_K, order, alpha, true, K_hash, Rand_beta);
%             [mrr_sgsketch, recall_sgsketch] = eva_link_sketch_dygnn(nodes_set,graph_current, graph_edge_truth,top_K, K_hash, Rand_beta, order, alpha);
            [mrr_forget_sgsketch,recall_forget_sgsketch] = eva_link_sketch_dygnn(nodes_set,graph_current_forget, graph_edge_truth,top_K, K_hash, Rand_beta, order, alpha);
            %
%             mrr_over_time(ind,:,:,:) = cat(3,mrr, mrr_forget, mrr_sgsketch, mrr_forget_sgsketch, mrr_sketch, mrr_forget_sketch);
%             recall_over_time(ind,:,:,:) = cat(3,recall, recall_forget, recall_sgsketch, recall_forget_sgsketch, recall_sketch, recall_forget_sketch);
            mrr_over_time(ind,:,:,:) = cat(3, mrr_forget, mrr_forget_sgsketch, mrr_forget_sketch);
            recall_over_time(ind,:,:,:) = cat(3,recall_forget, recall_forget_sgsketch, recall_forget_sketch);

        end

        %         disp(strcat('evaluating-', num2str(ii), ';accuracy-', num2str(accs), ';precision-', num2str(pres)));

    end

end
