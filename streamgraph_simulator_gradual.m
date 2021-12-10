function [edgestream, graph1, graph2] = streamgraph_generator_gradual2(num_node, deg, num_clusters, incluster_ratio, drift_point, overlap_rate, noise_level, num_streaming_edge)
%RANDGRAPH_GENERATOR Summary of this function goes here
%   Detailed explanation goes here

% num_node = 100;
% deg = 10;
% num_clusters = 3;
% incluster_ratio = 1/3*2;
% noise_level = 0;
% drift_point = 0.25;
% overlap_rate = 0.25;
% num_streaming_edge = 20000;

num_edge = num_node*deg/2;
nodes = [1:num_node]';
threshold = num_node/num_clusters(1);
labels1 = ceil(nodes./threshold);
labels2 = ceil(rem(nodes+threshold/2,num_node)./threshold);


% generate initial graph
edges_cand = unique(randi(length(nodes),num_edge*100,2),'rows');
edges_cand = edges_cand(edges_cand(:,2)>edges_cand(:,1),:);

inds = find(labels1(edges_cand(:,1))==labels1(edges_cand(:,2)));
inds_in = inds(randperm(length(inds),round(incluster_ratio*num_edge)));

inds = find(labels1(edges_cand(:,1))~=labels1(edges_cand(:,2)));
inds_out = inds(randperm(length(inds),round((1-incluster_ratio)*num_edge)));

edges1 = edges_cand([inds_in;inds_out],:);
edges1 = edges1(randperm(size(edges1,1)),:);
edges2 = edges1;

graph_upper = sparse(edges1(:,1),edges1(:,2),ones(size(edges1(:,1))),num_node,num_node);
graph1 = full(graph_upper+graph_upper'+speye(size(graph_upper)));

% generate target graph
% mask = zeros(size(graph_ini));
% mask(labels_ini~=labels_end,:)=1;
% mask(:,labels_ini~=labels_end)=1;
% 
% updated_edge_number = length(find(mask.*graph_ini));

num_in = sum((labels2(edges1(:,1))==labels2(edges1(:,2))));
num_out = sum((labels2(edges1(:,1))~=labels2(edges1(:,2))));
num_move = round((num_out*2-num_in)/3);

temp = find( (labels1(edges1(:,1))==labels1(edges1(:,2))) & (labels2(edges1(:,1))~=labels2(edges1(:,2))) );
edges2(temp(randperm(length(temp),num_move)),:)=[];

edges_cand = unique(randi(length(nodes),num_move*100,2),'rows');
edges_cand = edges_cand(edges_cand(:,2)>edges_cand(:,1),:);
edges_cand = setdiff(edges_cand,edges2,'rows');
inds = find(labels2(edges_cand(:,1))==labels2(edges_cand(:,2)));
inds_in = inds(randperm(length(inds),num_move));
edges2 = [edges2;edges_cand(inds_in,:)];

% 
% num_in = sum((labels2(edges2(:,1))==labels2(edges2(:,2))));
% num_out = sum((labels2(edges2(:,1))~=labels2(edges2(:,2))));


graph_upper = sparse(edges2(:,1),edges2(:,2),ones(size(edges2(:,1))),num_node,num_node);
graph2 = full(graph_upper+graph_upper'+speye(size(graph_upper)));

% sum(graph_ini(:))
% sum(graph_end(:))
% sum(sum(graph_ini~=graph_end))

%% first stream

nodes_label_flip = nodes(randperm(num_node,round(num_node*noise_level)));
labels1(nodes_label_flip) = rem(labels1(nodes_label_flip)-1+randi(num_clusters-1,length(nodes_label_flip),1),num_clusters)+1; % shift node labels

edgestream1 = edges1(randi(size(edges1,1), num_streaming_edge,1),:);

edgestream1 = [edgestream1,labels1(edgestream1)];


%% second stream

nodes_label_flip = nodes(randperm(num_node,round(num_node*noise_level)));
labels2(nodes_label_flip) = rem(labels2(nodes_label_flip)-1+randi(num_clusters-1,length(nodes_label_flip),1),num_clusters)+1; % shift node labels

edgestream2 = edges2(randi(size(edges2,1), num_streaming_edge,1),:);

edgestream2 = [edgestream2,labels2(edgestream2)];

%% gradual transition


offset = num_streaming_edge*overlap_rate;
offset1 = round(num_streaming_edge*drift_point);
offset2 = round(num_streaming_edge*drift_point+offset);

rand_samples = rand(offset2-offset1+1,1)*offset;
nodes_shuffled = randperm(num_node);

edgestream = zeros(size(edgestream1));
counter1 = 1; counter2=1;
labels = labels1;
cc1 = 0; cc2 = 0;
for ii=1:num_streaming_edge
%     disp(ii);
    if ii<=offset1
%         disp('stream1');
        edgestream(ii,:) = edgestream1(counter1,:);
        counter1 = counter1+1;
    elseif ii>offset2
%         disp('stream2');
        edgestream(ii,:) = edgestream2(counter2,:);
        counter2 = counter2+1;
    elseif ((ii>offset1) && (ii<=offset2))
%         break
        inds_end = ceil(((ii-offset1)/(offset2-offset1))*num_node);
        labels(nodes_shuffled(1:inds_end)) = labels2(nodes_shuffled(1:inds_end));
        
        if (rand_samples(ii-offset1)>ii-offset1)
%             disp('stream1')
            cc1 = cc1+1;
            edgestream(ii,:) = [edgestream1(counter1,1:2), labels(edgestream1(counter1,1:2))'];
            counter1 = counter1+1;
        else
%             disp('stream2')
            cc2 = cc2+1;
            edgestream(ii,:) = [edgestream2(counter2,1:2), labels(edgestream2(counter2,1:2))'];
            counter2 = counter2+1;
        end
%         ii = ii+1;
%         disp([ ii, length(unique(edgestream(1:ii,1:2),'rows'))]);
%         disp(sum(labels~=labels2));

    end
end




end

