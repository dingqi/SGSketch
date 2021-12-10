function [edgestream, graph1, graph2] = streamgraph_generator_abrupt2(num_node, deg, num_clusters, incluster_ratio, drift_point, noise_level, num_streaming_edge)
%RANDGRAPH_GENERATOR Summary of this function goes here
%   Detailed explanation goes here

% num_node = 100;
% deg = 10;
% num_clusters = 3;
% incluster_ratio = 1/3*2;
% noise_level = 0;
% drift_point = 0.25;
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

edgestream1 = edges1(randi(size(edges1,1), round(num_streaming_edge*drift_point),1),:);

edgestream1 = [edgestream1,labels1(edgestream1)];


%% second stream

nodes_label_flip = nodes(randperm(num_node,round(num_node*noise_level)));
labels2(nodes_label_flip) = rem(labels2(nodes_label_flip)-1+randi(num_clusters-1,length(nodes_label_flip),1),num_clusters)+1; % shift node labels

edgestream2 = edges2(randi(size(edges2,1), num_streaming_edge-round(num_streaming_edge*drift_point),1),:);

edgestream2 = [edgestream2,labels2(edgestream2)];

%%
edgestream = [edgestream1;edgestream2];


end

