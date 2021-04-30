function [squared_LP_metric, LP_metric_loc, LP_metric_miss, LP_metric_fal, LP_metric_switch]=ComputeLP_metric_all_error(X_estimate,t_b_estimate, length_estimate,X_truth,t_birth,t_death,c_gospa,gamma_track_metric,k,Nx)

%We calculate the error of the estimated set of all trajectories at the current time step normalised
%by the current time window, see, for example, (74) in
%Á. F. García-Fernández, L. Svensson, J. L. Williams, Y. Xia, K. Granström,
%"Trajectory Poisson multi-Bernoulli filters," accepted in IEEE-TSP.



%%We use the LP Trajectory metric for all trajectories up to the current
%%time step
X_multi_abu_format_prov=reshape(X_truth(:,1:k),Nx,size(X_truth,1)/Nx,k);
X_multi_abu_format= permute(X_multi_abu_format_prov,[1,3,2]);


%We construct the structures of X and Y
% X = struct('xState', [], 'tVec', [], 'iVec', []);
% Y = struct('xState', [], 'tVec', [], 'iVec', []);

%First X

index_alive=logical(sum(X_truth(:,1:k)~=0,2));
index_alive=index_alive(1:4:end);
X_multi_alive=X_multi_abu_format(:,1:k,index_alive);

X.xState = X_multi_alive;
X.tVec = t_birth(index_alive)';
X.iVec = min((t_death(index_alive)-1)',k)-t_birth(index_alive)'+1;

%Second Y
Y.tVec=t_b_estimate';
Y.iVec=length_estimate';
Y.xState=zeros(Nx,k,length(X_estimate));

for i=1:length(X_estimate)
    Y.xState(:,t_b_estimate(i):t_b_estimate(i)+length_estimate(i)-1,i)=reshape(X_estimate{i},Nx,length_estimate(i));
end

%We only consider the position elements to compute the error
X.xState=X.xState([1,3],:,:);
Y.xState=Y.xState([1,3],:,:);

[dxy, loc_cost, miss_cost, fa_cost, switch_cost] = LPTrajMetric_cluster(X, Y, c_gospa, 2, gamma_track_metric);


squared_LP_metric=dxy^2/k; %normalised by k
LP_metric_loc=sum(loc_cost)/k;
LP_metric_miss=sum(miss_cost)/k;
LP_metric_fal=sum(fa_cost)/k;
LP_metric_switch=sum(switch_cost)/k;