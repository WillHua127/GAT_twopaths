%% experiment settings
network = 'gcn_onepath'; dataset = 'cora';
public = 1; MaxObjectiveEvaluations = 1024;
filename = construct_filename(dataset, network);
lr = optimizableVariable('lr', [1e-8, 1e-2]);
weight_decay = optimizableVariable('weight_decay', [0, 5e-2]);
hidden = optimizableVariable('hidden', [1, 50], 'Type', 'integer');
nb_heads = optimizableVariable('nb_heads', [1, 40], 'Type', 'integer');
dropout = optimizableVariable('dropout', [0, 0.99]);
alpha = optimizableVariable('alpha', [0, 0.7]);
if strcmp(dataset, 'pubmed')
    range_n_blocks = [2, 25];
else
    range_n_blocks = [2, 200];
end
n_blocks = optimizableVariable('n_blocks', range_n_blocks, 'Type', 'integer', 'Optimize', strcmp(network, 'gcn_onepath'));
variables = [lr, weight_decay, hidden, nb_heads, dropout, alpha];
%% environment settings
NumWorkers = 16; create_worker_pool(NumWorkers);
implementation = 'pytorch'; TargetCluster = 'beluga';
%% run experiment
objfunc = @(x)surrogate(x, network, TargetCluster, strcmp(implementation, 'pytorch') && ~strcmp(TargetCluster, 'helios'), dataset, public, implementation); % if to change filename, breakpoint here
clear InitialObjective InitialX;
try
    load(filename);
catch ME
    try
        load(sprintf('%s_initial.mat', network)); InitialObjective = [];
    catch ME
        InitialX = []; InitialObjective = [];
    end
end
results = bayesopt(@(hps)objfunc(hps), variables, 'GPActiveSetSize', 5000, 'InitialX', InitialX, 'InitialObjective', InitialObjective, 'MaxObjectiveEvaluations', MaxObjectiveEvaluations + length(InitialObjective), 'IsObjectiveDeterministic', false, 'AcquisitionFunctionName', 'expected-improvement-plus', 'UseParallel', NumWorkers - 1 > 0);
save_results(filename, InitialX, InitialObjective, results.XTrace, results.ObjectiveTrace);

function filename = construct_filename(dataset, network)
filename = [dataset, '_public'];
filename = [filename, '_', network, '.mat'];
end

function POOL = create_worker_pool(NumWorkers)
if NumWorkers > 1
    POOL = gcp('nocreate');
    if isempty(POOL) || POOL.NumWorkers ~= NumWorkers
        delete(gcp('nocreate')); POOL = parpool('local');
    end
    if POOL.Cluster.NumWorkers ~= NumWorkers
        POOL.Cluster.NumWorkers = NumWorkers; Cluster = POOL.Cluster;
        delete(gcp('nocreate')); POOL = parpool('local', NumWorkers);
    end
end
end
